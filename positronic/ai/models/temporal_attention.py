"""
Positronic - Temporal Attention Network for MEV/Sandwich Attack Detection (MSAD)

Detects MEV/sandwich attacks by analyzing transaction sequences from the mempool.

Architecture:
    Positional Encoding -> Dense Embedding -> 2x TransformerBlock -> GlobalPool -> Classifier

Each TransformerBlock consists of:
    - Multi-Head Self-Attention (4 heads)
    - LayerNorm + Residual Connection
    - Feed-Forward Network (GELU activation)
    - LayerNorm + Residual Connection

The classifier outputs per-attack-type probabilities via sigmoid activation,
enabling multi-label detection of: sandwich, front-run, back-run, and gas manipulation.
"""

import numpy as np
from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout, MultiHeadAttention
from positronic.ai.engine.activations import GELU, Sigmoid
from positronic.ai.engine.functional import positional_encoding


class TransformerBlock(Model):
    """
    Single transformer encoder block with pre-norm residual connections.

    Implements the standard transformer encoder layer:
        1. Multi-head self-attention with dropout and residual + LayerNorm
        2. Position-wise feed-forward network with dropout and residual + LayerNorm

    Parameters
    ----------
    d_model : int
        Dimensionality of the model (input and output size). Default: 64.
    num_heads : int
        Number of parallel attention heads. Default: 4.
    d_ff : int
        Dimensionality of the inner feed-forward layer. Default: 128.
    dropout : float
        Dropout probability applied after attention and feed-forward. Default: 0.1.
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff1 = Dense(d_model, d_ff)
        self.ff2 = Dense(d_ff, d_model)
        self.activation = GELU()
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Forward pass through the transformer block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, seq_len, d_model).
        mask : optional
            Attention mask. Positions with mask value 0 are ignored.

        Returns
        -------
        Tensor
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Self-attention with residual connection and layer normalization
        attn_out = self.attention.forward(x, x, x, mask)
        attn_out = self.dropout1.forward(attn_out)
        x = self.norm1.forward(x + attn_out)

        # Feed-forward network with residual connection and layer normalization
        ff_out = self.ff1.forward(x)
        ff_out = self.activation(ff_out)
        ff_out = self.ff2.forward(ff_out)
        ff_out = self.dropout2.forward(ff_out)
        x = self.norm2.forward(x + ff_out)

        return x


class TemporalAttentionNet(Model):
    """
    Temporal Attention Network for MEV/Sandwich Attack Detection.

    Processes sequences of transaction features extracted from the mempool to detect
    sandwich attacks, front-running, back-running, and gas price manipulation patterns.

    The network embeds raw transaction features into a latent space, applies sinusoidal
    positional encoding to preserve ordering information, and passes the sequence through
    stacked transformer encoder blocks. A global average pooling layer aggregates the
    sequence into a fixed-size representation, which is then classified into per-attack-type
    probabilities via a sigmoid-activated head (multi-label output).

    Architecture
    ------------
        Input: (batch, seq_len, tx_features)
        -> Dense(tx_features, d_model=64) embedding
        -> + Sinusoidal Positional Encoding
        -> 2x TransformerBlock(d_model=64, heads=4, d_ff=128)
        -> Global Average Pooling over sequence dimension
        -> Dense(64, 32) + GELU
        -> Dense(32, num_attack_types=4) + Sigmoid
        -> Attack probabilities: [sandwich, frontrun, backrun, gas_manip]

    Parameters
    ----------
    tx_feature_dim : int
        Number of features per transaction in the input sequence. Default: 10.
    d_model : int
        Internal model dimensionality. Default: 64.
    num_heads : int
        Number of attention heads per transformer block. Default: 4.
    num_layers : int
        Number of stacked transformer encoder blocks. Default: 2.
    max_seq_len : int
        Maximum supported sequence length for positional encoding. Default: 50.

    Attributes
    ----------
    SANDWICH : int
        Index 0 in the output vector, corresponding to sandwich attack probability.
    FRONTRUN : int
        Index 1 in the output vector, corresponding to front-running probability.
    BACKRUN : int
        Index 2 in the output vector, corresponding to back-running probability.
    GAS_MANIP : int
        Index 3 in the output vector, corresponding to gas manipulation probability.
    NUM_ATTACKS : int
        Total number of attack types detected (4).
    """

    # Attack type indices for the output probability vector
    SANDWICH = 0
    FRONTRUN = 1
    BACKRUN = 2
    GAS_MANIP = 3
    NUM_ATTACKS = 4

    def __init__(
        self,
        tx_feature_dim: int = 10,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input embedding: project raw tx features to model dimension
        self.input_embed = Dense(tx_feature_dim, d_model)

        # Fixed sinusoidal positional encoding (not learned)
        self._pos_encoding = positional_encoding(max_seq_len, d_model)

        # Stack of transformer encoder blocks
        self.transformer_blocks = []
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerBlock(d_model, num_heads, d_ff=d_model * 2)
            )

        # Classification head
        self.classifier_fc1 = Dense(d_model, d_model // 2)
        self.classifier_act = GELU()
        self.classifier_fc2 = Dense(d_model // 2, self.NUM_ATTACKS)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, mask=None) -> Tensor:
        """
        Forward pass through the temporal attention network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, seq_len, tx_feature_dim) containing
            transaction feature sequences from the mempool.
        mask : optional
            Attention mask for the transformer blocks. Positions with 0 are masked out.

        Returns
        -------
        Tensor
            Output tensor of shape (batch, NUM_ATTACKS) containing per-attack-type
            probabilities in [0, 1]. Indices correspond to SANDWICH, FRONTRUN,
            BACKRUN, and GAS_MANIP class attributes.
        """
        batch_size = x.data.shape[0]
        seq_len = x.data.shape[1]

        # Embed transaction features into model dimension
        h = self.input_embed.forward(x)

        # Add sinusoidal positional encoding (broadcast across batch)
        pos_enc = self._pos_encoding[:seq_len]
        h = h + Tensor(np.broadcast_to(pos_enc, (batch_size, seq_len, self.d_model)))

        # Pass through transformer encoder blocks
        for block in self.transformer_blocks:
            h = block.forward(h, mask)

        # Global average pooling over the sequence dimension
        # h shape: (batch, seq_len, d_model) -> pooled shape: (batch, d_model)
        pooled = Tensor(
            np.mean(h.data, axis=1),
            requires_grad=h.requires_grad,
            _children=(h,),
        )

        def _pool_backward():
            if h.requires_grad:
                h.grad = h.grad + np.broadcast_to(
                    pooled.grad[:, np.newaxis, :] / seq_len, h.data.shape
                )

        pooled._backward = _pool_backward

        # Classification head: Dense -> GELU -> Dense -> Sigmoid
        out = self.classifier_fc1.forward(pooled)
        out = self.classifier_act(out)
        out = self.classifier_fc2.forward(out)
        out = self.sigmoid(out)

        return out

    def score(self, tx_sequence: np.ndarray) -> float:
        """
        Score a transaction sequence for MEV attack risk.

        Convenience method that runs a forward pass in eval mode and returns
        the maximum attack probability as a single scalar risk score.

        Parameters
        ----------
        tx_sequence : np.ndarray
            Transaction feature array of shape (seq_len, tx_feature_dim) or
            (1, seq_len, tx_feature_dim). Each row is a transaction with 10 features.

        Returns
        -------
        float
            Risk score in [0, 1], where higher values indicate greater MEV risk.
            Computed as the maximum probability across all attack types.
        """
        if tx_sequence.ndim == 2:
            tx_sequence = tx_sequence[np.newaxis, :]

        self.eval()
        x = Tensor(tx_sequence, requires_grad=False)
        probs = self.forward(x)
        self.train()

        # Return the highest attack probability as the overall risk score
        return float(np.max(probs.data))

    def train_step(self, sequences: np.ndarray, labels: np.ndarray, optimizer) -> float:
        """
        Execute a single training step with binary cross-entropy loss.

        Parameters
        ----------
        sequences : np.ndarray
            Batch of transaction sequences, shape (batch, seq_len, tx_feature_dim).
        labels : np.ndarray
            Binary ground-truth labels, shape (batch, NUM_ATTACKS). Each element is
            0 or 1 indicating absence or presence of the corresponding attack type.
        optimizer : object
            Optimizer instance with zero_grad() and step() methods.

        Returns
        -------
        float
            Scalar binary cross-entropy loss value for this batch.
        """
        optimizer.zero_grad()

        x = Tensor(sequences, requires_grad=False)
        target = Tensor(labels, requires_grad=False)

        pred = self.forward(x)

        # Clamp predictions to avoid log(0) in BCE computation
        eps = 1e-7
        pred_clamp = Tensor(
            np.clip(pred.data, eps, 1 - eps),
            requires_grad=True,
            _children=(pred,),
        )

        def _clamp_bwd():
            if pred.requires_grad:
                mask = (pred.data >= eps) & (pred.data <= 1 - eps)
                pred.grad = pred.grad + pred_clamp.grad * mask.astype(float)

        pred_clamp._backward = _clamp_bwd

        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        loss_data = -(
            target.data * np.log(pred_clamp.data)
            + (1 - target.data) * np.log(1 - pred_clamp.data)
        )
        loss = Tensor(np.mean(loss_data), requires_grad=True, _children=(pred_clamp,))

        def _loss_bwd():
            n = pred_clamp.data.size
            grad = (
                -target.data / pred_clamp.data
                + (1 - target.data) / (1 - pred_clamp.data)
            ) / n
            if pred_clamp.requires_grad:
                pred_clamp.grad = pred_clamp.grad + loss.grad * grad

        loss._backward = _loss_bwd

        loss.backward()
        optimizer.step()

        return float(loss.data)

    @staticmethod
    def extract_sequence_features(tx, pending_txs, max_seq_len=50):
        """
        Extract a feature matrix from a target transaction and pending mempool transactions.

        Builds a (seq_len, 10) feature matrix from the most recent pending transactions,
        where each row encodes one transaction relative to the target transaction.

        Parameters
        ----------
        tx : Transaction
            The target transaction being analyzed for MEV risk. Used as a reference
            for computing relative features (gas ratio, value ratio, address matching).
        pending_txs : list of Transaction
            List of pending transactions from the mempool, ordered chronologically.
            Only the last ``max_seq_len`` transactions are used.
        max_seq_len : int
            Maximum number of transactions to include in the sequence. Default: 50.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (seq_len, 10) with dtype float32. If no pending
            transactions are available, returns a zero array of shape (1, 10).

        Notes
        -----
        The 10 features per transaction are:
            0. value_log: log1p of the transaction value
            1. gas_price_log: log1p of the gas price
            2. is_contract: 1.0 if tx_type is CONTRACT_CREATE (3) or CONTRACT_CALL (4)
            3. same_sender: 1.0 if the sender matches the target transaction
            4. same_recipient: 1.0 if the recipient matches the target transaction
            5. gas_ratio: pending tx gas price / target tx gas price
            6. value_ratio: pending tx value / target tx value
            7. time_delta: time difference (placeholder, 0.0)
            8. nonce: transaction nonce
            9. tx_type: transaction type code
        """
        features = []
        for ptx in pending_txs[-max_seq_len:]:
            feat = [
                np.log1p(float(ptx.value)),
                np.log1p(float(ptx.gas_price)),
                float(ptx.tx_type in (3, 4)),  # CONTRACT_CREATE, CONTRACT_CALL
                float(ptx.sender == tx.sender),
                float(ptx.recipient == tx.recipient),
                float(ptx.gas_price) / max(float(tx.gas_price), 1.0),
                float(ptx.value) / max(float(tx.value), 1.0),
                0.0,  # time delta (would need timestamps)
                float(ptx.nonce),
                float(ptx.tx_type),
            ]
            features.append(feat)

        if not features:
            return np.zeros((1, 10), dtype=np.float32)

        return np.array(features, dtype=np.float32)
