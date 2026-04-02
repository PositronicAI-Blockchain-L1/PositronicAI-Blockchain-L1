"""
Positronic - LSTM + Temporal Attention Network
===============================================

Monitors network-wide metrics time series to detect coordinated attacks
as part of the Economic Stability Guardian (ESG) subsystem.

Architecture
------------
    LSTM(input=9, hidden=32, layers=2)
        -> Temporal Attention (Bahdanau-style)
        -> Dense(32, 16) + GELU
        -> Dense(16, 5) + Sigmoid
        -> [volume_spike, gas_manip, spam, coordinated, congestion]

Input Features (9 per timestep)
-------------------------------
    tx_rate              - transactions per second
    avg_value            - average transaction value
    total_volume         - total value transferred
    unique_senders       - distinct sender addresses
    unique_recipients    - distinct recipient addresses
    avg_gas_price        - mean gas price across transactions
    mempool_size         - pending transaction count
    block_fullness       - block utilization ratio [0, 1]
    validator_participation - fraction of active validators

Output (5 attack probabilities)
-------------------------------
    Each output is an independent sigmoid probability in [0, 1]:
        0 - volume_spike       : abnormal transaction volume surge
        1 - gas_manipulation   : gas price manipulation detected
        2 - spam_flood         : high-frequency low-value spam
        3 - coordinated        : multi-actor coordinated attack
        4 - congestion         : deliberate network congestion
"""

import numpy as np
from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout, LSTM as LSTMLayer
from positronic.ai.engine.activations import GELU, Sigmoid, Tanh


class TemporalAttention(Model):
    """
    Bahdanau-style temporal attention over LSTM hidden states.

    For each timestep *t* in the sequence the mechanism computes an
    alignment score, normalises across the sequence via softmax, and
    produces a single fixed-length context vector as the weighted sum
    of all hidden states.

    Equations
    ---------
        score_t = v^T * tanh(W_h * h_t + b)
        alpha_t = softmax(score_t)          -- over t
        context = sum_t(alpha_t * h_t)

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the LSTM hidden states that will be attended
        over.  Both the projection matrix ``W_h`` and the score vector
        ``v`` are sized according to this value.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = Dense(hidden_size, hidden_size)
        self.v = Tensor(
            np.random.randn(hidden_size, 1) * 0.1,
            requires_grad=True,
        )
        self.tanh = Tanh()

    def forward(self, lstm_outputs: Tensor) -> Tensor:
        """
        Compute the attention-weighted context vector.

        Parameters
        ----------
        lstm_outputs : Tensor
            Shape ``(batch, seq_len, hidden_size)`` -- the full sequence
            of LSTM hidden states.

        Returns
        -------
        Tensor
            Shape ``(batch, hidden_size)`` -- a single context vector per
            batch element summarising the most relevant timesteps.
        """
        batch_size = lstm_outputs.data.shape[0]
        seq_len = lstm_outputs.data.shape[1]
        hidden_size = lstm_outputs.data.shape[2]

        # -- Flatten sequence into (batch*seq_len, hidden_size) for the
        #    dense projection.  A custom backward propagates gradients
        #    back into the original 3-D tensor.
        reshaped = Tensor(
            lstm_outputs.data.reshape(-1, hidden_size),
            requires_grad=lstm_outputs.requires_grad,
            _children=(lstm_outputs,),
        )

        def _reshape_bwd():
            if lstm_outputs.requires_grad:
                lstm_outputs.grad = (
                    lstm_outputs.grad
                    + reshaped.grad.reshape(batch_size, seq_len, hidden_size)
                )

        reshaped._backward = _reshape_bwd

        # -- Project and score: score = v^T * tanh(W_h * h_t) -----------
        projected = self.W_h.forward(reshaped)      # (batch*seq_len, hidden)
        activated = self.tanh(projected)             # (batch*seq_len, hidden)
        scores = activated @ self.v                  # (batch*seq_len, 1)

        # Reshape scores to (batch, seq_len) for softmax over the time
        # dimension.
        scores_2d = Tensor(
            scores.data.reshape(batch_size, seq_len),
            requires_grad=True,
            _children=(scores,),
        )

        def _scores_bwd():
            if scores.requires_grad:
                scores.grad = scores.grad + scores_2d.grad.reshape(-1, 1)

        scores_2d._backward = _scores_bwd

        # -- Softmax (numerically stable) over sequence dimension -------
        exp_scores = np.exp(
            scores_2d.data - np.max(scores_2d.data, axis=1, keepdims=True)
        )
        alpha = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)

        # -- Context = weighted sum of hidden states --------------------
        alpha_expanded = alpha[:, :, np.newaxis]  # (batch, seq_len, 1)
        context_data = np.sum(
            lstm_outputs.data * alpha_expanded, axis=1
        )  # (batch, hidden)

        context = Tensor(
            context_data,
            requires_grad=True,
            _children=(lstm_outputs, scores_2d),
        )

        def _ctx_bwd():
            if lstm_outputs.requires_grad:
                # Gradient of the weighted sum w.r.t. hidden states.
                lstm_outputs.grad = (
                    lstm_outputs.grad
                    + context.grad[:, np.newaxis, :] * alpha_expanded
                )
            if scores_2d.requires_grad:
                # Chain through softmax: d_context/d_alpha * d_alpha/d_scores.
                d_alpha = np.sum(
                    lstm_outputs.data * context.grad[:, np.newaxis, :], axis=2
                )
                # Softmax Jacobian: diag(alpha) - alpha * alpha^T applied
                # to the incoming gradient vector.
                s = np.sum(d_alpha * alpha, axis=1, keepdims=True)
                d_scores = alpha * (d_alpha - s)
                scores_2d.grad = scores_2d.grad + d_scores

        context._backward = _ctx_bwd

        return context


class LSTMAttentionNet(Model):
    """
    LSTM + Temporal Attention network for network stability monitoring.

    This model ingests a sliding window of ``NetworkMetrics`` snapshots
    and outputs independent probabilities for five categories of network
    attack.  It is the neural backbone of the Economic Stability
    Guardian (ESG) component inside the Positronic AI Validation Gate.

    Parameters
    ----------
    input_size : int, default 9
        Number of features per timestep (see module docstring).
    hidden_size : int, default 32
        LSTM hidden / cell state dimensionality.
    num_layers : int, default 2
        Number of stacked LSTM layers.

    Attributes
    ----------
    VOLUME_SPIKE : int
        Output index for volume spike detection.
    GAS_MANIPULATION : int
        Output index for gas price manipulation.
    SPAM_FLOOD : int
        Output index for spam / dust flooding.
    COORDINATED : int
        Output index for coordinated multi-actor attacks.
    CONGESTION : int
        Output index for deliberate congestion attacks.
    NUM_ATTACKS : int
        Total number of attack categories (5).
    METRICS_DIM : int
        Expected input feature dimensionality (9).
    """

    # -- Attack type indices --------------------------------------------
    VOLUME_SPIKE = 0
    GAS_MANIPULATION = 1
    SPAM_FLOOD = 2
    COORDINATED = 3
    CONGESTION = 4
    NUM_ATTACKS = 5
    METRICS_DIM = 9

    def __init__(
        self,
        input_size: int = 9,
        hidden_size: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked LSTM layers -- each feeds into the next.
        self.lstm_layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.lstm_layers.append(LSTMLayer(in_size, hidden_size))

        self.dropout = Dropout(0.1)

        # Temporal attention mechanism
        self.attention = TemporalAttention(hidden_size)

        # Classification head
        self.fc1 = Dense(hidden_size, hidden_size // 2)
        self.act = GELU()
        self.fc2 = Dense(hidden_size // 2, self.NUM_ATTACKS)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the full forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, input_size)`` -- a batch of metric
            time series.

        Returns
        -------
        Tensor
            Shape ``(batch, NUM_ATTACKS)`` -- independent attack
            probabilities in [0, 1].
        """
        h = x
        for lstm in self.lstm_layers:
            h, _ = lstm.forward(h)
            h = self.dropout.forward(h)

        # Temporal attention collapses the sequence dimension.
        context = self.attention.forward(h)  # (batch, hidden_size)

        # Classification head
        out = self.fc1.forward(context)
        out = self.act(out)
        out = self.fc2.forward(out)
        out = self.sigmoid(out)

        return out

    def score(self, metrics_sequence: np.ndarray) -> float:
        """
        Score a single metrics sequence for overall network risk.

        This is the primary inference entry point used by the ESG at
        runtime.  The five attack probabilities are combined into a
        single scalar via a fixed importance weighting.

        Parameters
        ----------
        metrics_sequence : np.ndarray
            Shape ``(seq_len, 9)`` or ``(1, seq_len, 9)``.

        Returns
        -------
        float
            Aggregate risk score in [0, 1].
        """
        if metrics_sequence.ndim == 2:
            metrics_sequence = metrics_sequence[np.newaxis, :]

        self.eval()
        x = Tensor(metrics_sequence.astype(np.float32), requires_grad=False)
        probs = self.forward(x)
        self.train()

        # Importance-weighted combination of attack probabilities.
        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        return float(
            np.sum(probs.data.flatten()[: self.NUM_ATTACKS] * weights)
        )

    def train_step(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        optimizer,
    ) -> float:
        """
        Execute one gradient-descent training step.

        Parameters
        ----------
        sequences : np.ndarray
            Shape ``(batch, seq_len, input_size)`` -- batched metric
            time series.
        labels : np.ndarray
            Shape ``(batch, NUM_ATTACKS)`` -- binary ground-truth labels
            for each attack category.
        optimizer
            Any optimizer exposing ``zero_grad()`` and ``step()``.

        Returns
        -------
        float
            Scalar binary cross-entropy loss for this step.
        """
        optimizer.zero_grad()

        x = Tensor(sequences.astype(np.float32), requires_grad=False)
        target = Tensor(labels.astype(np.float32), requires_grad=False)

        pred = self.forward(x)

        # -- Binary cross-entropy loss ---------------------------------
        eps = 1e-7
        p = np.clip(pred.data, eps, 1 - eps)
        loss_val = -np.mean(
            target.data * np.log(p) + (1 - target.data) * np.log(1 - p)
        )

        loss = Tensor(
            np.array(loss_val), requires_grad=True, _children=(pred,)
        )

        def _bwd():
            n = pred.data.size
            grad = (
                -target.data / np.clip(pred.data, eps, None)
                + (1 - target.data) / np.clip(1 - pred.data, eps, None)
            ) / n
            if pred.requires_grad:
                pred.grad = pred.grad + loss.grad * grad

        loss._backward = _bwd

        loss.backward()
        optimizer.step()

        return float(loss.data)

    @staticmethod
    def metrics_to_array(metrics_list) -> np.ndarray:
        """
        Convert a list of ``NetworkMetrics`` objects to a numpy array.

        Parameters
        ----------
        metrics_list : list
            Sequence of ``NetworkMetrics`` instances, each exposing the
            nine standard metric attributes.

        Returns
        -------
        np.ndarray
            Shape ``(len(metrics_list), 9)`` with dtype ``float32``.
            Returns a zero array of shape ``(1, 9)`` when the input
            list is empty.
        """
        features = []
        for m in metrics_list:
            features.append([
                m.tx_rate,
                m.avg_value,
                float(m.total_volume),
                float(m.unique_senders),
                float(m.unique_recipients),
                m.avg_gas_price,
                float(m.mempool_size),
                m.block_fullness,
                m.validator_participation,
            ])
        if features:
            return np.array(features, dtype=np.float32)
        return np.zeros((1, 9), dtype=np.float32)
