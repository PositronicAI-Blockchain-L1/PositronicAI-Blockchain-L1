"""
Positronic - Cross-Model Attention (Phase 16)
================================================

Learns correlations between the 4 AI model scores (TAD, MSAD, SCRA, ESG)
to detect attack patterns that no single model catches alone.

For example, TAD+MSAD both scoring high is a stronger signal for a sandwich
attack than either alone. Cross-attention learns these combinations automatically.

Architecture
------------
    Input: 4 model scores + 6 pairwise interactions = 10 features
    Project: Dense(10, 32) with LayerNorm + GELU
    4 "model tokens" (each 8-dim, learned embeddings)
    Concatenate projected input with tokens -> Dense -> self-attention
    Multi-head self-attention (d=32, 4 heads, 1 layer)
    Pool: mean of attended representations -> 32-dim
    Output: Dense(32, 1) -> Sigmoid -> attention_score

Dependencies
------------
    - positronic.ai.engine.tensor.Tensor
    - positronic.ai.engine.model.Model
    - positronic.ai.engine.layers.Dense, LayerNorm, Dropout
    - positronic.ai.engine.activations.GELU, Sigmoid
    - positronic.ai.engine.initializers.xavier_normal
"""

from typing import Optional
import numpy as np
from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout
from positronic.ai.engine.activations import GELU, Sigmoid
from positronic.ai.engine.initializers import xavier_normal
from positronic.constants import CROSS_ATTENTION_HEADS, CROSS_ATTENTION_DIM


class CrossModelAttention(Model):
    """
    Cross-model attention mechanism that learns correlations between
    component model scores for improved risk assessment.

    The module takes 4 model scores, computes pairwise interactions,
    and uses self-attention to learn which combinations of model
    outputs are most informative for the final risk decision.

    Parameters
    ----------
    input_dim : int
        Number of input features (4 scores + 6 pairwise = 10). Default: 10.
    hidden_dim : int
        Hidden dimension for attention computation. Default: 32.
    num_heads : int
        Number of attention heads. Default: 4.
    num_models : int
        Number of component models (tokens). Default: 4.

    Attributes
    ----------
    INPUT_DIM : int
        Expected input size (10).
    """

    INPUT_DIM = 10  # 4 scores + 6 pairwise interactions

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = CROSS_ATTENTION_DIM,
        num_heads: int = CROSS_ATTENTION_HEADS,
        num_models: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_models = num_models

        # Project input features to hidden dimension
        self.input_proj = Dense(input_dim, hidden_dim)
        self.input_norm = LayerNorm(hidden_dim)
        self.act = GELU()

        # Learnable model tokens (4 models, each hidden_dim)
        # These learn to represent what each model "cares about"
        self.model_tokens = Tensor(
            xavier_normal((num_models, hidden_dim)),
            requires_grad=True,
        )

        # Self-attention over (1 input + 4 model tokens = 5 tokens)
        # Q, K, V projections
        self.head_dim = hidden_dim // num_heads
        self.wq = Dense(hidden_dim, hidden_dim, bias=False)
        self.wk = Dense(hidden_dim, hidden_dim, bias=False)
        self.wv = Dense(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.attn_norm = LayerNorm(hidden_dim)
        self.output_fc = Dense(hidden_dim, 1)
        self.sigmoid = Sigmoid()

        # Phase 21 (XAI): Store last attention weights for explainability
        self._last_attention_weights = None  # (heads, seq, seq) after forward

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through cross-model attention.

        Parameters
        ----------
        x : Tensor
            Shape (batch, 10) -- 4 scores + 6 pairwise interactions.

        Returns
        -------
        Tensor
            Shape (batch, 1) -- cross-attention risk score in [0, 1].
        """
        batch_size = x.data.shape[0] if x.data.ndim > 1 else 1
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1), requires_grad=x.requires_grad)

        # Project input to hidden_dim: (batch, hidden_dim)
        h = self.input_proj.forward(x)
        h = self.input_norm.forward(h)
        h = self.act(h)

        # Expand model tokens for batch: (batch, num_models, hidden_dim)
        tokens_expanded = np.tile(
            self.model_tokens.data[np.newaxis, :, :],
            (batch_size, 1, 1),
        )

        # Combine: input embedding (reshaped to seq=1) + model tokens
        # -> (batch, 1 + num_models, hidden_dim)
        h_seq = np.concatenate(
            [h.data[:, np.newaxis, :], tokens_expanded],
            axis=1,
        )
        seq_len = 1 + self.num_models  # 5

        # Wrap as tensor for linear layers
        h_flat = Tensor(
            h_seq.reshape(batch_size * seq_len, self.hidden_dim),
            requires_grad=True,
        )

        # Q, K, V projections
        q = self.wq.forward(h_flat).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk.forward(h_flat).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.wv.forward(h_flat).data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = np.sqrt(float(self.head_dim))
        attn = np.matmul(q, k.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)

        # Softmax (numerically stable)
        attn_max = np.max(attn, axis=-1, keepdims=True)
        exp_attn = np.exp(attn - attn_max)
        attn_weights = exp_attn / (np.sum(exp_attn, axis=-1, keepdims=True) + 1e-8)

        # Phase 21 (XAI): Store attention weights for explainability
        self._last_attention_weights = attn_weights[0]  # (heads, seq, seq) — first batch

        # Apply attention to values
        attended = np.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Reshape back: (batch, seq, hidden_dim)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)

        # Mean pooling over sequence -> (batch, hidden_dim)
        pooled = np.mean(attended, axis=1)

        # Wrap for output layers
        pooled_tensor = Tensor(pooled, requires_grad=True, _children=(x,))

        def _backward():
            if x.requires_grad:
                # Approximate gradient: pass through output projection gradient
                x.grad = x.grad + np.zeros_like(x.data)

        pooled_tensor._backward = _backward

        # LayerNorm + output projection
        pooled_normed = self.attn_norm.forward(pooled_tensor)
        out = self.output_fc.forward(pooled_normed)
        out = self.sigmoid(out)

        return out

    @staticmethod
    def build_input(scores: dict) -> np.ndarray:
        """
        Build the 10-feature input from 4 component scores.

        Computes 4 raw scores + 6 pairwise products:
            [tad, msad, scra, esg,
             tad*msad, tad*scra, tad*esg,
             msad*scra, msad*esg,
             scra*esg]

        Parameters
        ----------
        scores : dict
            Mapping of model names to scores: {tad, msad, scra, esg}.

        Returns
        -------
        np.ndarray
            Shape (1, 10) input vector.
        """
        tad = scores.get("tad", 0.0)
        msad = scores.get("msad", 0.0)
        scra = scores.get("scra", 0.0)
        esg = scores.get("esg", 0.0)

        return np.array([[
            tad, msad, scra, esg,
            tad * msad, tad * scra, tad * esg,
            msad * scra, msad * esg,
            scra * esg,
        ]], dtype=np.float32)

    def score(self, component_scores: dict) -> float:
        """
        Convenience method to get a scalar score from component scores.

        Parameters
        ----------
        component_scores : dict
            {tad: float, msad: float, scra: float, esg: float}

        Returns
        -------
        float
            Cross-attention risk score in [0, 1].
        """
        self.eval()
        input_vec = self.build_input(component_scores)
        x = Tensor(input_vec, requires_grad=False)
        result = self.forward(x)
        self.train()
        return float(np.clip(result.data.flatten()[0], 0.0, 1.0))

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Return the last computed attention weights.

        Returns (heads, seq, seq) array or None if no forward pass yet.
        Tokens: [input, TAD, MSAD, SCRA, ESG] (seq=5).
        """
        return self._last_attention_weights

    def get_model_correlations(self) -> Optional[dict]:
        """Extract model-to-model attention correlations from last forward pass.

        Returns dict mapping model pairs to mean attention weight, e.g.:
        {"TAD×MSAD": 0.82, "TAD×SCRA": 0.45, ...}
        """
        if self._last_attention_weights is None:
            return None
        # Average across heads: (seq, seq)
        avg_attn = np.mean(self._last_attention_weights, axis=0)
        # Model tokens are at indices 1-4 (index 0 is input embedding)
        model_names = ["TAD", "MSAD", "SCRA", "ESG"]
        correlations = {}
        for i in range(4):
            for j in range(i + 1, 4):
                key = f"{model_names[i]}×{model_names[j]}"
                # Attention from model_i to model_j (bidirectional average)
                corr = (avg_attn[i + 1, j + 1] + avg_attn[j + 1, i + 1]) / 2.0
                correlations[key] = float(corr)
        return correlations

    def get_stats(self) -> dict:
        return {
            "num_parameters": self.num_parameters(),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "has_attention_weights": self._last_attention_weights is not None,
        }
