"""
Positronic - Learned Meta-Ensemble
===================================

Neural network that learns optimal, context-sensitive weighting of
component model scores.  Deployed inside the AI Validation Gate, it
replaces the static weighted average (0.30 / 0.25 / 0.25 / 0.20) once
sufficient labelled feedback has been collected.

Architecture (Base MLP Path)
----------------------------
    Dense(10, 16) + LayerNorm + GELU + Dropout(0.1)
    Dense(16,  8) + LayerNorm + GELU + Dropout(0.1)
    Dense( 8,  1) + Sigmoid

Phase 16: Cross-Model Attention Path
-------------------------------------
    CrossModelAttention: 4 scores + 6 pairwise -> self-attention -> score
    Gated combination: final = gate * attn_score + (1-gate) * base_score
    Gate starts at 0.0 (safe: defaults to old behavior)
    Cross-attention activates after 500+ training steps

Input Features (10)
-------------------
    Component scores (4):
        tad   - Transaction Anomaly Detector score
        msad  - Multi-Scale Anomaly Detector score
        scra  - Smart Contract Risk Analyser score
        esg   - Economic Stability Guardian score

    Transaction context (6):
        tx_type            - encoded transaction type (0 = transfer, etc.)
        value_log          - log-scaled transaction value
        sender_nonce       - sender's cumulative transaction count
        sender_reputation  - reputation score in [0, 1]
        is_contract        - 1 if recipient is a smart contract, else 0
        gas_price_ratio    - offered gas / median network gas

Output
------
    Single float in [0, 1] representing the final risk score.

Fallback Behaviour
------------------
    Before ``_min_steps`` (200) training iterations have been performed
    the model returns a deterministic static weighted average of the
    four component scores, ensuring safe operation from first boot.
"""

import logging
import numpy as np
from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import Dense, LayerNorm, Dropout
from positronic.ai.engine.activations import GELU, Sigmoid
from positronic.constants import (
    CROSS_ATTENTION_MIN_STEPS,
    META_ENSEMBLE_GATE_INIT,
)

logger = logging.getLogger("positronic.ai.meta_ensemble")


class MetaEnsemble(Model):
    """
    Learned meta-ensemble for combining component model scores.

    The model accepts a 10-dimensional input vector (four component
    scores concatenated with six transaction context features) and
    produces a single risk score via a small three-layer network.

    Phase 16 adds a cross-model attention path that learns correlations
    between model scores. A learned gate blends the attention path with
    the base MLP path, starting at 0.0 (pure MLP) and gradually
    shifting toward cross-attention as training progresses.

    Until enough supervised labels have been collected and applied
    through ``train_step``, ``score()`` transparently falls back to a
    static weighted average identical to the one hard-coded in earlier
    versions of the AI Validation Gate.

    Parameters
    ----------
    (none -- all hyper-parameters are fixed for reproducibility)

    Attributes
    ----------
    INPUT_DIM : int
        Expected input dimensionality (10).
    """

    INPUT_DIM = 10  # 4 component scores + 6 context features

    def __init__(self):
        super().__init__()

        # -- Hidden layers (base MLP path) ---------------------------------
        self.fc1 = Dense(self.INPUT_DIM, 16)
        self.norm1 = LayerNorm(16)
        self.fc2 = Dense(16, 8)
        self.norm2 = LayerNorm(8)
        self.fc3 = Dense(8, 1)

        # -- Shared activations / regularisation ---------------------------
        self.act = GELU()
        self.sigmoid = Sigmoid()
        self.dropout = Dropout(0.1)

        # -- Training bookkeeping -----------------------------------------
        self._trained: bool = False
        self._training_steps: int = 0
        self._min_steps: int = 200  # minimum steps before neural path

        # -- Static fallback weights (legacy gate behaviour) ---------------
        self._static_weights: dict = {
            "tad": 0.30,
            "msad": 0.25,
            "scra": 0.25,
            "esg": 0.20,
        }

        # -- Phase 16: Cross-model attention path --------------------------
        self._cross_attention = None
        self._cross_attention_ready: bool = False
        self._cross_attention_min_steps: int = CROSS_ATTENTION_MIN_STEPS

        # Gated combination: final = gate * attn + (1-gate) * base
        # gate starts at 0.0 (safe: purely old behavior at first)
        self._gate_value: float = META_ENSEMBLE_GATE_INIT

        # Lazy-initialize cross-attention to avoid circular imports
        self._init_cross_attention()

    def _init_cross_attention(self):
        """Initialize cross-model attention (lazy, safe)."""
        try:
            from positronic.ai.models.cross_attention import CrossModelAttention
            self._cross_attention = CrossModelAttention()
        except (ImportError, Exception):
            self._cross_attention = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return ``True`` when the ensemble has been trained enough to
        replace the static fallback."""
        return self._trained and self._training_steps >= self._min_steps

    @property
    def cross_attention_active(self) -> bool:
        """Return ``True`` when cross-attention path is active."""
        return (
            self._cross_attention is not None
            and self._cross_attention_ready
            and self._training_steps >= self._cross_attention_min_steps
        )

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the neural forward pass (base MLP path).

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, 10)`` or ``(1, 10)``.

        Returns
        -------
        Tensor
            Shape ``(batch, 1)`` -- predicted risk scores in [0, 1].
        """
        h = self.fc1.forward(x)
        h = self.norm1.forward(h)
        h = self.act(h)
        h = self.dropout.forward(h)

        h = self.fc2.forward(h)
        h = self.norm2.forward(h)
        h = self.act(h)
        h = self.dropout.forward(h)

        h = self.fc3.forward(h)
        h = self.sigmoid(h)

        return h

    # -----------------------------------------------------------------
    # Inference API
    # -----------------------------------------------------------------

    def score(self, component_scores: dict, tx_context: dict) -> float:
        """
        Produce the final risk score from component outputs and context.

        If the model has not yet been trained for at least
        ``_min_steps`` iterations, it falls back to the deterministic
        static weighted average used by legacy versions of the AI
        Validation Gate.

        Phase 16: When cross-attention is active, blends MLP and
        cross-attention scores via a learned gate.

        Parameters
        ----------
        component_scores : dict
            Mapping of component name to scalar score::

                {'tad': float, 'msad': float, 'scra': float, 'esg': float}

        tx_context : dict
            Transaction context features::

                {
                    'tx_type': int,
                    'value_log': float,
                    'sender_nonce': int,
                    'sender_reputation': float,
                    'is_contract': int,
                    'gas_price_ratio': float,
                }

        Returns
        -------
        float
            Final risk score in [0, 1].
        """
        if not self.is_ready:
            # Deterministic static weighted average (legacy behaviour).
            return sum(
                component_scores.get(k, 0) * w
                for k, w in self._static_weights.items()
            )

        # -- Build the 10-feature input vector -------------------------
        input_vec = np.array(
            [[
                component_scores.get("tad", 0.0),
                component_scores.get("msad", 0.0),
                component_scores.get("scra", 0.0),
                component_scores.get("esg", 0.0),
                float(tx_context.get("tx_type", 0)),
                tx_context.get("value_log", 0.0),
                float(tx_context.get("sender_nonce", 0)),
                tx_context.get("sender_reputation", 1.0),
                float(tx_context.get("is_contract", 0)),
                tx_context.get("gas_price_ratio", 1.0),
            ]],
            dtype=np.float32,
        )

        self.eval()
        x = Tensor(input_vec, requires_grad=False)
        base_result = self.forward(x)
        base_score = float(np.clip(base_result.data.flatten()[0], 0.0, 1.0))

        # Phase 16: Cross-model attention path
        if self.cross_attention_active:
            try:
                attn_score = self._cross_attention.score(component_scores)
                gate = self._gate_value
                final_score = gate * attn_score + (1.0 - gate) * base_score
            except Exception as e:
                logger.debug("cross_attention_fallback: %s", e)
                final_score = base_score
        else:
            final_score = base_score

        self.train()

        return float(np.clip(final_score, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Training API
    # -----------------------------------------------------------------

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        optimizer,
    ) -> float:
        """
        Execute one supervised training step.

        Parameters
        ----------
        inputs : np.ndarray
            Shape ``(batch, 10)`` -- component scores concatenated with
            transaction context features.
        targets : np.ndarray
            Shape ``(batch, 1)`` -- ground-truth risk scores obtained
            from human review or post-hoc outcome analysis.
        optimizer
            Any optimizer exposing ``zero_grad()`` and ``step()``.

        Returns
        -------
        float
            Scalar mean-squared-error loss for this step.
        """
        optimizer.zero_grad()

        x = Tensor(inputs.astype(np.float32), requires_grad=False)
        y = Tensor(targets.astype(np.float32), requires_grad=False)

        pred = self.forward(x)

        # -- MSE loss --------------------------------------------------
        diff = pred - y
        loss = (diff * diff).mean()

        loss.backward()
        optimizer.step()

        # -- Bookkeeping -----------------------------------------------
        self._training_steps += 1
        if self._training_steps >= self._min_steps:
            self._trained = True

        # Phase 16: Activate cross-attention after sufficient training
        if (self._training_steps >= self._cross_attention_min_steps
                and self._cross_attention is not None):
            self._cross_attention_ready = True
            # Gradually increase gate: from 0 at 500 steps to 0.3 at 2000 steps
            progress = min(
                (self._training_steps - self._cross_attention_min_steps) / 1500.0,
                1.0,
            )
            self._gate_value = 0.3 * progress  # Max gate = 0.3 (base MLP stays dominant)

        return float(loss.data)

    # -----------------------------------------------------------------
    # Phase 21: Explainability API
    # -----------------------------------------------------------------

    def get_contribution_breakdown(self, component_scores: dict) -> dict:
        """Per-model contribution breakdown for XAI.

        Returns dict: model_name → {raw_score, weight, weighted_score, contribution_pct}.
        """
        model_names = {"tad": "TAD", "msad": "MSAD", "scra": "SCRA", "esg": "ESG"}
        total_weighted = sum(
            component_scores.get(k, 0) * w for k, w in self._static_weights.items()
        )
        breakdown = {}
        for key, weight in self._static_weights.items():
            raw = component_scores.get(key, 0.0)
            weighted = raw * weight
            pct = (weighted / total_weighted * 100) if total_weighted > 0 else 0.0
            breakdown[model_names[key]] = {
                "raw_score": round(raw, 4),
                "weight": weight,
                "weighted_score": round(weighted, 4),
                "contribution_pct": round(pct, 1),
            }
        return breakdown

    def get_attention_correlations(self) -> dict:
        """Return cross-model attention correlations if available."""
        if self._cross_attention is None:
            return {}
        try:
            return self._cross_attention.get_model_correlations() or {}
        except Exception as e:
            logger.debug("get_model_correlations_failed: %s", e)
            return {}

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return a snapshot of the ensemble's internal state for
        monitoring dashboards and health checks.

        Returns
        -------
        dict
            Keys: ``trained``, ``training_steps``, ``is_ready``,
            ``num_parameters``, ``cross_attention_active``, ``gate_value``.
        """
        return {
            "trained": self._trained,
            "training_steps": self._training_steps,
            "is_ready": self.is_ready,
            "num_parameters": self.num_parameters(),
            "cross_attention_active": self.cross_attention_active,
            "gate_value": self._gate_value,
        }
