"""
Positronic - Online Learning Extension (Phase 32h)

Extends the online learning system with labeled transaction sources,
confidence-weighted training batches, quality gates, and Elastic Weight
Consolidation (EWC) placeholders.

Active ONLY in Phase C (block > 500,000) of the cold start lifecycle.
When not in Phase C, all methods silently return False / no-op.

This module does NOT modify the existing OnlineLearner. It is a
standalone extension that can be composed alongside it.

Dependencies:
    - ColdStartManager from positronic.ai.cold_start_manager (optional)
    - numpy for EWC computation
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from positronic.utils.logging import get_logger
from positronic.constants import (
    OLE_MIN_BATCH_SIZE,
    OLE_MAX_BUFFER_SIZE,
    OLE_TRAIN_INTERVAL_BLOCKS,
    OLE_RATE_LIMIT_PER_HOUR,
    OLE_QUALITY_GATE_DROP,
    OLE_EWC_LAMBDA,
)

logger = get_logger(__name__)


class OnlineLearningExtension:
    """Phase 32h: Online Learning Extension with labeled TX sources.

    Provides confidence-weighted labeled transaction ingestion, quality-gated
    training, and EWC regularisation placeholders. The extension is only
    active during Phase C of the cold start lifecycle (block height > 500K).
    If no ``ColdStartManager`` is provided, the extension is always active.

    Args:
        cold_start_manager: Optional ColdStartManager instance. When given,
            the extension checks ``get_current_phase(block_height)`` and
            only activates in Phase C. When *None*, the extension is
            unconditionally active (useful for testing or post-cold-start
            deployments).
    """

    # Confidence weights for each label source.
    LABEL_SOURCES: Dict[str, float] = {
        "validator_consensus": 0.67,
        "appeal_result": 1.0,
        "confirmed_attack": 0.90,
    }

    def __init__(self, cold_start_manager=None):
        self._csm = cold_start_manager

        # Labeled TX buffer: list of dicts with tx_hash, label, source,
        # reporter, block_height, confidence, timestamp.
        self._buffer: List[dict] = []

        # Rate limiting: {reporter_hex: [timestamp, ...]}
        self._rate_limits: Dict[str, List[float]] = {}

        # Stats tracking
        self._total_submitted: int = 0
        self._last_training_block: int = 0

    # ─── Phase Gate ──────────────────────────────────────────────

    def is_active(self, block_height: int) -> bool:
        """Check if the extension is active at this block height.

        Returns True only during Phase C (block > 500K), or always True
        if no ColdStartManager was provided.

        Args:
            block_height: Current block height.

        Returns:
            True if the extension should operate.
        """
        if self._csm is None:
            return True
        return self._csm.get_current_phase(block_height) == "C"

    # ─── Labeled TX Submission ───────────────────────────────────

    def submit_labeled_tx(
        self,
        tx_hash: bytes,
        label: str,
        source: str,
        reporter: bytes,
        block_height: int,
    ) -> bool:
        """Submit a labeled transaction for future training.

        Args:
            tx_hash: Transaction hash (32 bytes).
            label: Human-readable label (e.g. ``"fraud"``, ``"legit"``).
            source: Label source — must be a key in ``LABEL_SOURCES``.
            reporter: Reporter address (20 bytes).
            block_height: Block height at submission time.

        Returns:
            True if accepted, False if not in Phase C, source is invalid,
            or the reporter is rate-limited.
        """
        if not self.is_active(block_height):
            return False

        if source not in self.LABEL_SOURCES:
            logger.debug(
                "Rejected labeled TX: unknown source %s", source,
            )
            return False

        # Rate limit check
        now = time.time()
        reporter_hex = reporter.hex()

        if reporter_hex in self._rate_limits:
            cutoff = now - 3600
            self._rate_limits[reporter_hex] = [
                t for t in self._rate_limits[reporter_hex] if t > cutoff
            ]
            if len(self._rate_limits[reporter_hex]) >= OLE_RATE_LIMIT_PER_HOUR:
                logger.debug(
                    "Rate-limited labeled TX from reporter %s (%d/hour)",
                    reporter_hex[:16],
                    OLE_RATE_LIMIT_PER_HOUR,
                )
                return False

        # Accept the labeled TX
        confidence = self.LABEL_SOURCES[source]
        entry = {
            "tx_hash": tx_hash,
            "label": label,
            "source": source,
            "reporter": reporter_hex,
            "block_height": block_height,
            "confidence": confidence,
            "timestamp": now,
        }
        self._buffer.append(entry)

        # Cap buffer size
        if len(self._buffer) > OLE_MAX_BUFFER_SIZE:
            self._buffer = self._buffer[-OLE_MAX_BUFFER_SIZE:]

        # Update rate limit tracker
        if reporter_hex not in self._rate_limits:
            self._rate_limits[reporter_hex] = []
        self._rate_limits[reporter_hex].append(now)

        self._total_submitted += 1

        logger.debug(
            "Labeled TX accepted: tx=%s source=%s confidence=%.2f",
            tx_hash.hex()[:16],
            source,
            confidence,
        )
        return True

    # ─── Training Trigger ────────────────────────────────────────

    def should_train(self, block_height: int) -> bool:
        """Determine whether training should occur at this block.

        Training triggers every ``OLE_TRAIN_INTERVAL_BLOCKS`` blocks when
        the buffer has at least ``OLE_MIN_BATCH_SIZE`` samples.

        Args:
            block_height: Current block height.

        Returns:
            True if training should run now.
        """
        if len(self._buffer) < OLE_MIN_BATCH_SIZE:
            return False
        return block_height % OLE_TRAIN_INTERVAL_BLOCKS == 0

    # ─── Training Batch ──────────────────────────────────────────

    def get_training_batch(self) -> List[dict]:
        """Return the current buffer contents as a training batch.

        Each entry includes the ``confidence`` key derived from the label
        source weight.

        Returns:
            List of dicts with keys: tx_hash, label, source, reporter,
            block_height, confidence, timestamp.
        """
        return list(self._buffer)

    # ─── Quality Gate ────────────────────────────────────────────

    def record_training_result(
        self, accuracy: float, previous_accuracy: float,
    ) -> Tuple[bool, str]:
        """Apply the quality gate after a training run.

        If accuracy drops by more than ``OLE_QUALITY_GATE_DROP`` (5%) from
        the previous accuracy, the update is reverted.

        Args:
            accuracy: Accuracy of the newly trained model.
            previous_accuracy: Accuracy before training.

        Returns:
            Tuple of (accepted: bool, message: str). ``("reverted")`` when
            the quality gate trips, ``("accepted")`` otherwise.
        """
        drop = previous_accuracy - accuracy
        if drop > OLE_QUALITY_GATE_DROP:
            logger.debug(
                "Quality gate tripped: accuracy dropped %.4f (threshold %.4f)",
                drop,
                OLE_QUALITY_GATE_DROP,
            )
            return (False, "reverted")

        return (True, "accepted")

    # ─── EWC (Elastic Weight Consolidation) ──────────────────────

    @staticmethod
    def compute_fisher_diagonal(model_weights: dict) -> dict:
        """Compute the Fisher Information diagonal (placeholder).

        In a production system, this would compute the diagonal of the
        Fisher Information Matrix by accumulating squared gradients over
        a representative dataset. This placeholder returns ones.

        Args:
            model_weights: Dict mapping parameter names to numpy arrays.

        Returns:
            Dict with the same keys and shapes, filled with ones.
        """
        return {
            key: np.ones_like(val) for key, val in model_weights.items()
        }

    @staticmethod
    def apply_ewc_penalty(
        new_weights: dict,
        old_weights: dict,
        fisher: dict,
        lambda_ewc: float = OLE_EWC_LAMBDA,
    ) -> float:
        """Compute the EWC regularisation penalty.

        penalty = lambda * sum_i( F_i * (theta_new_i - theta_old_i)^2 )

        Args:
            new_weights: New model parameters (dict of numpy arrays).
            old_weights: Old model parameters (dict of numpy arrays).
            fisher: Fisher diagonal (dict of numpy arrays, same structure).
            lambda_ewc: Regularisation strength (default ``OLE_EWC_LAMBDA``).

        Returns:
            Scalar penalty value (float).
        """
        penalty = 0.0
        for key in new_weights:
            diff = new_weights[key] - old_weights[key]
            penalty += float(np.sum(fisher[key] * diff ** 2))
        return lambda_ewc * penalty

    # ─── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a summary dictionary of extension state.

        Returns:
            Dictionary with buffer_size, total_submitted, rate_limits,
            and last_training_block.
        """
        return {
            "buffer_size": len(self._buffer),
            "total_submitted": self._total_submitted,
            "rate_limits": {
                k: len(v) for k, v in self._rate_limits.items()
            },
            "last_training_block": self._last_training_block,
        }
