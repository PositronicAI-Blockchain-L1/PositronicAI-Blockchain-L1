"""
Positronic - Concept Drift Detection (Phase 32j)

Monitors AI model scoring behavior for statistical drift from established
baselines.  When a model's rolling mean deviates significantly from its
frozen baseline, the detector classifies the drift severity and recommends
corrective action (e.g. reducing model weight in the meta-ensemble).

Severity levels:
    NONE   — drift < LOW threshold (currently 10%)
    LOW    — drift [0, 10%), log only
    MEDIUM — drift [10%, 30%), alert operators
    HIGH   — drift >= 30%, recommend 20% weight reduction

Usage::

    from positronic.ai.concept_drift import ConceptDriftDetector

    detector = ConceptDriftDetector()
    for score in training_scores:
        detector.record_score("tad", score, block_height)
    detector.set_baseline("tad")
    # ... later ...
    alert = detector.check_drift("tad", current_block)
    if alert and alert.severity == DriftSeverity.HIGH:
        # reduce model weight externally
        ...
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Deque, Dict, List, Optional

from positronic.constants import (
    DRIFT_LOW_THRESHOLD,
    DRIFT_MAX_ALERTS,
    DRIFT_MEDIUM_THRESHOLD,
    DRIFT_WEIGHT_REDUCTION,
    DRIFT_WINDOW_SIZE,
)
from positronic.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class DriftSeverity(IntEnum):
    """Classification of how far a model has drifted from baseline."""
    NONE = 0
    LOW = 1       # < 10% drift, log only
    MEDIUM = 2    # 10-30% drift, alert
    HIGH = 3      # > 30% drift, reduce model weight 20%


@dataclass
class DriftAlert:
    """Record of a detected drift event."""
    model_name: str
    severity: DriftSeverity
    drift_percentage: float
    baseline_mean: float
    current_mean: float
    block_height: int
    timestamp: float
    action_taken: str   # "none", "alert", "weight_reduced"


# ---------------------------------------------------------------------------
# Internal per-model state
# ---------------------------------------------------------------------------

@dataclass
class _ModelState:
    """Internal bookkeeping for a single model's score window."""
    scores: Deque[float]
    baseline_mean: Optional[float]

    @property
    def current_mean(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ConceptDriftDetector:
    """Detects concept drift in AI model scores against a frozen baseline.

    Parameters
    ----------
    window_size : int
        Number of recent scores to keep in the rolling window (default from
        ``DRIFT_WINDOW_SIZE``).
    max_alerts : int
        Maximum stored alerts; oldest are pruned when exceeded (default from
        ``DRIFT_MAX_ALERTS``).
    """

    def __init__(
        self,
        window_size: int = DRIFT_WINDOW_SIZE,
        max_alerts: int = DRIFT_MAX_ALERTS,
    ) -> None:
        self._window_size: int = window_size
        self._max_alerts: int = max_alerts
        self._models: Dict[str, _ModelState] = {}
        self._alerts: List[DriftAlert] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self, model_name: str) -> _ModelState:
        """Lazily create tracking state for *model_name*."""
        if model_name not in self._models:
            self._models[model_name] = _ModelState(
                scores=deque(maxlen=self._window_size),
                baseline_mean=None,
            )
        return self._models[model_name]

    @staticmethod
    def _calc_drift_pct(baseline: float, current: float) -> float:
        """Return absolute drift percentage between *baseline* and *current*."""
        return abs(current - baseline) / max(baseline, 1e-6) * 100.0

    @staticmethod
    def _classify(drift_pct: float) -> DriftSeverity:
        if drift_pct >= DRIFT_MEDIUM_THRESHOLD:
            return DriftSeverity.HIGH
        if drift_pct >= DRIFT_LOW_THRESHOLD:
            return DriftSeverity.MEDIUM
        if drift_pct > 0.0:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    @staticmethod
    def _action_for(severity: DriftSeverity) -> str:
        if severity == DriftSeverity.HIGH:
            return "weight_reduced"
        if severity == DriftSeverity.MEDIUM:
            return "alert"
        return "none"

    def _prune_alerts(self) -> None:
        """Keep only the newest *max_alerts* entries."""
        while len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_score(self, model_name: str, score: float, block_height: int) -> None:
        """Append a score to the rolling window for *model_name*."""
        state = self._ensure_model(model_name)
        state.scores.append(score)
        logger.debug(
            "drift score recorded model=%s score=%.4f block=%d window=%d",
            model_name, score, block_height, len(state.scores),
        )

    def set_baseline(self, model_name: str) -> None:
        """Freeze the current rolling mean as the baseline for *model_name*."""
        state = self._ensure_model(model_name)
        state.baseline_mean = state.current_mean
        logger.info(
            "drift baseline set model=%s baseline=%.6f samples=%d",
            model_name, state.baseline_mean, len(state.scores),
        )

    def check_drift(
        self,
        model_name: str,
        block_height: int,
    ) -> Optional[DriftAlert]:
        """Compare rolling mean against baseline and return an alert if drifted.

        Returns ``None`` when:
        - no baseline has been set, or
        - drift severity is NONE (i.e. < ``DRIFT_LOW_THRESHOLD`` and exactly 0).
        """
        state = self._ensure_model(model_name)

        if state.baseline_mean is None:
            return None

        current = state.current_mean
        drift_pct = self._calc_drift_pct(state.baseline_mean, current)
        severity = self._classify(drift_pct)

        if severity == DriftSeverity.NONE:
            return None

        alert = DriftAlert(
            model_name=model_name,
            severity=severity,
            drift_percentage=drift_pct,
            baseline_mean=state.baseline_mean,
            current_mean=current,
            block_height=block_height,
            timestamp=time.time(),
            action_taken=self._action_for(severity),
        )
        self._alerts.append(alert)
        self._prune_alerts()

        logger.warning(
            "drift detected model=%s severity=%s drift=%.2f%% action=%s block=%d",
            model_name, severity.name, drift_pct, alert.action_taken, block_height,
        )
        return alert

    def get_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[DriftSeverity] = None,
    ) -> List[DriftAlert]:
        """Return stored alerts, optionally filtered by model and/or severity."""
        result = self._alerts
        if model_name is not None:
            result = [a for a in result if a.model_name == model_name]
        if severity is not None:
            result = [a for a in result if a.severity == severity]
        return list(result)

    def get_model_stats(self, model_name: str) -> dict:
        """Return current statistics for *model_name*.

        Keys: ``baseline_mean``, ``current_mean``, ``drift_percentage``,
        ``window_size``, ``scores_recorded``.
        """
        state = self._ensure_model(model_name)
        current = state.current_mean
        baseline = state.baseline_mean
        if baseline is not None:
            drift_pct = self._calc_drift_pct(baseline, current)
        else:
            drift_pct = 0.0
        return {
            "baseline_mean": baseline,
            "current_mean": current,
            "drift_percentage": drift_pct,
            "window_size": self._window_size,
            "scores_recorded": len(state.scores),
        }

    def restore_weight(self, model_name: str) -> bool:
        """Check whether drift has recovered enough to restore model weight.

        Returns ``True`` if the current drift is below ``DRIFT_LOW_THRESHOLD``
        (i.e. < 10%), meaning the model is safe to use at full weight again.
        """
        state = self._ensure_model(model_name)
        if state.baseline_mean is None:
            return False
        drift_pct = self._calc_drift_pct(state.baseline_mean, state.current_mean)
        can_restore = drift_pct < DRIFT_LOW_THRESHOLD
        if can_restore:
            logger.info(
                "drift recovered model=%s drift=%.2f%% — weight restore OK",
                model_name, drift_pct,
            )
        else:
            logger.info(
                "drift still elevated model=%s drift=%.2f%% — restore denied",
                model_name, drift_pct,
            )
        return can_restore

    def get_status(self) -> dict:
        """Return drift status for every tracked model."""
        return {
            name: self.get_model_stats(name)
            for name in self._models
        }
