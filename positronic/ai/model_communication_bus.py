"""
Positronic - Phase 32i: Model Communication Bus

TX-scoped signal sharing system between PoNC models (TAD, MSAD, SCRA, ESG).
Allows models to share observations within a single transaction's validation
context without cross-TX contamination or self-subscription.

Usage:
    bus = ModelCommunicationBus()
    bus.begin_tx(tx_hash)
    bus.publish("tad", "anomaly_high", 0.9)        # TAD publishes
    signals = bus.read_signals("msad")               # MSAD reads (excludes own)
    boost = bus.compute_context_boost("msad")        # compute bp adjustment
    bus.end_tx()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from positronic.utils.logging import get_logger
from positronic.constants import MODEL_BUS_BOOST_CAP

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Predefined signal types per model
# ---------------------------------------------------------------------------

SIGNAL_TYPES: Dict[str, List[str]] = {
    "tad": ["anomaly_high", "anomaly_low", "pattern_novel"],
    "msad": ["mev_detected", "sandwich_suspected", "frontrun_detected"],
    "scra": ["reentrancy_risk", "selfdestruct_risk", "proxy_upgrade"],
    "esg": ["volatility_high", "liquidity_low", "whale_activity"],
}

# Signal types that indicate *lower* risk (negative boost direction).
# All other signal types are treated as risk-increasing (positive boost).
_NEGATIVE_SIGNAL_TYPES = frozenset({
    "anomaly_low",
})

# All valid model sources
_VALID_SOURCES = frozenset(SIGNAL_TYPES.keys())

# Flat set of all valid signal type strings (for fast validation)
_ALL_SIGNAL_TYPES = frozenset(
    sig for sigs in SIGNAL_TYPES.values() for sig in sigs
)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSignal:
    """A single observation published by a PoNC model during TX validation.

    Attributes:
        source: Model name that published the signal (tad, msad, scra, esg).
        signal_type: Semantic label for the observation.
        value: Signal strength in [0.0, 1.0].
        tx_hash: The transaction this signal is scoped to.
        timestamp: Unix timestamp when the signal was created.
    """
    source: str
    signal_type: str
    value: float
    tx_hash: bytes
    timestamp: float


# ---------------------------------------------------------------------------
# Model Communication Bus
# ---------------------------------------------------------------------------

class ModelCommunicationBus:
    """TX-scoped signal sharing bus for PoNC models.

    Within a ``begin_tx`` / ``end_tx`` window, models can publish
    observations and read signals from *other* models (no self-subscription).
    Signals auto-clear on each ``begin_tx`` to prevent cross-TX contamination.

    Args:
        boost_cap: Maximum context boost magnitude in basis points.
                   Defaults to ``MODEL_BUS_BOOST_CAP`` (500).
    """

    def __init__(self, boost_cap: int = MODEL_BUS_BOOST_CAP) -> None:
        self._boost_cap: int = boost_cap

        # Current TX context
        self._active: bool = False
        self._current_tx_hash: Optional[bytes] = None
        self._signals: List[ModelSignal] = []

        # Lifetime statistics (persist across TXs)
        self._total_signals: int = 0
        self._by_model: Dict[str, int] = {}
        self._by_type: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # TX lifecycle
    # ------------------------------------------------------------------

    def begin_tx(self, tx_hash: bytes) -> None:
        """Start a new TX context, clearing all signals from any prior TX.

        Args:
            tx_hash: The hash of the transaction being validated.
        """
        self._signals.clear()
        self._current_tx_hash = tx_hash
        self._active = True
        logger.debug("Bus: begin_tx %s", tx_hash.hex()[:16])

    def end_tx(self) -> None:
        """Finalize the current TX context and clear signals."""
        logger.debug("Bus: end_tx %s",
                      self._current_tx_hash.hex()[:16] if self._current_tx_hash else "none")
        self._signals.clear()
        self._current_tx_hash = None
        self._active = False

    def is_active(self) -> bool:
        """Return True if the bus is within a TX context."""
        return self._active

    # ------------------------------------------------------------------
    # Publish / Read
    # ------------------------------------------------------------------

    def publish(self, source: str, signal_type: str, value: float) -> None:
        """Publish a signal from *source* into the current TX context.

        Args:
            source: Model name (must be one of tad, msad, scra, esg).
            signal_type: Must be a valid type for the given source.
            value: Signal strength; clamped to [0.0, 1.0].

        Raises:
            RuntimeError: If no TX context is active.
            ValueError: If source or signal_type is invalid.
        """
        if not self._active:
            raise RuntimeError("No active TX context — call begin_tx() first")

        # Validate source
        if source not in _VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{source}'; must be one of {sorted(_VALID_SOURCES)}"
            )

        # Validate signal_type for this source
        if signal_type not in SIGNAL_TYPES[source]:
            raise ValueError(
                f"Invalid signal_type '{signal_type}' for source '{source}'; "
                f"valid types: {SIGNAL_TYPES[source]}"
            )

        # Clamp value to [0.0, 1.0]
        value = max(0.0, min(1.0, value))

        signal = ModelSignal(
            source=source,
            signal_type=signal_type,
            value=value,
            tx_hash=self._current_tx_hash,
            timestamp=time.time(),
        )
        self._signals.append(signal)

        # Update lifetime stats
        self._total_signals += 1
        self._by_model[source] = self._by_model.get(source, 0) + 1
        self._by_type[signal_type] = self._by_type.get(signal_type, 0) + 1

        logger.debug("Bus: publish %s/%s=%.3f tx=%s",
                      source, signal_type, value,
                      self._current_tx_hash.hex()[:16] if self._current_tx_hash else "?")

    def read_signals(self, reader: str) -> List[ModelSignal]:
        """Read all signals NOT published by *reader* in the current TX.

        Args:
            reader: The model name requesting the signals.

        Returns:
            List of ``ModelSignal`` objects from other models.
        """
        if not self._active:
            return []

        return [s for s in self._signals if s.source != reader]

    # ------------------------------------------------------------------
    # Context boost
    # ------------------------------------------------------------------

    def compute_context_boost(self, reader: str) -> int:
        """Compute a basis-point adjustment from signals visible to *reader*.

        Positive signals (risk-increasing) contribute +bp.
        Negative signals (risk-decreasing) contribute -bp.
        The result is clamped to [-boost_cap, +boost_cap].

        Args:
            reader: The model requesting the boost.

        Returns:
            Integer basis-point adjustment, clamped to +-boost_cap.
        """
        if not self._active:
            return 0

        signals = self.read_signals(reader)
        if not signals:
            return 0

        # Sum signed contributions.
        # Each signal contributes value * 1000 bp (so 1.0 = 1000bp per signal).
        # Negative signal types subtract instead of add.
        raw_bp = 0.0
        for sig in signals:
            contribution = sig.value * 1000.0  # scale to bp
            if sig.signal_type in _NEGATIVE_SIGNAL_TYPES:
                raw_bp -= contribution
            else:
                raw_bp += contribution

        # Convert to int and clamp
        boost = int(raw_bp)
        boost = max(-self._boost_cap, min(self._boost_cap, boost))
        return boost

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return lifetime statistics about signal traffic.

        Returns:
            Dict with keys: total_signals, by_model, by_type.
        """
        return {
            "total_signals": self._total_signals,
            "by_model": dict(self._by_model),
            "by_type": dict(self._by_type),
        }
