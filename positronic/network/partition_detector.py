"""
Positronic - Network Partition Detector

Detects network partitions and triggers graceful recovery.
Monitors block arrival timing, peer count, and chain tip divergence
to identify when the node may be isolated from the main network.

Phase 17 GOD CHAIN addition.

**Fail-open**: If detection logic fails, the node assumes HEALTHY.
Recovery actions are non-destructive (only triggers peer discovery
and status requests — never drops data or modifies chain state).
"""

import time
from enum import IntEnum
from typing import List, Callable, Dict, Optional
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger
from positronic.constants import (
    BLOCK_TIME,
    PARTITION_BLOCK_TIMEOUT_MULTIPLIER,
    PARTITION_MIN_PEERS_THRESHOLD,
)


logger = get_logger(__name__)


class PartitionState(IntEnum):
    """Network health states."""
    HEALTHY = 0      # Normal operation
    DEGRADED = 1     # Some warning signals, not yet confirmed
    PARTITIONED = 2  # Confirmed partition detected
    RECOVERING = 3   # Active recovery in progress


@dataclass
class PartitionEvent:
    """Record of a partition event for diagnostics."""
    timestamp: float
    old_state: PartitionState
    new_state: PartitionState
    reason: str
    peer_count: int = 0
    blocks_behind: int = 0


class PartitionDetector:
    """Detects network partitions and triggers recovery.

    Detection signals:
        1. No new blocks for > ``BLOCK_TIME * PARTITION_BLOCK_TIMEOUT_MULTIPLIER``
        2. Peer count drops below ``PARTITION_MIN_PEERS_THRESHOLD``
        3. Multiple peers report significantly higher chain heights

    Recovery actions (all non-destructive):
        1. Trigger aggressive peer discovery
        2. Request chain status from all remaining peers
        3. Log partition event for forensic analysis

    The detector never:
        - Drops blocks or transactions
        - Modifies the chain state
        - Disconnects existing peers
        - Rejects incoming connections

    Example::

        detector = PartitionDetector()
        detector.on_block_received(block_height)
        state = detector.check_health()
    """

    def __init__(self):
        self._state: PartitionState = PartitionState.HEALTHY
        self._last_block_time: float = time.time()
        self._last_block_height: int = 0
        self._peer_count: int = 0
        self._peer_heights: Dict[str, int] = {}
        self._recovery_callbacks: List[Callable] = []
        self._event_history: List[PartitionEvent] = []
        self._max_events: int = 100
        self._block_timeout: float = BLOCK_TIME * PARTITION_BLOCK_TIMEOUT_MULTIPLIER

    @property
    def state(self) -> PartitionState:
        """Current partition state."""
        return self._state

    @property
    def is_healthy(self) -> bool:
        """True if network appears healthy."""
        return self._state == PartitionState.HEALTHY

    def on_block_received(self, height: int):
        """Notify the detector that a new block was received.

        This resets timeout counters and may transition from
        DEGRADED/PARTITIONED back to HEALTHY.

        Args:
            height: The height of the received block.
        """
        try:
            self._last_block_time = time.time()
            self._last_block_height = max(self._last_block_height, height)

            if self._state in (PartitionState.DEGRADED, PartitionState.RECOVERING):
                self._transition(PartitionState.HEALTHY, "Block received")
            elif self._state == PartitionState.PARTITIONED:
                self._transition(PartitionState.RECOVERING, "Block received during partition")
        except Exception as e:
            logger.debug("Error in on_block_received: %s", e)

    def on_peer_count_changed(self, count: int):
        """Update the current peer count.

        Args:
            count: Number of connected peers.
        """
        self._peer_count = count

    def on_peer_height_reported(self, peer_id: str, height: int):
        """Record a peer's reported chain height.

        Args:
            peer_id: Identifier of the reporting peer.
            height: The peer's chain height.
        """
        self._peer_heights[peer_id] = height
        # LRU: keep only last 50 peers
        if len(self._peer_heights) > 50:
            oldest = list(self._peer_heights.keys())[0]
            del self._peer_heights[oldest]

    def register_recovery_callback(self, callback: Callable):
        """Register a function to be called when partition is detected.

        Callbacks receive no arguments and should trigger non-destructive
        recovery actions (e.g., aggressive peer discovery).

        Args:
            callback: Callable to invoke on partition detection.
        """
        self._recovery_callbacks.append(callback)

    def check_health(self) -> PartitionState:
        """Evaluate current network health and update state.

        Checks multiple signals and transitions state accordingly.
        Triggers recovery callbacks if a new partition is detected.

        Returns:
            Current PartitionState after evaluation.
        """
        try:
            signals = self._evaluate_signals()

            if signals["block_timeout"] and signals["low_peers"]:
                # Strong partition signal
                if self._state != PartitionState.PARTITIONED:
                    self._transition(PartitionState.PARTITIONED,
                                   "Block timeout + low peer count")
                    self._trigger_recovery()

            elif signals["block_timeout"] or signals["low_peers"]:
                # Weak signal — degraded
                if self._state == PartitionState.HEALTHY:
                    self._transition(PartitionState.DEGRADED,
                                   "Block timeout" if signals["block_timeout"]
                                   else "Low peer count")

            elif signals["height_divergence"]:
                # We're behind — might be partitioned
                if self._state == PartitionState.HEALTHY:
                    self._transition(PartitionState.DEGRADED,
                                   "Chain height divergence")

            else:
                # All clear
                if self._state in (PartitionState.DEGRADED, PartitionState.RECOVERING):
                    self._transition(PartitionState.HEALTHY, "All signals clear")

            return self._state

        except Exception as e:
            # Fail-open: assume healthy
            logger.debug("Partition health check error (fail-open): %s", e)
            return PartitionState.HEALTHY

    def _evaluate_signals(self) -> Dict[str, bool]:
        """Evaluate individual partition signals.

        Returns:
            Dictionary of signal name → active boolean.
        """
        now = time.time()

        # Signal 1: No blocks received for too long
        block_timeout = (now - self._last_block_time) > self._block_timeout

        # Signal 2: Too few peers
        low_peers = self._peer_count < PARTITION_MIN_PEERS_THRESHOLD

        # Signal 3: Peers report much higher heights
        height_divergence = False
        if self._peer_heights:
            max_peer_height = max(self._peer_heights.values())
            if max_peer_height > self._last_block_height + 10:
                height_divergence = True

        return {
            "block_timeout": block_timeout,
            "low_peers": low_peers,
            "height_divergence": height_divergence,
        }

    def _transition(self, new_state: PartitionState, reason: str):
        """Transition to a new state and record the event."""
        if new_state == self._state:
            return

        event = PartitionEvent(
            timestamp=time.time(),
            old_state=self._state,
            new_state=new_state,
            reason=reason,
            peer_count=self._peer_count,
            blocks_behind=max(
                0,
                max(self._peer_heights.values(), default=0) - self._last_block_height,
            ),
        )
        self._event_history.append(event)
        if len(self._event_history) > self._max_events:
            self._event_history = self._event_history[-self._max_events:]

        self._state = new_state

    def _trigger_recovery(self):
        """Invoke all registered recovery callbacks."""
        for cb in self._recovery_callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning("Recovery callback failed: %s", e)

    def get_stats(self) -> Dict:
        """Return partition detector state for monitoring.

        Returns:
            Dictionary with current state, signal values, and event history.
        """
        signals = {}
        try:
            signals = self._evaluate_signals()
        except Exception as e:
            logger.debug("Failed to evaluate signals for stats: %s", e)

        return {
            "state": self._state.name,
            "is_healthy": self.is_healthy,
            "last_block_age_seconds": round(time.time() - self._last_block_time, 1),
            "last_block_height": self._last_block_height,
            "peer_count": self._peer_count,
            "signals": signals,
            "recent_events": len(self._event_history),
            "block_timeout_seconds": self._block_timeout,
        }
