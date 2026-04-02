"""
Positronic - Emergency Controller
State machine for network emergency states: NORMAL, DEGRADED, PAUSED, HALTED, UPGRADING.
"""

import time
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from positronic.types import EmergencyStateDict

logger = logging.getLogger("positronic.emergency.controller")


class NetworkState(IntEnum):
    """Network operational states."""
    NORMAL = 0       # Full operation
    DEGRADED = 1     # Block production yes, limited transactions
    PAUSED = 2       # No blocks, no transactions, RPC read-only
    HALTED = 3       # No blocks, no transactions, emergency-only RPC
    UPGRADING = 4    # No blocks, no transactions, RPC read-only


@dataclass
class EmergencyEvent:
    """Record of a state transition event."""
    timestamp: float
    old_state: NetworkState
    new_state: NetworkState
    reason: str
    triggered_by: str       # "founder", "multisig", "auto:health", "upgrade_manager"
    block_height: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmergencyController:
    """
    Network emergency state machine.

    Controls block production and transaction acceptance based on network state.
    Founder can pause/resume. Multi-sig required for halt/resume-from-halt.
    HealthMonitor can auto-degrade/pause.
    """

    def __init__(self, node=None):
        self._node = node
        self._state: NetworkState = NetworkState.NORMAL
        self._event_log: List[EmergencyEvent] = []
        self._last_state_change: float = time.time()
        self._state_reason: str = ""
        self._multisig = None  # Set later via set_multisig()
        self._upgrade_manager = None  # Set later via set_upgrade_manager()

    @property
    def state(self) -> NetworkState:
        return self._state

    def set_multisig(self, multisig):
        """Wire up the MultisigManager for multi-signature operations."""
        self._multisig = multisig

    def set_upgrade_manager(self, upgrade_manager):
        """Wire up the UpgradeManager."""
        self._upgrade_manager = upgrade_manager

    # === Query ===

    def get_state(self) -> EmergencyStateDict:
        """Return current state and metadata."""
        return {
            "state": int(self._state),
            "state_name": self._state.name,
            "since": self._last_state_change,
            "reason": self._state_reason,
            "block_height": self._get_block_height(),
            "event_count": len(self._event_log),
        }

    def get_event_log(self, limit: int = 50) -> List[EmergencyEvent]:
        """Return recent events, newest first."""
        return list(reversed(self._event_log[-limit:]))

    # === Block/TX gates (called by Node and Mempool) ===

    def should_produce_blocks(self) -> bool:
        """Called by Node._block_production_loop before creating a block."""
        return self._state in (NetworkState.NORMAL, NetworkState.DEGRADED)

    def should_accept_transactions(self) -> bool:
        """Called by Mempool / Node before accepting a transaction."""
        return self._state == NetworkState.NORMAL

    # === Multi-sig pause/resume (formerly founder-only, now requires 3-of-5) ===

    def pause_network(self, reason: str, action_id: str,
                      founder_pubkey: bytes = b"", signature: bytes = b"") -> bool:
        """
        Pause the network (requires 3-of-5 multi-sig).

        Previously founder-only (single key), now requires multisig to prevent
        a compromised founder key from unilaterally pausing the network.

        Args:
            reason: Human-readable reason for the pause.
            action_id: MultisigManager action ID with sufficient signatures.
            founder_pubkey: Deprecated, ignored. Kept for API compatibility.
            signature: Deprecated, ignored. Kept for API compatibility.
        """
        if self._state not in (NetworkState.NORMAL, NetworkState.DEGRADED):
            logger.warning(f"Cannot pause from state {self._state.name}")
            return False

        if self._multisig is None:
            logger.error("MultiSigManager not configured — cannot pause")
            return False

        if not self._multisig.is_executable(action_id):
            logger.warning(f"Pause action {action_id} not executable (insufficient sigs)")
            return False

        self._multisig.execute(action_id)
        self._transition(NetworkState.PAUSED, reason, "multisig")
        return True

    def resume_network(self, action_id: str,
                       founder_pubkey: bytes = b"", signature: bytes = b"") -> bool:
        """
        Resume from PAUSED state (requires 3-of-5 multi-sig).

        Previously founder-only, now requires multisig for consistency.

        Args:
            action_id: MultisigManager action ID with sufficient signatures.
            founder_pubkey: Deprecated, ignored. Kept for API compatibility.
            signature: Deprecated, ignored. Kept for API compatibility.
        """
        if self._state != NetworkState.PAUSED:
            logger.warning(f"Cannot resume from state {self._state.name}")
            return False

        if self._multisig is None:
            logger.error("MultiSigManager not configured — cannot resume")
            return False

        if not self._multisig.is_executable(action_id):
            logger.warning(f"Resume action {action_id} not executable (insufficient sigs)")
            return False

        self._multisig.execute(action_id)
        self._transition(NetworkState.NORMAL, "Resumed via multi-sig", "multisig")
        return True

    # === Multi-sig controls ===

    def emergency_halt(self, reason: str, action_id: str) -> bool:
        """
        Emergency halt (requires 3-of-5 multi-sig).
        Checks MultiSigManager for action_id readiness.
        """
        if self._multisig is None:
            logger.error("MultiSigManager not configured")
            return False

        if not self._multisig.is_executable(action_id):
            logger.warning(f"Action {action_id} not executable (insufficient sigs or timelock)")
            return False

        self._multisig.execute(action_id)
        self._transition(NetworkState.HALTED, reason, "multisig")
        return True

    def resume_from_halt(self, action_id: str) -> bool:
        """
        Resume from HALTED state (requires 3-of-5 multi-sig).
        """
        if self._state != NetworkState.HALTED:
            logger.warning(f"Cannot resume-from-halt in state {self._state.name}")
            return False

        if self._multisig is None:
            logger.error("MultiSigManager not configured")
            return False

        if not self._multisig.is_executable(action_id):
            logger.warning(f"Action {action_id} not executable")
            return False

        self._multisig.execute(action_id)
        self._transition(NetworkState.NORMAL, "Resumed from halt via multi-sig", "multisig")
        return True

    # === Automatic triggers (called by HealthMonitor) ===

    def on_health_degraded(self, metrics: dict) -> None:
        """Auto-transition to DEGRADED when health monitor reports issues."""
        if self._state in (NetworkState.HALTED, NetworkState.UPGRADING):
            return  # Don't override more severe states
        if self._state == NetworkState.NORMAL:
            self._transition(
                NetworkState.DEGRADED,
                f"Health degraded: {metrics}",
                "auto:health",
            )

    def on_health_critical(self, metrics: dict) -> None:
        """Auto-transition to PAUSED when health monitor reports critical failure."""
        if self._state in (NetworkState.HALTED, NetworkState.UPGRADING):
            return
        if self._state in (NetworkState.NORMAL, NetworkState.DEGRADED):
            self._transition(
                NetworkState.PAUSED,
                f"Health critical: {metrics}",
                "auto:health",
            )

    def on_health_recovered(self, metrics: dict) -> None:
        """Auto-recover from DEGRADED to NORMAL when health improves."""
        if self._state == NetworkState.DEGRADED:
            self._transition(
                NetworkState.NORMAL,
                f"Health recovered: {metrics}",
                "auto:health",
            )
        # NOTE: Does NOT recover from PAUSED/HALTED — those need manual action

    # === Upgrade Manager integration ===

    def enter_upgrade_mode(self, upgrade_id: str) -> bool:
        """Transition to UPGRADING state (called by UpgradeManager)."""
        if self._state not in (NetworkState.NORMAL, NetworkState.DEGRADED):
            return False
        self._transition(
            NetworkState.UPGRADING,
            f"Upgrade {upgrade_id} activating",
            "upgrade_manager",
        )
        return True

    def exit_upgrade_mode(self, upgrade_id: str, success: bool) -> None:
        """Exit UPGRADING state (called by UpgradeManager)."""
        if self._state != NetworkState.UPGRADING:
            return
        reason = f"Upgrade {upgrade_id} {'completed' if success else 'rolled back'}"
        self._transition(NetworkState.NORMAL, reason, "upgrade_manager")

    # === Internal helpers ===

    def _transition(self, new_state: NetworkState, reason: str, triggered_by: str) -> None:
        """Record state transition."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        self._state_reason = reason

        event = EmergencyEvent(
            timestamp=self._last_state_change,
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            triggered_by=triggered_by,
            block_height=self._get_block_height(),
            metadata={},
        )
        self._event_log.append(event)

        logger.warning(
            f"EMERGENCY STATE: {old_state.name} -> {new_state.name} "
            f"(by {triggered_by}): {reason}"
        )

    def _verify_founder(self, pubkey: bytes, signature: bytes, message: bytes) -> bool:
        """Verify Ed25519 signature matches genesis founder."""
        try:
            from positronic.crypto.keys import KeyPair
            if not KeyPair.verify(pubkey, signature, message):
                return False

            from positronic.crypto.address import address_from_pubkey
            from positronic.core.genesis import get_genesis_founder_keypair

            caller_address = address_from_pubkey(pubkey)
            founder_kp = get_genesis_founder_keypair()
            return caller_address == founder_kp.address
        except Exception as e:
            logger.error(f"Founder verification failed: {e}")
            return False

    def _get_block_height(self) -> int:
        """Get current block height from node."""
        if self._node and hasattr(self._node, 'blockchain'):
            return self._node.blockchain.height
        return 0
