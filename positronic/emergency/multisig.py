"""
Positronic - Multi-Signature Manager
3-of-5 Ed25519 multi-sig for critical network actions (halt, rollback, upgrade).
"""

import time
import uuid
import hashlib
import json
import logging
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from positronic.constants import (
    EMERGENCY_MULTISIG_REQUIRED,
    EMERGENCY_MULTISIG_TOTAL,
    EMERGENCY_TIMELOCK_HALT,
    EMERGENCY_TIMELOCK_RESUME,
    EMERGENCY_TIMELOCK_ROLLBACK,
    EMERGENCY_TIMELOCK_UPGRADE,
    EMERGENCY_TIMELOCK_KEY_ROTATION,
)

logger = logging.getLogger("positronic.emergency.multisig")


class MultiSigAction(IntEnum):
    """Types of actions that require multi-sig approval."""
    EMERGENCY_HALT = 1
    RESUME_FROM_HALT = 2
    ROLLBACK = 3
    UPGRADE_APPROVE = 4
    KEY_ROTATION = 5
    PAUSE_NETWORK = 6
    RESUME_NETWORK = 7


class ActionStatus(IntEnum):
    """Status of a pending multi-sig action."""
    PENDING = 0
    READY = 1       # Enough signatures, waiting for timelock
    EXECUTED = 2
    EXPIRED = 3
    CANCELLED = 4


# Timelock per action type (seconds)
_TIMELOCKS = {
    MultiSigAction.EMERGENCY_HALT: EMERGENCY_TIMELOCK_HALT,
    MultiSigAction.RESUME_FROM_HALT: EMERGENCY_TIMELOCK_RESUME,
    MultiSigAction.ROLLBACK: EMERGENCY_TIMELOCK_ROLLBACK,
    MultiSigAction.UPGRADE_APPROVE: EMERGENCY_TIMELOCK_UPGRADE,
    MultiSigAction.KEY_ROTATION: EMERGENCY_TIMELOCK_KEY_ROTATION,
    MultiSigAction.PAUSE_NETWORK: EMERGENCY_TIMELOCK_HALT,    # Immediate (0s)
    MultiSigAction.RESUME_NETWORK: EMERGENCY_TIMELOCK_RESUME,  # 1h cooldown
}


@dataclass
class PendingAction:
    """A multi-sig action awaiting signatures."""
    action_id: str
    action_type: MultiSigAction
    params: Dict[str, Any]
    proposer: bytes
    signatures: Dict[str, bytes] = field(default_factory=dict)  # pubkey_hex -> signature
    created_at: float = 0.0
    executed_at: Optional[float] = None
    status: ActionStatus = ActionStatus.PENDING


class MultiSigManager:
    """
    Manages 3-of-5 multi-signature authorization for critical network operations.

    Lifecycle:
    1. register_keys() -- set 5 authorized public keys
    2. create_action() -- proposer creates an action (auto-signs)
    3. sign_action()   -- other signers add their signatures
    4. is_executable() -- check if enough sigs + timelock passed
    5. execute()       -- mark action as executed
    """

    def __init__(self):
        self.required: int = EMERGENCY_MULTISIG_REQUIRED
        self.total: int = EMERGENCY_MULTISIG_TOTAL
        self.registered_keys: List[bytes] = []
        self.pending_actions: Dict[str, PendingAction] = {}

    def register_keys(self, pubkeys: List[bytes]) -> bool:
        """Register the authorized signer public keys."""
        if len(pubkeys) != self.total:
            logger.error(f"Expected {self.total} keys, got {len(pubkeys)}")
            return False
        self.registered_keys = list(pubkeys)
        logger.info(f"Registered {len(pubkeys)} multi-sig keys")
        return True

    def get_signing_message(self, action_id: str, action_type: MultiSigAction, params: dict) -> bytes:
        """
        Build the canonical message that signers must sign.
        Format: action_id + action_type + sha256(sorted params JSON)
        This prevents replay attacks and ensures each signer explicitly approves
        the specific action.
        """
        params_json = json.dumps(params, sort_keys=True, separators=(',', ':'))
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()
        message = f"{action_id}:{int(action_type)}:{params_hash}"
        return message.encode()

    def create_action(
        self,
        action_type: MultiSigAction,
        params: dict,
        proposer_pubkey: bytes,
        signature: bytes,
    ) -> Optional[str]:
        """
        Create a new pending action. Proposer auto-signs.
        Returns action_id or None on failure.
        """
        if not self._is_registered(proposer_pubkey):
            logger.warning("Proposer not in registered keys")
            return None

        action_id = str(uuid.uuid4())

        # Verify signature (proposer signs with empty action_id initially)
        message = self.get_signing_message("", action_type, params)
        if not self._verify_sig(proposer_pubkey, signature, message):
            logger.warning("Invalid proposer signature")
            return None

        pubkey_hex = proposer_pubkey.hex()
        action = PendingAction(
            action_id=action_id,
            action_type=action_type,
            params=params,
            proposer=proposer_pubkey,
            signatures={pubkey_hex: signature},
            created_at=time.time(),
            status=ActionStatus.PENDING,
        )
        self.pending_actions[action_id] = action
        logger.info(f"Action {action_id} created: {action_type.name}")
        return action_id

    def sign_action(self, action_id: str, signer_pubkey: bytes, signature: bytes) -> bool:
        """Add a signature to a pending action."""
        action = self.pending_actions.get(action_id)
        if not action:
            logger.warning(f"Action {action_id} not found")
            return False

        if action.status not in (ActionStatus.PENDING, ActionStatus.READY):
            logger.warning(f"Action {action_id} is {action.status.name}, cannot sign")
            return False

        if not self._is_registered(signer_pubkey):
            logger.warning("Signer not in registered keys")
            return False

        pubkey_hex = signer_pubkey.hex()
        if pubkey_hex in action.signatures:
            logger.warning("Signer already signed this action")
            return False

        # Verify signature
        message = self.get_signing_message(action_id, action.action_type, action.params)
        if not self._verify_sig(signer_pubkey, signature, message):
            logger.warning("Invalid signer signature")
            return False

        action.signatures[pubkey_hex] = signature
        logger.info(f"Action {action_id}: signature {len(action.signatures)}/{self.required}")

        if len(action.signatures) >= self.required:
            action.status = ActionStatus.READY

        return True

    def is_executable(self, action_id: str) -> bool:
        """Check if action has enough signatures AND timelock has passed."""
        action = self.pending_actions.get(action_id)
        if not action:
            return False

        if action.status == ActionStatus.EXECUTED:
            return False

        # Check signature count
        if len(action.signatures) < self.required:
            return False

        # Check timelock
        timelock = _TIMELOCKS.get(action.action_type, 0)
        elapsed = time.time() - action.created_at
        if elapsed < timelock:
            return False

        return True

    def execute(self, action_id: str) -> bool:
        """Mark action as executed."""
        action = self.pending_actions.get(action_id)
        if not action:
            return False

        if not self.is_executable(action_id):
            return False

        action.status = ActionStatus.EXECUTED
        action.executed_at = time.time()
        logger.info(f"Action {action_id} executed: {action.action_type.name}")
        return True

    def cancel_action(self, action_id: str, proposer_pubkey: bytes) -> bool:
        """Cancel a pending action (proposer only)."""
        action = self.pending_actions.get(action_id)
        if not action:
            return False

        if action.status not in (ActionStatus.PENDING, ActionStatus.READY):
            return False

        if action.proposer != proposer_pubkey:
            logger.warning("Only the proposer can cancel an action")
            return False

        action.status = ActionStatus.CANCELLED
        logger.info(f"Action {action_id} cancelled")
        return True

    def get_action(self, action_id: str) -> Optional[PendingAction]:
        """Get a specific action by ID."""
        return self.pending_actions.get(action_id)

    def get_pending_actions(self) -> List[PendingAction]:
        """Get all PENDING or READY actions."""
        return [
            a for a in self.pending_actions.values()
            if a.status in (ActionStatus.PENDING, ActionStatus.READY)
        ]

    # === Internal helpers ===

    def _is_registered(self, pubkey: bytes) -> bool:
        """Check if a public key is in the registered set."""
        return pubkey in self.registered_keys

    def _verify_sig(self, pubkey: bytes, signature: bytes, message: bytes) -> bool:
        """Verify an Ed25519 signature."""
        try:
            from positronic.crypto.keys import KeyPair
            return KeyPair.verify(pubkey, signature, message)
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False
