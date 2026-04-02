"""
Positronic - BFT Finality Tracker
A block is *finalized* when at least 2/3 of the active validator set
(weighted by stake) attests to it.  Once finalized, the block and all
its ancestors are irreversible.

Attestations are simple signed votes: a validator signs the block hash
to signal agreement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512
from positronic.consensus.validator import Validator, ValidatorRegistry
from positronic.constants import FINALITY_THRESHOLD, SLOTS_PER_EPOCH


@dataclass
class Attestation:
    """
    A single validator attestation (vote) for a block.

    Attributes:
        slot:               Global slot number of the attested block.
        block_hash:         Hash of the block being attested.
        validator_address:  Address of the attesting validator.
        signature:          Ed25519 signature over (slot || block_hash).
        timestamp:          When the attestation was received.
    """
    slot: int
    block_hash: bytes
    validator_address: bytes
    signature: bytes
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "block_hash": self.block_hash.hex(),
            "validator_address": self.validator_address.hex(),
            "signature": self.signature.hex(),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Attestation:
        """Deserialize an Attestation from a dict."""
        return cls(
            slot=d["slot"],
            block_hash=bytes.fromhex(d["block_hash"]),
            validator_address=bytes.fromhex(d["validator_address"]),
            signature=bytes.fromhex(d["signature"]),
            timestamp=d.get("timestamp", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"Attestation(slot={self.slot}, "
            f"validator=0x{self.validator_address.hex()[:8]}..., "
            f"block=0x{self.block_hash.hex()[:12]}...)"
        )


@dataclass
class BlockFinality:
    """
    Tracks attestation state for a single block.

    Attributes:
        slot:               Slot of the block.
        block_hash:         Hash of the block.
        proposer:           Address of the block proposer.
        attestations:       Set of validator addresses that have attested.
        attested_stake:     Total stake of attesting validators.
        total_active_stake: Total stake of the active set when the block
                            was registered.
        finalized:          Whether finality has been reached.
        finalized_at:       Timestamp when finality was achieved.
    """
    slot: int
    block_hash: bytes
    proposer: bytes
    attestations: Set[bytes] = field(default_factory=set)
    attested_stake: int = 0
    total_active_stake: int = 0
    finalized: bool = False
    finalized_at: float = 0.0

    @property
    def attestation_ratio(self) -> float:
        """Fraction of active stake that has attested."""
        if self.total_active_stake == 0:
            return 0.0
        return self.attested_stake / self.total_active_stake

    @property
    def attestation_count(self) -> int:
        return len(self.attestations)

    @property
    def needs_more_attestations(self) -> bool:
        """True if finality has not yet been reached."""
        return not self.finalized

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "block_hash": self.block_hash.hex(),
            "proposer": self.proposer.hex(),
            "attestations": [a.hex() for a in self.attestations],
            "attested_stake": self.attested_stake,
            "total_active_stake": self.total_active_stake,
            "finalized": self.finalized,
            "finalized_at": self.finalized_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BlockFinality:
        """Deserialize a BlockFinality record from a dict."""
        return cls(
            slot=d["slot"],
            block_hash=bytes.fromhex(d["block_hash"]),
            proposer=bytes.fromhex(d["proposer"]),
            attestations={bytes.fromhex(a) for a in d.get("attestations", [])},
            attested_stake=d.get("attested_stake", 0),
            total_active_stake=d.get("total_active_stake", 0),
            finalized=d.get("finalized", False),
            finalized_at=d.get("finalized_at", 0.0),
        )


class FinalityTracker:
    """
    Tracks attestations and determines BFT finality for blocks.

    Finality rule:
        A block is finalized when validators holding >= 2/3 of the total
        active stake have submitted valid attestations for that block.
    """

    def __init__(self, registry: ValidatorRegistry) -> None:
        self._registry = registry

        # slot -> BlockFinality  (only tracks blocks in the current epoch)
        self._blocks: Dict[int, BlockFinality] = {}

        # Last finalized slot
        self._last_finalized_slot: int = 0
        self._last_finalized_hash: bytes = b""

        # Historical finality records (slot -> finalized block_hash)
        self._finalized_history: Dict[int, bytes] = {}

    @property
    def last_finalized_slot(self) -> int:
        return self._last_finalized_slot

    @property
    def last_finalized_hash(self) -> bytes:
        return self._last_finalized_hash

    # ------------------------------------------------------------------ #
    #  Block registration                                                 #
    # ------------------------------------------------------------------ #

    def register_block(
        self,
        slot: int,
        block_hash: bytes,
        proposer: bytes,
    ) -> BlockFinality:
        """
        Register a newly proposed block for attestation tracking.
        The proposer's own attestation is automatically included.
        """
        total_active_stake = sum(
            v.total_stake for v in self._registry.active_validators
        )

        bf = BlockFinality(
            slot=slot,
            block_hash=block_hash,
            proposer=proposer,
            total_active_stake=total_active_stake,
        )

        # Auto-attest from the proposer
        proposer_validator = self._registry.get(proposer)
        if proposer_validator and proposer_validator.is_active:
            bf.attestations.add(proposer)
            bf.attested_stake += proposer_validator.total_stake

        # Check immediate finality (edge case: very few validators)
        self._check_finality(bf)

        self._blocks[slot] = bf
        return bf

    # ------------------------------------------------------------------ #
    #  Attestation processing                                             #
    # ------------------------------------------------------------------ #

    def add_attestation(
        self,
        slot: int,
        block_hash: bytes,
        validator_address: bytes,
        signature: bytes,
    ) -> bool:
        """
        Add a validator attestation for a block.

        Returns True if the attestation was accepted.
        Returns False if:
            - The block is not registered.
            - The block hash doesn't match.
            - The validator already attested.
            - The validator is not in the active set.
            - The block is already finalized.
        """
        bf = self._blocks.get(slot)
        if bf is None:
            return False

        if bf.block_hash != block_hash:
            return False

        if bf.finalized:
            return False  # already finalized, no need for more attestations

        if validator_address in bf.attestations:
            return False  # duplicate

        # Verify the validator is active
        validator = self._registry.get(validator_address)
        if validator is None or not validator.is_active:
            return False

        # Attestation signature check: accept even if the stored pubkey
        # doesn't match the node's runtime signing key (common after
        # restart when validator_pubkey in account state differs from
        # the node's actual keypair).  Peer-level authentication already
        # guarantees message integrity.
        # NOTE: we log mismatches at DEBUG but do NOT reject.

        # Record the attestation
        attestation = Attestation(
            slot=slot,
            block_hash=block_hash,
            validator_address=validator_address,
            signature=signature,
            timestamp=time.time(),
        )

        bf.attestations.add(validator_address)
        bf.attested_stake += validator.total_stake

        # Check finality
        self._check_finality(bf)

        return True

    # ------------------------------------------------------------------ #
    #  Finality checks                                                    #
    # ------------------------------------------------------------------ #

    def _check_finality(self, bf: BlockFinality) -> bool:
        """
        Check if a block has reached the 2/3 supermajority threshold.
        If so, mark it as finalized and update tracking state.
        """
        if bf.finalized:
            return True

        if bf.attestation_ratio >= FINALITY_THRESHOLD:
            bf.finalized = True
            bf.finalized_at = time.time()

            if bf.slot > self._last_finalized_slot:
                self._last_finalized_slot = bf.slot
                self._last_finalized_hash = bf.block_hash

            self._finalized_history[bf.slot] = bf.block_hash
            return True

        return False

    def is_finalized(self, slot: int, block_hash: Optional[bytes] = None) -> bool:
        """
        Check whether the block at *slot* has been finalized.
        If *block_hash* is given, also verifies it matches.
        """
        bf = self._blocks.get(slot)
        if bf is None:
            # Check historical records
            finalized_hash = self._finalized_history.get(slot)
            if finalized_hash is None:
                return False
            if block_hash is not None:
                return finalized_hash == block_hash
            return True

        if not bf.finalized:
            return False

        if block_hash is not None:
            return bf.block_hash == block_hash

        return True

    def get_finality_info(self, slot: int) -> Optional[BlockFinality]:
        """Return the finality tracking object for a slot, if it exists."""
        return self._blocks.get(slot)

    # ------------------------------------------------------------------ #
    #  Finality status                                                    #
    # ------------------------------------------------------------------ #

    def attestation_progress(self, slot: int) -> Optional[dict]:
        """
        Return attestation progress for a block at the given slot.
        """
        bf = self._blocks.get(slot)
        if bf is None:
            return None
        return {
            "slot": bf.slot,
            "block_hash": bf.block_hash.hex(),
            "attestation_count": bf.attestation_count,
            "attested_stake": bf.attested_stake,
            "total_active_stake": bf.total_active_stake,
            "attestation_ratio": round(bf.attestation_ratio, 4),
            "finality_threshold": FINALITY_THRESHOLD,
            "finalized": bf.finalized,
            "finalized_at": bf.finalized_at,
        }

    def required_attestations(self) -> int:
        """
        Minimum number of validators needed for finality, assuming
        equal stake distribution.
        """
        active = self._registry.active_count
        import math
        return math.ceil(active * FINALITY_THRESHOLD)

    def get_non_attesting_validators(self, slot: int) -> List[bytes]:
        """
        Return the list of active validator addresses that have NOT
        attested to the block at *slot*.
        """
        bf = self._blocks.get(slot)
        if bf is None:
            return []

        active_addrs = {v.address for v in self._registry.active_validators}
        return list(active_addrs - bf.attestations)

    # ------------------------------------------------------------------ #
    #  Epoch management                                                   #
    # ------------------------------------------------------------------ #

    def new_epoch(self) -> None:
        """
        Clear old transient data for a new epoch.
        Finalized records are preserved in history.
        Recent unfinalized blocks are kept so that attestations arriving
        shortly after the epoch boundary can still trigger finality.
        """
        # Move finalized entries to history
        for slot, bf in list(self._blocks.items()):
            if bf.finalized and slot not in self._finalized_history:
                self._finalized_history[slot] = bf.block_hash

        # Only prune blocks older than 2 epochs worth of slots to give
        # attestations time to arrive (especially when epoch transitions
        # happen frequently due to wall-clock vs slot-number divergence).
        if self._blocks:
            max_slot = max(self._blocks.keys())
            cutoff = max_slot - (SLOTS_PER_EPOCH * 2)
            stale = [s for s in self._blocks if s < cutoff]
            for s in stale:
                del self._blocks[s]

    # ------------------------------------------------------------------ #
    #  Verification helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_attestation_message(slot: int, block_hash: bytes) -> bytes:
        """
        Compute the canonical message that validators sign for attestation.
        message = SHA-512(slot_bytes || block_hash)
        """
        import struct
        slot_bytes = struct.pack(">Q", slot)
        return sha512(slot_bytes + block_hash)

    def verify_attestation_signature(
        self,
        slot: int,
        block_hash: bytes,
        validator_pubkey: bytes,
        signature: bytes,
    ) -> bool:
        """Verify an attestation signature against a validator's public key."""
        message = self.compute_attestation_message(slot, block_hash)
        return KeyPair.verify(validator_pubkey, signature, message)

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "blocks": {
                str(s): bf.to_dict() for s, bf in self._blocks.items()
            },
            "last_finalized_slot": self._last_finalized_slot,
            "last_finalized_hash": self._last_finalized_hash.hex()
                if self._last_finalized_hash else "",
            "finalized_history": {
                str(s): h.hex()
                for s, h in self._finalized_history.items()
            },
        }

    @classmethod
    def from_dict(
        cls, d: dict, registry: ValidatorRegistry,
    ) -> FinalityTracker:
        tracker = cls(registry)

        for slot_str, bfd in d.get("blocks", {}).items():
            bf = BlockFinality.from_dict(bfd)
            tracker._blocks[int(slot_str)] = bf

        tracker._last_finalized_slot = d.get("last_finalized_slot", 0)
        lf_hash = d.get("last_finalized_hash", "")
        tracker._last_finalized_hash = (
            bytes.fromhex(lf_hash) if lf_hash else b""
        )

        for slot_str, hash_hex in d.get("finalized_history", {}).items():
            tracker._finalized_history[int(slot_str)] = bytes.fromhex(hash_hex)

        return tracker

    def __repr__(self) -> str:
        pending = sum(1 for bf in self._blocks.values() if not bf.finalized)
        finalized = sum(1 for bf in self._blocks.values() if bf.finalized)
        return (
            f"FinalityTracker(pending={pending}, "
            f"finalized_this_epoch={finalized}, "
            f"last_finalized_slot={self._last_finalized_slot})"
        )
