"""
Positronic - Attestation Tracking (Consensus v2)
All active validators attest (vote) on each block.
Attestation rewards are distributed pro-rata by stake.

Layer 2 of the Three-Layer system:
  Block Producers (Layer 1) propose blocks.
  Attesters (Layer 2) vote on block validity — tracked here.
  Node Operators (Layer 3) relay and earn relay fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Attestation:
    """A single validator attestation for a block."""
    validator: bytes
    stake: int


class AttestationTracker:
    """
    Tracks validator attestations (votes) for blocks.
    Distributes attestation rewards pro-rata by stake.

    Usage:
        1. After block production, active validators submit attestations.
        2. record_attestation() stores each vote (deduplicated per validator).
        3. distribute_attestation_rewards() splits reward pro-rata.
        4. clear_block() removes processed attestations.
    """

    def __init__(self) -> None:
        # block_hash -> list of Attestation
        self._attestations: Dict[bytes, List[Attestation]] = {}
        # block_hash -> set of validator addresses (dedup tracking)
        self._attested_validators: Dict[bytes, set] = {}
        # Counters for stats
        self._total_recorded = 0
        self._total_distributed = 0
        self._blocks_processed = 0

    def record_attestation(
        self,
        block_hash: bytes,
        validator: bytes,
        stake: int,
    ) -> bool:
        """
        Record a validator attestation for a block.
        Returns True if recorded, False if duplicate.
        """
        # Dedup: one attestation per validator per block
        if block_hash not in self._attested_validators:
            self._attested_validators[block_hash] = set()

        if validator in self._attested_validators[block_hash]:
            return False

        self._attested_validators[block_hash].add(validator)

        if block_hash not in self._attestations:
            self._attestations[block_hash] = []
        self._attestations[block_hash].append(Attestation(validator, stake))
        self._total_recorded += 1
        return True

    def get_attestations(self, block_hash: bytes) -> List[Attestation]:
        """Get all attestations for a block."""
        return self._attestations.get(block_hash, [])

    def get_total_attesting_stake(self, block_hash: bytes) -> int:
        """Total stake that attested to a block."""
        return sum(a.stake for a in self.get_attestations(block_hash))

    def get_attester_count(self, block_hash: bytes) -> int:
        """Number of validators that attested to a block."""
        return len(self.get_attestations(block_hash))

    def distribute_attestation_rewards(
        self,
        block_hash: bytes,
        total_reward: int,
    ) -> Dict[bytes, int]:
        """
        Distribute attestation reward pro-rata by stake.
        Returns mapping of {validator_address: payout_amount}.
        """
        attestations = self.get_attestations(block_hash)
        total_stake = sum(a.stake for a in attestations)
        if total_stake == 0:
            return {}

        payouts: Dict[bytes, int] = {}
        distributed = 0
        for i, a in enumerate(attestations):
            if i == len(attestations) - 1:
                # Last attester gets remainder to avoid rounding dust
                share = total_reward - distributed
            else:
                share = int(total_reward * a.stake / total_stake)
            payouts[a.validator] = share
            distributed += share

        self._total_distributed += distributed
        self._blocks_processed += 1
        return payouts

    def clear_block(self, block_hash: bytes) -> None:
        """Remove attestations for a processed block."""
        self._attestations.pop(block_hash, None)
        self._attested_validators.pop(block_hash, None)

    def get_stats(self) -> dict:
        """Return attestation tracking statistics."""
        return {
            "tracked_blocks": len(self._attestations),
            "total_attestations": sum(
                len(v) for v in self._attestations.values()
            ),
            "total_recorded": self._total_recorded,
            "total_distributed": self._total_distributed,
            "blocks_processed": self._blocks_processed,
        }

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            "attestations": {
                bh.hex(): [{"validator": a.validator.hex(), "stake": a.stake}
                           for a in atts]
                for bh, atts in self._attestations.items()
            },
            "stats": {
                "total_recorded": self._total_recorded,
                "total_distributed": self._total_distributed,
                "blocks_processed": self._blocks_processed,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> AttestationTracker:
        """Deserialize from state."""
        tracker = cls()
        for bh_hex, atts in d.get("attestations", {}).items():
            bh = bytes.fromhex(bh_hex)
            for a in atts:
                tracker.record_attestation(
                    bh,
                    bytes.fromhex(a["validator"]),
                    a["stake"],
                )
        stats = d.get("stats", {})
        tracker._total_recorded = stats.get("total_recorded", 0)
        tracker._total_distributed = stats.get("total_distributed", 0)
        tracker._blocks_processed = stats.get("blocks_processed", 0)
        return tracker

    def __repr__(self) -> str:
        return (
            f"AttestationTracker(blocks={len(self._attestations)}, "
            f"total_recorded={self._total_recorded})"
        )
