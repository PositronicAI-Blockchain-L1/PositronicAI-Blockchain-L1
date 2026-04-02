"""
Positronic - Validator Election per Epoch (Consensus v2)
At each epoch boundary ALL eligible validators (with MIN_STAKE) are
activated into the active set.  Block producers (21 per epoch) are
selected via stake × trust_multiplier weighted-random from the full set.

Weight formula: effective_weight = stake × trust_multiplier
  NEWCOMER   (0-99)     → 0.5x weight  (new validators earn less slots)
  APPRENTICE (100-499)  → 0.75x weight
  TRUSTED    (500-1999) → 1.0x weight  (baseline)
  VETERAN    (2000-9999)→ 1.5x weight
  LEGEND     (10000+)   → 2.0x weight  (double slot allocation)

The epoch seed is derived from the previous epoch's last block hash,
providing unpredictable but verifiable randomness for proposer ordering.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from positronic.crypto.hashing import sha512, hash_to_int
from positronic.consensus.validator import (
    Validator,
    ValidatorRegistry,
    ValidatorStatus,
)
from positronic.constants import (
    MAX_VALIDATORS,
    MIN_VALIDATORS,
    SLOTS_PER_EPOCH,
    MIN_STAKE,
    COMMITTEE_SIZE,
)


@dataclass(frozen=True)
class ElectionResult:
    """
    Immutable record of a validator election for a specific epoch.

    Attributes:
        epoch:            The epoch number this election applies to.
        epoch_seed:       32-byte seed derived from the previous epoch's
                          last block hash; used for proposer shuffling.
        active_set:       Ordered list of elected validators (by stake desc).
        proposer_order:   Mapping slot_in_epoch -> validator address for
                          deterministic block proposer assignment.
        total_active_stake: Sum of stakes in the active set.
    """
    epoch: int
    epoch_seed: bytes
    active_set: tuple  # tuple[Validator, ...] for immutability
    proposer_order: dict  # {slot_in_epoch: validator_address}
    total_active_stake: int

    @property
    def active_count(self) -> int:
        return len(self.active_set)

    @property
    def active_addresses(self) -> List[bytes]:
        return [v.address for v in self.active_set]

    @property
    def active_pubkeys(self) -> List[bytes]:
        return [v.pubkey for v in self.active_set]

    def get_proposer(self, slot_in_epoch: int) -> Optional[bytes]:
        """Return the validator address assigned to propose at *slot_in_epoch*."""
        return self.proposer_order.get(slot_in_epoch)

    def is_active(self, address: bytes) -> bool:
        """Check whether *address* is in the active set for this epoch."""
        return any(v.address == address for v in self.active_set)

    def validator_at_index(self, index: int) -> Optional[Validator]:
        if 0 <= index < len(self.active_set):
            return self.active_set[index]
        return None

    def to_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "epoch_seed": self.epoch_seed.hex(),
            "active_set": [v.to_dict() for v in self.active_set],
            "proposer_order": {
                str(slot): addr.hex()
                for slot, addr in self.proposer_order.items()
            },
            "total_active_stake": self.total_active_stake,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ElectionResult:
        active = tuple(Validator.from_dict(v) for v in d["active_set"])
        proposer_order = {
            int(slot): bytes.fromhex(addr)
            for slot, addr in d["proposer_order"].items()
        }
        return cls(
            epoch=d["epoch"],
            epoch_seed=bytes.fromhex(d["epoch_seed"]),
            active_set=active,
            proposer_order=proposer_order,
            total_active_stake=d["total_active_stake"],
        )

    def __repr__(self) -> str:
        """Human-readable representation of election result."""
        return (
            f"ElectionResult(epoch={self.epoch}, "
            f"validators={self.active_count}, "
            f"total_stake={self.total_active_stake})"
        )


class ValidatorElection:
    """
    Runs the deterministic validator election algorithm (Consensus v2).

    At each epoch boundary:
    1. Rank all eligible validators by total_stake descending.
    2. Activate ALL eligible validators (no top-N cap).
    3. Update their status to ACTIVE; demote ineligible to PENDING.
    4. Compute proposer assignment: 21 producers per epoch via weighted-random.
    """

    def __init__(self, registry: ValidatorRegistry) -> None:
        self._registry = registry
        # Cache the last election result
        self._last_result: Optional[ElectionResult] = None
        # History of election results by epoch
        self._history: Dict[int, ElectionResult] = {}

    @property
    def registry(self) -> ValidatorRegistry:
        return self._registry

    @property
    def last_result(self) -> Optional[ElectionResult]:
        return self._last_result

    def get_result(self, epoch: int) -> Optional[ElectionResult]:
        return self._history.get(epoch)

    # ------------------------------------------------------------------ #
    #  Core election                                                      #
    # ------------------------------------------------------------------ #

    def run_election(
        self,
        epoch: int,
        epoch_seed: bytes,
        trust_manager=None,
    ) -> ElectionResult:
        """
        Execute the election for *epoch*.

        Consensus v2: Three-Layer system
        1. ALL eligible validators are in the ACTIVE set (no cap).
        2. Per-slot proposer is selected from a COMMITTEE_SIZE subset
           via stake-weighted random, rotated each slot.
        3. This scales to 10,000+ validators while keeping slot production fast.

        Args:
            epoch:      The epoch number being elected.
            epoch_seed: Seed derived from the previous epoch's last block
                        hash (use genesis hash for epoch 0).

        Returns:
            An ElectionResult describing the new active set and proposer
            schedule for the epoch.

        Raises:
            ValueError if fewer than MIN_VALIDATORS are eligible.
        """
        # 1. Gather eligible candidates ranked by total_stake
        candidates = self._registry.ranked_by_stake(eligible_only=True)

        if len(candidates) < MIN_VALIDATORS:
            raise ValueError(
                f"Not enough eligible validators: {len(candidates)} "
                f"< MIN_VALIDATORS ({MIN_VALIDATORS})"
            )

        # 2. Activate ALL eligible validators (Consensus v2: no top-N cap)
        elected = candidates

        # 3. Update statuses in the registry
        elected_addrs = {v.address for v in elected}

        for v in self._registry.all_validators:
            if v.address in elected_addrs:
                self._registry.activate(v.address)
            elif v.status == ValidatorStatus.ACTIVE:
                # Was active last epoch but didn't make the cut
                self._registry.deactivate(v.address)

        # 4. Compute total active stake
        total_active_stake = sum(v.total_stake for v in elected)

        # 5. Build proposer order using committee sharding
        proposer_order = self._compute_proposer_order(
            elected, epoch_seed, epoch, trust_manager=trust_manager,
        )

        result = ElectionResult(
            epoch=epoch,
            epoch_seed=epoch_seed,
            active_set=tuple(elected),
            proposer_order=proposer_order,
            total_active_stake=total_active_stake,
        )

        self._last_result = result
        self._history[epoch] = result
        return result

    # ------------------------------------------------------------------ #
    #  Proposer assignment                                                #
    # ------------------------------------------------------------------ #

    def _compute_proposer_order(
        self,
        validators: List[Validator],
        epoch_seed: bytes,
        epoch: int,
        trust_manager=None,
    ) -> Dict[int, bytes]:
        """
        Deterministic proposer assignment with committee sharding.

        Weighting: effective_weight = stake × trust_multiplier
          - NEWCOMER (0.5x): new validators earn fewer slots until proven
          - APPRENTICE (0.75x): building track record
          - TRUSTED (1.0x): baseline
          - VETERAN (1.5x): experienced validators earn proportionally more
          - LEGEND (2.0x): elite validators earn double slots

        When validators > COMMITTEE_SIZE:
            For each slot, select a COMMITTEE_SIZE subset (committee),
            then weighted-select the proposer from the committee.

        When validators <= COMMITTEE_SIZE:
            All validators are in every committee (backward compatible).
        """
        if not validators:
            return {}

        def _weight(v: "Validator") -> int:
            """Compute effective weight = stake × trust_multiplier (integer)."""
            if trust_manager is None:
                return max(1, v.total_stake)
            mult = trust_manager.get_mining_multiplier(v.address)
            return max(1, int(v.total_stake * mult))

        total_weight = sum(_weight(v) for v in validators)
        if total_weight == 0:
            return {
                slot: validators[slot % len(validators)].address
                for slot in range(SLOTS_PER_EPOCH)
            }

        order: Dict[int, bytes] = {}
        for slot in range(SLOTS_PER_EPOCH):
            slot_bytes = struct.pack(">I", slot)
            slot_seed = sha512(epoch_seed + slot_bytes)

            # Select committee for this slot
            if len(validators) > COMMITTEE_SIZE:
                committee = self._select_committee(
                    validators, slot_seed, COMMITTEE_SIZE
                )
            else:
                committee = validators

            # Trust-weighted selection from committee
            committee_weight = sum(_weight(v) for v in committee)
            if committee_weight == 0:
                order[slot] = committee[slot % len(committee)].address
                continue

            # Build cumulative weights for committee
            cumulative = []
            running = 0
            for v in committee:
                running += _weight(v)
                cumulative.append((running, v.address))

            # Select proposer using trust-weighted random
            proposer_seed = sha512(slot_seed + b"PROPOSER")
            value = hash_to_int(proposer_seed) % committee_weight

            selected = cumulative[-1][1]
            for threshold, addr in cumulative:
                if value < threshold:
                    selected = addr
                    break

            order[slot] = selected

        # Fair rotation: balance slot distribution across validators,
        # proportional to each validator's effective weight (stake × trust).
        from collections import Counter
        import math
        n = len(validators)
        if n > 1:
            slot_counts = Counter(order.values())

            # Compute weight-proportional caps: higher trust = more slots allowed
            validator_caps: Dict[bytes, int] = {}
            validator_mins: Dict[bytes, int] = {}
            for v in validators:
                fair_share = (_weight(v) / total_weight) * SLOTS_PER_EPOCH
                validator_caps[v.address] = max(1, math.ceil(fair_share))
                validator_mins[v.address] = max(1, int(fair_share))

            # Step 1: give unassigned validators at least 1 slot
            unassigned = [v for v in validators if v.address not in slot_counts]
            for ua_val in unassigned:
                most_common_addr = slot_counts.most_common(1)[0][0]
                if slot_counts[most_common_addr] > 1:
                    for s in reversed(range(SLOTS_PER_EPOCH)):
                        if order[s] == most_common_addr:
                            order[s] = ua_val.address
                            slot_counts[most_common_addr] -= 1
                            slot_counts[ua_val.address] = 1
                            break

            # Step 2: redistribute from over-capped to least-assigned
            for _round in range(SLOTS_PER_EPOCH):  # bounded iterations
                over = [a for a, c in slot_counts.items()
                        if c > validator_caps.get(a, SLOTS_PER_EPOCH)]
                if not over:
                    break
                recv_candidates = sorted(
                    [v.address for v in validators
                     if slot_counts.get(v.address, 0) < validator_caps.get(v.address, SLOTS_PER_EPOCH)],
                    key=lambda a: slot_counts.get(a, 0),
                )
                if not recv_candidates:
                    break
                recv = recv_candidates[0]
                over_addr = over[0]
                reassigned = False
                for s in reversed(range(SLOTS_PER_EPOCH)):
                    if order[s] == over_addr:
                        order[s] = recv
                        slot_counts[over_addr] -= 1
                        slot_counts[recv] = slot_counts.get(recv, 0) + 1
                        reassigned = True
                        break
                if not reassigned:
                    break

        return order

    @staticmethod
    def _select_committee(
        validators: List[Validator],
        seed: bytes,
        size: int,
    ) -> List[Validator]:
        """
        Deterministically select a committee of 'size' validators.
        Uses Fisher-Yates shuffle seeded by the slot seed.
        """
        indices = list(range(len(validators)))
        # Fisher-Yates shuffle using seed
        for i in range(len(indices) - 1, 0, -1):
            idx_seed = sha512(seed + struct.pack(">I", i))
            j = hash_to_int(idx_seed) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        # Take first 'size' from shuffled indices
        selected = [validators[idx] for idx in indices[:size]]
        return selected

    # ------------------------------------------------------------------ #
    #  Epoch seed derivation                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def derive_epoch_seed(previous_block_hash: bytes, epoch: int) -> bytes:
        """
        Derive the epoch seed from the last block hash of the prior epoch.
        seed = SHA-512(previous_block_hash || epoch_number)
        For epoch 0, use a well-known genesis seed.
        """
        epoch_bytes = struct.pack(">Q", epoch)
        return sha512(previous_block_hash + epoch_bytes)

    @staticmethod
    def genesis_seed() -> bytes:
        """
        Deterministic seed for epoch 0 (before any blocks exist).
        seed = SHA-512(b"Positronic GENESIS EPOCH SEED")
        """
        return sha512(b"Positronic GENESIS EPOCH SEED")

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_proposer_for_slot(
        self, epoch: int, slot_in_epoch: int,
    ) -> Optional[bytes]:
        """
        Return the proposer address for a given (epoch, slot_in_epoch).
        Returns None if the election for that epoch hasn't been run.
        """
        result = self._history.get(epoch)
        if result is None:
            return None
        return result.get_proposer(slot_in_epoch)

    def is_validator_active(self, address: bytes, epoch: int) -> bool:
        """Check if *address* was active in a specific epoch."""
        result = self._history.get(epoch)
        if result is None:
            return False
        return result.is_active(address)

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "history": {
                str(e): r.to_dict() for e, r in self._history.items()
            },
        }

    @classmethod
    def from_dict(
        cls, d: dict, registry: ValidatorRegistry,
    ) -> ValidatorElection:
        election = cls(registry)
        for epoch_str, rd in d.get("history", {}).items():
            result = ElectionResult.from_dict(rd)
            election._history[int(epoch_str)] = result
        if election._history:
            latest = max(election._history.keys())
            election._last_result = election._history[latest]
        return election

    def __repr__(self) -> str:
        return (
            f"ValidatorElection(epochs_run={len(self._history)}, "
            f"latest={self._last_result.epoch if self._last_result else None})"
        )
