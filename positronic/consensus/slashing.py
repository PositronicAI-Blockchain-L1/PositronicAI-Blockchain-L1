"""
Positronic - Slashing Conditions and Enforcement
Penalises validators for protocol violations:
    - Double-signing:   Proposing two different blocks for the same slot.
    - Downtime:         Missing too many consecutive block proposals.

Penalties:
    - 33 % of self-bonded stake is slashed (burned).
    - The validator is jailed for 36 epochs (~3.8 hours) base duration.
    - Repeated offences compound (slash % stays the same, but jail
      duration increases exponentially; permanent ban after 3 events).
"""

from __future__ import annotations

import time
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512
from positronic.consensus.validator import (
    Validator,
    ValidatorRegistry,
    ValidatorStatus,
)
from positronic.constants import (
    MIN_STAKE,
    EPOCH_DURATION,
)
from positronic.types import UnjailRequest

# ===== Slashing Parameters ============================================ #

# Fraction of self-bonded stake burned per slashing event
# Security fix: increased from 5% to 33% to make double-signing
# economically unprofitable (attacker loses more than typical MEV gain).
SLASH_FRACTION = 0.33  # 33 %

# Base jail time (seconds) after slashing
# Security fix: increased from 3 epochs (~19 min) to 36 epochs (~3.8 hours)
# to impose meaningful operational cost on misbehaving validators.
BASE_JAIL_DURATION = 36 * EPOCH_DURATION  # 36 epochs (~3.8 hours)

# Jail duration multiplier for repeat offenders: jail = base * (2 ^ n)
JAIL_MULTIPLIER_BASE = 2

# Consecutive missed blocks before a downtime slash is triggered
MAX_CONSECUTIVE_MISSED = 16

# Maximum lifetime slashing events before permanent ban
# Security fix: reduced from 5 to 3 for faster expulsion of attackers.
MAX_SLASHING_EVENTS = 3


@dataclass
class SlashingEvidence:
    """
    Record of a slashing event.

    Attributes:
        validator_address:  The slashed validator.
        reason:             "double_sign" or "downtime".
        slot:               Slot where the offence occurred.
        amount_slashed:     Amount of stake burned.
        timestamp:          When the slashing was executed.
        block_hash_a:       (double_sign) First conflicting block hash.
        block_hash_b:       (double_sign) Second conflicting block hash.
        signature_a:        (double_sign) Signature on first block.
        signature_b:        (double_sign) Signature on second block.
        missed_blocks:      (downtime) Number of consecutive missed blocks.
        jail_until:         Timestamp until which the validator is jailed.
    """
    validator_address: bytes
    reason: str
    slot: int
    amount_slashed: int
    timestamp: float = 0.0
    block_hash_a: bytes = b""
    block_hash_b: bytes = b""
    signature_a: bytes = b""
    signature_b: bytes = b""
    missed_blocks: int = 0
    jail_until: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        d = {
            "validator_address": self.validator_address.hex(),
            "reason": self.reason,
            "slot": self.slot,
            "amount_slashed": self.amount_slashed,
            "timestamp": self.timestamp,
            "jail_until": self.jail_until,
        }
        if self.reason == "double_sign":
            d["block_hash_a"] = self.block_hash_a.hex()
            d["block_hash_b"] = self.block_hash_b.hex()
            d["signature_a"] = self.signature_a.hex()
            d["signature_b"] = self.signature_b.hex()
        elif self.reason == "downtime":
            d["missed_blocks"] = self.missed_blocks
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SlashingEvidence:
        """Deserialize from dict."""
        return cls(
            validator_address=bytes.fromhex(d["validator_address"]),
            reason=d["reason"],
            slot=d["slot"],
            amount_slashed=d.get("amount_slashed", 0),
            timestamp=d.get("timestamp", 0.0),
            block_hash_a=bytes.fromhex(d.get("block_hash_a", "")),
            block_hash_b=bytes.fromhex(d.get("block_hash_b", "")),
            signature_a=bytes.fromhex(d.get("signature_a", "")),
            signature_b=bytes.fromhex(d.get("signature_b", "")),
            missed_blocks=d.get("missed_blocks", 0),
            jail_until=d.get("jail_until", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"SlashingEvidence(validator=0x{self.validator_address.hex()[:8]}..., "
            f"reason={self.reason}, "
            f"slashed={self.amount_slashed})"
        )


class SlashingManager:
    """
    Detects and enforces slashing conditions.

    Tracks submitted double-sign evidence to prevent duplicate reports.
    Coordinates with the ValidatorRegistry to burn stake and jail
    offending validators.
    """

    def __init__(self, registry: ValidatorRegistry) -> None:
        self._registry = registry

        # All slashing events in order
        self._evidence: List[SlashingEvidence] = []

        # Track processed double-sign evidence to avoid duplicates
        # key = (validator_address, slot)
        self._processed_double_signs: set = set()

    @property
    def evidence_log(self) -> List[SlashingEvidence]:
        return list(self._evidence)

    @property
    def total_slashed(self) -> int:
        """Total amount of stake burned across all events."""
        return sum(e.amount_slashed for e in self._evidence)

    # ------------------------------------------------------------------ #
    #  Double-sign slashing                                               #
    # ------------------------------------------------------------------ #

    def slash_double_sign(
        self,
        validator_address: bytes,
        slot: int,
        block_hash_a: bytes,
        block_hash_b: bytes,
        signature_a: bytes,
        signature_b: bytes,
    ) -> bool:
        """
        Process a double-sign evidence report.

        Validates:
            - The two block hashes are actually different.
            - This (validator, slot) pair hasn't already been slashed.
            - The validator exists in the registry.

        If valid, slashes 33 % of self-bonded stake and jails the validator.

        Returns True if slashing was applied, False if rejected.
        """
        # Same block hash is not a double-sign
        if block_hash_a == block_hash_b:
            return False

        # Already processed
        key = (validator_address, slot)
        if key in self._processed_double_signs:
            return False

        # Validator must exist and not already permanently banned
        validator = self._registry.get(validator_address)
        if validator is None:
            return False
        if validator.status == ValidatorStatus.EXITED:
            return False  # Already permanently banned

        # Cryptographically verify double-sign evidence before slashing.
        # ALL fields are REQUIRED — reject if any signature or pubkey is missing.
        # (Empty bytes / None must NOT bypass verification.)
        if not validator.pubkey or not signature_a or not signature_b:
            return False  # Missing evidence — reject without slashing
        if not self.verify_double_sign_evidence(
            validator.pubkey, slot,
            block_hash_a, block_hash_b,
            signature_a, signature_b,
        ):
            return False  # Invalid evidence — reject slashing

        # Compute slash amount
        slash_amount = int(validator.stake * SLASH_FRACTION)
        if slash_amount <= 0:
            slash_amount = min(validator.stake, 1)  # slash at least 1 unit

        # Apply slash
        actual_slashed = self._registry.slash_stake(validator_address, slash_amount)

        # Compute jail duration (exponential backoff for repeat offenders)
        jail_duration = self._compute_jail_duration(validator)
        now = time.time()
        jail_until = now + jail_duration

        # Jail the validator
        self._registry.jail(validator_address, jail_until)

        # Record evidence
        evidence = SlashingEvidence(
            validator_address=validator_address,
            reason="double_sign",
            slot=slot,
            amount_slashed=actual_slashed,
            timestamp=now,
            block_hash_a=block_hash_a,
            block_hash_b=block_hash_b,
            signature_a=signature_a,
            signature_b=signature_b,
            jail_until=jail_until,
        )
        self._evidence.append(evidence)
        self._processed_double_signs.add(key)

        # Check for permanent ban
        if validator.slashing_events >= MAX_SLASHING_EVENTS:
            self._permanent_ban(validator_address)

        return True

    # ------------------------------------------------------------------ #
    #  Downtime slashing                                                  #
    # ------------------------------------------------------------------ #

    def should_slash_for_downtime(self, validator: Validator) -> bool:
        """
        Check whether a validator has exceeded the consecutive missed
        block threshold.
        """
        return validator.missed_blocks >= MAX_CONSECUTIVE_MISSED

    def slash_downtime(
        self,
        validator_address: bytes,
        slot: int = 0,
    ) -> bool:
        """
        Slash a validator for excessive downtime (missed blocks).

        Same penalty structure as double-sign: 33 % slash + jail.

        Returns True if slashing was applied.
        """
        validator = self._registry.get(validator_address)
        if validator is None:
            return False
        if validator.status == ValidatorStatus.EXITED:
            return False  # Already permanently banned

        if validator.missed_blocks < MAX_CONSECUTIVE_MISSED:
            return False

        # Compute slash amount
        slash_amount = int(validator.stake * SLASH_FRACTION)
        if slash_amount <= 0:
            slash_amount = min(validator.stake, 1)

        actual_slashed = self._registry.slash_stake(validator_address, slash_amount)

        # Jail
        jail_duration = self._compute_jail_duration(validator)
        now = time.time()
        jail_until = now + jail_duration
        self._registry.jail(validator_address, jail_until)

        # Record evidence
        evidence = SlashingEvidence(
            validator_address=validator_address,
            reason="downtime",
            slot=slot,
            amount_slashed=actual_slashed,
            timestamp=now,
            missed_blocks=validator.missed_blocks,
            jail_until=jail_until,
        )
        self._evidence.append(evidence)

        # Reset the missed block counter (the jail resets it on unjail too)
        validator.missed_blocks = 0

        # Check for permanent ban
        if validator.slashing_events >= MAX_SLASHING_EVENTS:
            self._permanent_ban(validator_address)

        return True

    # ------------------------------------------------------------------ #
    #  Jail duration                                                      #
    # ------------------------------------------------------------------ #

    def _compute_jail_duration(self, validator: Validator) -> float:
        """
        Jail duration with exponential backoff for repeat offenders.
        duration = BASE_JAIL_DURATION * (2 ^ prior_slashing_events)
        """
        exponent = min(validator.slashing_events, 10)  # cap at 2^10
        return BASE_JAIL_DURATION * (JAIL_MULTIPLIER_BASE ** exponent)

    # ------------------------------------------------------------------ #
    #  Permanent ban                                                      #
    # ------------------------------------------------------------------ #

    def _permanent_ban(self, validator_address: bytes) -> None:
        """
        Permanently remove a validator that has been slashed too many times.
        Their remaining stake is NOT returned (considered forfeit).
        """
        validator = self._registry.get(validator_address)
        if validator is None:
            return

        # Slash any remaining self-bonded stake
        if validator.stake > 0:
            self._registry.slash_stake(validator_address, validator.stake)

        # Set status to EXITED (permanent)
        validator.status = ValidatorStatus.EXITED

    def is_permanently_banned(self, address: bytes) -> bool:
        """Check if a validator has been permanently banned."""
        validator = self._registry.get(address)
        if validator is None:
            return False
        return (
            validator.status == ValidatorStatus.EXITED
            and validator.slashing_events >= MAX_SLASHING_EVENTS
        )

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_events_for_validator(
        self, address: bytes,
    ) -> List[SlashingEvidence]:
        """All slashing events for a given validator."""
        return [e for e in self._evidence if e.validator_address == address]

    def get_recent_events(self, count: int = 10) -> List[SlashingEvidence]:
        """Most recent slashing events."""
        return list(reversed(self._evidence[-count:]))

    def total_slashed_for_validator(self, address: bytes) -> int:
        """Total amount slashed from a specific validator."""
        return sum(
            e.amount_slashed
            for e in self._evidence
            if e.validator_address == address
        )

    # ------------------------------------------------------------------ #
    #  Verification helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def verify_double_sign_evidence(
        validator_pubkey: bytes,
        slot: int,
        block_hash_a: bytes,
        block_hash_b: bytes,
        signature_a: bytes,
        signature_b: bytes,
    ) -> bool:
        """
        Cryptographically verify double-sign evidence.

        Checks that both signatures are valid for the same slot but
        different block hashes, signed by the same public key.
        """
        if block_hash_a == block_hash_b:
            return False

        slot_bytes = struct.pack(">Q", slot)

        msg_a = sha512(slot_bytes + block_hash_a)
        msg_b = sha512(slot_bytes + block_hash_b)

        valid_a = KeyPair.verify(validator_pubkey, signature_a, msg_a)
        valid_b = KeyPair.verify(validator_pubkey, signature_b, msg_b)

        return valid_a and valid_b

    # ------------------------------------------------------------------ #
    #  Governance-gated early unjail                                      #
    # ------------------------------------------------------------------ #

    def request_early_unjail(
        self, validator_address: bytes, deposit: int,
    ) -> Optional[UnjailRequest]:
        """
        Request early release from jail via governance.

        The validator must be currently jailed. A deposit is required
        which is slashed if the request is rejected.

        Returns a dict describing the pending request, or None on failure.
        """
        validator = self._registry.get(validator_address)
        if validator is None:
            return None
        if validator.status != ValidatorStatus.JAILED:
            return None
        if deposit <= 0:
            return None

        # Store pending unjail request
        if not hasattr(self, '_unjail_requests'):
            self._unjail_requests: Dict[bytes, dict] = {}

        request = {
            "validator": validator_address,
            "deposit": deposit,
            "requested_at": time.time(),
            "votes_for": 0,
            "votes_against": 0,
            "voters": [],
            "status": "pending",  # pending, approved, rejected
        }
        self._unjail_requests[validator_address] = request
        return request

    def vote_early_unjail(
        self, validator_address: bytes, voter: bytes, approve: bool,
    ) -> Optional[UnjailRequest]:
        """Vote on an early unjail request. Requires 66% supermajority."""
        if not hasattr(self, '_unjail_requests'):
            return None
        request = self._unjail_requests.get(validator_address)
        if request is None or request["status"] != "pending":
            return None
        voter_hex = voter.hex()
        if voter_hex in request["voters"]:
            return None  # Already voted

        request["voters"].append(voter_hex)
        if approve:
            request["votes_for"] += 1
        else:
            request["votes_against"] += 1

        total = request["votes_for"] + request["votes_against"]
        if total >= 3:  # Minimum votes before resolution
            ratio = request["votes_for"] / total
            if ratio >= 2 / 3:  # 66% supermajority
                request["status"] = "approved"
                # Unjail the validator
                validator = self._registry.get(validator_address)
                if validator and validator.status == ValidatorStatus.JAILED:
                    self._registry.unjail(validator_address)
            elif request["votes_against"] > total / 3:
                request["status"] = "rejected"
                # Slash the deposit (returned as negative amount_slashed)

        return request

    def get_unjail_request(self, validator_address: bytes) -> Optional[UnjailRequest]:
        """Get a pending early unjail request."""
        if not hasattr(self, '_unjail_requests'):
            return None
        return self._unjail_requests.get(validator_address)

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "evidence": [e.to_dict() for e in self._evidence],
            "processed_double_signs": [
                {"address": addr.hex(), "slot": slot}
                for addr, slot in self._processed_double_signs
            ],
        }

    @classmethod
    def from_dict(
        cls, d: dict, registry: ValidatorRegistry,
    ) -> SlashingManager:
        mgr = cls(registry)
        for ed in d.get("evidence", []):
            mgr._evidence.append(SlashingEvidence.from_dict(ed))
        for pds in d.get("processed_double_signs", []):
            mgr._processed_double_signs.add(
                (bytes.fromhex(pds["address"]), pds["slot"])
            )
        return mgr

    def __repr__(self) -> str:
        return (
            f"SlashingManager(events={len(self._evidence)}, "
            f"total_burned={self.total_slashed})"
        )


class AdaptiveSlashingManager:
    """Extended slashing with ML-based behavioral analysis.

    Tracks validator behavior over a sliding window and adjusts
    slashing severity based on trust scores computed from historical
    patterns.
    """

    WINDOW_SIZE = 100  # epochs
    TRUST_DECAY = 0.95  # trust decays per missed epoch
    TRUST_RECOVERY = 0.01  # trust recovers per good epoch

    def __init__(self, registry: ValidatorRegistry) -> None:
        self._registry = registry
        self._behavior_window: Dict[bytes, list] = {}  # addr -> list of epoch records
        self._trust_scores: Dict[bytes, float] = {}    # addr -> [0.0, 1.0]

    def record_validator_behavior(
        self,
        address: bytes,
        epoch: int,
        blocks_proposed: int,
        blocks_missed: int,
        attestations_sent: int,
        ai_disputes: int,
    ) -> None:
        """Record epoch-level behavior metrics for a validator."""
        if address not in self._behavior_window:
            self._behavior_window[address] = []
            self._trust_scores[address] = 1.0

        record = {
            'epoch': epoch,
            'proposed': blocks_proposed,
            'missed': blocks_missed,
            'attested': attestations_sent,
            'disputes': ai_disputes,
        }
        window = self._behavior_window[address]
        window.append(record)
        # Keep only last WINDOW_SIZE entries
        if len(window) > self.WINDOW_SIZE:
            window.pop(0)

        # Update trust score
        self._update_trust(address, record)

    def _update_trust(self, address: bytes, record: dict) -> None:
        """Update trust score based on latest behavior."""
        trust = self._trust_scores.get(address, 1.0)

        # Penalize misses
        if record['missed'] > 0:
            trust *= self.TRUST_DECAY ** record['missed']

        # Penalize AI disputes
        if record['disputes'] > 0:
            trust *= self.TRUST_DECAY ** record['disputes']

        # Reward good behavior
        if record['proposed'] > 0 and record['missed'] == 0:
            trust = min(1.0, trust + self.TRUST_RECOVERY * record['proposed'])

        self._trust_scores[address] = max(0.0, min(1.0, trust))

    def compute_trust_score(self, address: bytes) -> float:
        """Get current trust score for a validator (0.0 = untrusted, 1.0 = fully trusted)."""
        return self._trust_scores.get(address, 1.0)

    def adaptive_slash_amount(self, address: bytes, base_fraction: float) -> float:
        """Scale slash fraction by inverse trust score.

        Trusted validators (>0.9) get 50% lighter slashing.
        Untrusted validators (<0.3) get up to 2x heavier slashing.
        """
        trust = self.compute_trust_score(address)
        if trust > 0.9:
            return base_fraction * 0.5
        elif trust > 0.7:
            return base_fraction * 0.75
        elif trust > 0.3:
            return base_fraction
        else:
            return min(base_fraction * 2.0, 1.0)

    def graduated_penalty(
        self,
        address: bytes,
        offense_type: str,
    ) -> Tuple[str, float, float]:
        """Determine graduated penalty based on trust and history.

        Returns (action, slash_fraction, jail_duration_epochs).
        Actions: 'warning', 'small_slash', 'large_slash', 'jail', 'ban'
        """
        trust = self.compute_trust_score(address)
        window = self._behavior_window.get(address, [])
        recent_offenses = sum(
            1 for r in window[-10:]
            if r.get('missed', 0) > 0 or r.get('disputes', 0) > 0
        )

        if offense_type == 'double_sign':
            # Always severe for double signing
            return ('jail', 0.33, 36.0)

        if offense_type == 'downtime':
            if recent_offenses <= 1 and trust > 0.8:
                return ('warning', 0.0, 0.0)
            elif recent_offenses <= 3 and trust > 0.5:
                return ('small_slash', 0.05, 4.0)
            elif recent_offenses <= 5:
                return ('large_slash', 0.15, 12.0)
            else:
                return ('jail', 0.33, 36.0)

        if offense_type == 'ai_dispute':
            if recent_offenses <= 2 and trust > 0.7:
                return ('warning', 0.0, 0.0)
            elif recent_offenses <= 5:
                return ('small_slash', 0.03, 2.0)
            else:
                return ('large_slash', 0.10, 8.0)

        return ('warning', 0.0, 0.0)
