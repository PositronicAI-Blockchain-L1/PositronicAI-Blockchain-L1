"""
Positronic - Validator Data Model and Registry
Manages the set of active validators in the DPoS consensus.
Top 21 validators by total stake (own + delegated) produce blocks.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512
from positronic.crypto.address import address_from_pubkey
from positronic.constants import (
    MIN_STAKE,
    MAX_VALIDATORS,
    MIN_VALIDATORS,
    FEE_NVN_SHARE,
    BLOCK_PRODUCER_COUNT,
    QUARANTINE_JUDGE_COUNT,
    POOL_CONCENTRATION_LIMIT,
    MIN_VALIDATORS_FOR_POOLS,
)
from positronic.utils.logging import get_logger

logger = get_logger("positronic.consensus.validator")


class ValidatorStatus(Enum):
    """Lifecycle states for a validator."""
    PENDING = auto()      # Registered but not yet elected
    ACTIVE = auto()       # Elected into the active set
    JAILED = auto()       # Slashed and temporarily removed
    EXITING = auto()      # Voluntary exit in progress (unbonding)
    EXITED = auto()       # Fully exited, stake returned


@dataclass
class Validator:
    """
    Represents a single DPoS validator on the Positronic network.

    Attributes:
        address:          20-byte derived address (identity).
        pubkey:           32-byte Ed25519 public key for block signing.
        stake:            Self-bonded stake in base units.
        delegated_stake:  Total stake delegated to this validator by others.
        commission_rate:  Fraction of delegator rewards kept by validator (0.0-1.0).
        status:           Current lifecycle status.
        is_nvn:           True if this validator also runs a Neural Validator Node.
        jailed_until:     Timestamp until which the validator is jailed.
        missed_blocks:    Count of consecutive missed block proposals.
        total_missed:     Lifetime missed block count.
        proposed_blocks:  Lifetime proposed block count.
        registered_at:    Timestamp of initial registration.
        last_proposed_at: Timestamp of most recent block proposal.
        slashing_events:  Number of times this validator has been slashed.
        extra_data:       Arbitrary metadata (e.g. node URL, description).
    """
    address: bytes
    pubkey: bytes
    stake: int = 0
    delegated_stake: int = 0
    commission_rate: float = 0.05   # 5 % default commission
    status: ValidatorStatus = ValidatorStatus.PENDING
    is_nvn: bool = False
    jailed_until: float = 0.0
    missed_blocks: int = 0
    total_missed: int = 0
    proposed_blocks: int = 0
    registered_at: float = 0.0
    last_proposed_at: float = 0.0
    slashing_events: int = 0
    extra_data: bytes = b""

    # ------------------------------------------------------------------ #
    #  Derived properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def total_stake(self) -> int:
        """Total stake = self-bonded + delegated."""
        return self.stake + self.delegated_stake

    @property
    def is_active(self) -> bool:
        return self.status == ValidatorStatus.ACTIVE

    @property
    def is_jailed(self) -> bool:
        return self.status == ValidatorStatus.JAILED

    @property
    def is_eligible(self) -> bool:
        """Can this validator be elected (meets minimum stake, not jailed)?"""
        return (
            self.total_stake >= MIN_STAKE
            and self.status in (ValidatorStatus.PENDING, ValidatorStatus.ACTIVE)
        )

    @property
    def address_hex(self) -> str:
        return "0x" + self.address.hex()

    @property
    def uptime(self) -> float:
        """Uptime ratio: proposed / (proposed + total_missed)."""
        total = self.proposed_blocks + self.total_missed
        if total == 0:
            return 1.0
        return self.proposed_blocks / total

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "pubkey": self.pubkey.hex(),
            "stake": self.stake,
            "delegated_stake": self.delegated_stake,
            "commission_rate": self.commission_rate,
            "status": self.status.name,
            "is_nvn": self.is_nvn,
            "jailed_until": self.jailed_until,
            "missed_blocks": self.missed_blocks,
            "total_missed": self.total_missed,
            "proposed_blocks": self.proposed_blocks,
            "registered_at": self.registered_at,
            "last_proposed_at": self.last_proposed_at,
            "slashing_events": self.slashing_events,
            "extra_data": self.extra_data.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Validator:
        return cls(
            address=bytes.fromhex(d["address"].removeprefix("0x")),
            pubkey=bytes.fromhex(d["pubkey"]),
            stake=d.get("stake", 0),
            delegated_stake=d.get("delegated_stake", 0),
            commission_rate=d.get("commission_rate", 0.05),
            status=ValidatorStatus[d.get("status", "PENDING")],
            is_nvn=d.get("is_nvn", False),
            jailed_until=d.get("jailed_until", 0.0),
            missed_blocks=d.get("missed_blocks", 0),
            total_missed=d.get("total_missed", 0),
            proposed_blocks=d.get("proposed_blocks", 0),
            registered_at=d.get("registered_at", 0.0),
            last_proposed_at=d.get("last_proposed_at", 0.0),
            slashing_events=d.get("slashing_events", 0),
            extra_data=bytes.fromhex(d.get("extra_data", "")),
        )

    def __repr__(self) -> str:
        return (
            f"Validator(addr={self.address_hex[:12]}..., "
            f"stake={self.total_stake}, "
            f"status={self.status.name})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Validator):
            return False
        return self.address == other.address

    def __hash__(self) -> int:
        return hash(self.address)


# ====================================================================== #
#  Validator Registry                                                     #
# ====================================================================== #

class ValidatorRegistry:
    """
    In-memory registry of all known validators.
    Provides lookup, ranking, and active-set management.
    """

    def __init__(self) -> None:
        # address (bytes) -> Validator
        self._validators: Dict[bytes, Validator] = {}
        # pubkey (bytes) -> address (bytes) for fast lookup
        self._pubkey_index: Dict[bytes, bytes] = {}

    # ------------------------------------------------------------------ #
    #  Registration                                                       #
    # ------------------------------------------------------------------ #

    def register(
        self,
        pubkey: bytes,
        stake: int,
        commission_rate: float = 0.05,
        is_nvn: bool = False,
        extra_data: bytes = b"",
    ) -> Validator:
        """
        Register a new validator.  Requires at least MIN_STAKE.
        Returns the created Validator.
        Raises ValueError on duplicate or insufficient stake.
        """
        if stake < MIN_STAKE:
            raise ValueError(
                f"Insufficient stake: {stake} < MIN_STAKE ({MIN_STAKE})"
            )

        # Hard cap: prevent memory exhaustion from unlimited registrations
        from positronic.constants import MAX_VALIDATORS_HARD_CAP
        if len(self._validators) >= MAX_VALIDATORS_HARD_CAP:
            raise ValueError(
                f"Validator hard cap reached: {MAX_VALIDATORS_HARD_CAP}"
            )

        address = address_from_pubkey(pubkey)
        if address in self._validators:
            raise ValueError(
                f"Validator already registered: 0x{address.hex()}"
            )

        if not 0.0 <= commission_rate <= 1.0:
            raise ValueError(
                f"Commission rate must be between 0.0 and 1.0, got {commission_rate}"
            )

        validator = Validator(
            address=address,
            pubkey=pubkey,
            stake=stake,
            commission_rate=commission_rate,
            status=ValidatorStatus.PENDING,
            is_nvn=is_nvn,
            registered_at=time.time(),
            extra_data=extra_data,
        )

        self._validators[address] = validator
        self._pubkey_index[pubkey] = address
        return validator

    # ------------------------------------------------------------------ #
    #  Lookups                                                            #
    # ------------------------------------------------------------------ #

    def get(self, address: bytes) -> Optional[Validator]:
        """Retrieve a validator by address."""
        return self._validators.get(address)

    def get_by_pubkey(self, pubkey: bytes) -> Optional[Validator]:
        """Retrieve a validator by its Ed25519 public key."""
        addr = self._pubkey_index.get(pubkey)
        if addr is None:
            return None
        return self._validators.get(addr)

    def contains(self, address: bytes) -> bool:
        return address in self._validators

    @property
    def all_validators(self) -> List[Validator]:
        """All registered validators regardless of status."""
        return list(self._validators.values())

    @property
    def active_validators(self) -> List[Validator]:
        """Only validators with ACTIVE status."""
        return [v for v in self._validators.values() if v.is_active]

    @property
    def eligible_validators(self) -> List[Validator]:
        """Validators eligible for election (meet stake, not jailed)."""
        return [v for v in self._validators.values() if v.is_eligible]

    @property
    def nvn_validators(self) -> List[Validator]:
        """Active validators that also run Neural Validator Nodes."""
        return [v for v in self.active_validators if v.is_nvn]

    @property
    def active_count(self) -> int:
        return len(self.active_validators)

    @property
    def total_count(self) -> int:
        return len(self._validators)

    # ------------------------------------------------------------------ #
    #  Ranking                                                            #
    # ------------------------------------------------------------------ #

    def ranked_by_stake(self, eligible_only: bool = True) -> List[Validator]:
        """
        Return validators sorted by total_stake descending.
        Ties broken by earliest registration time.
        """
        pool = self.eligible_validators if eligible_only else self.all_validators
        return sorted(
            pool,
            key=lambda v: (-v.total_stake, v.registered_at),
        )

    def top_validators(self, count: int = MAX_VALIDATORS) -> List[Validator]:
        """Return the top-N validators by stake (candidates for the active set)."""
        return self.ranked_by_stake(eligible_only=True)[:count]

    # ------------------------------------------------------------------ #
    #  Status transitions                                                 #
    # ------------------------------------------------------------------ #

    def activate(self, address: bytes) -> None:
        """Move a validator to ACTIVE status."""
        v = self._require(address)
        v.status = ValidatorStatus.ACTIVE

    def deactivate(self, address: bytes) -> None:
        """Move an active validator to PENDING (removed from active set)."""
        v = self._require(address)
        if v.status == ValidatorStatus.ACTIVE:
            v.status = ValidatorStatus.PENDING

    def jail(self, address: bytes, until: float) -> None:
        """Jail a validator until the given timestamp."""
        v = self._require(address)
        v.status = ValidatorStatus.JAILED
        v.jailed_until = until

    def unjail(self, address: bytes) -> bool:
        """
        Unjail a validator if the jail period has elapsed.
        Returns True if successfully unjailed.
        """
        v = self._require(address)
        if v.status != ValidatorStatus.JAILED:
            return False
        if time.time() < v.jailed_until:
            return False
        v.status = ValidatorStatus.PENDING
        v.jailed_until = 0.0
        v.missed_blocks = 0
        return True

    def begin_exit(self, address: bytes) -> None:
        """Start the voluntary exit (unbonding) process."""
        v = self._require(address)
        if v.status not in (ValidatorStatus.ACTIVE, ValidatorStatus.PENDING):
            raise ValueError(
                f"Validator {v.address_hex} cannot exit from status {v.status.name}"
            )
        v.status = ValidatorStatus.EXITING

    def complete_exit(self, address: bytes) -> int:
        """
        Finalise exit: return the validator's own stake.
        Returns the stake amount to be refunded.
        """
        v = self._require(address)
        if v.status != ValidatorStatus.EXITING:
            raise ValueError(
                f"Validator {v.address_hex} is not in EXITING status"
            )
        refund = v.stake
        v.stake = 0
        v.status = ValidatorStatus.EXITED
        return refund

    def remove(self, address: bytes) -> Optional[Validator]:
        """Permanently remove a validator from the registry."""
        v = self._validators.pop(address, None)
        if v is not None:
            self._pubkey_index.pop(v.pubkey, None)
        return v

    # ------------------------------------------------------------------ #
    #  Block tracking                                                     #
    # ------------------------------------------------------------------ #

    def record_proposed_block(self, address: bytes, timestamp: float) -> None:
        """Record that a validator successfully proposed a block."""
        v = self._require(address)
        v.proposed_blocks += 1
        v.missed_blocks = 0  # reset consecutive miss counter
        v.last_proposed_at = timestamp

    def record_missed_block(self, address: bytes) -> None:
        """Record that a validator missed their block proposal slot."""
        v = self._require(address)
        v.missed_blocks += 1
        v.total_missed += 1

    # ------------------------------------------------------------------ #
    #  Stake mutations (used by staking module)                           #
    # ------------------------------------------------------------------ #

    def add_stake(self, address: bytes, amount: int) -> None:
        """Increase a validator's self-bonded stake."""
        v = self._require(address)
        v.stake += amount

    def slash_stake(self, address: bytes, amount: int) -> int:
        """
        Reduce a validator's self-bonded stake by *amount*.
        Returns the actual amount slashed (may be less if stake is small).
        """
        v = self._require(address)
        actual = min(amount, v.stake)
        v.stake -= actual
        v.slashing_events += 1
        return actual

    def add_delegation(self, validator_address: bytes, amount: int) -> None:
        """Increase the delegated stake on a validator."""
        v = self._require(validator_address)
        v.delegated_stake += amount

    def remove_delegation(self, validator_address: bytes, amount: int) -> int:
        """
        Decrease the delegated stake on a validator.
        Returns actual amount removed.
        """
        v = self._require(validator_address)
        actual = min(amount, v.delegated_stake)
        v.delegated_stake -= actual
        return actual

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Serialize the full validator registry to a dict."""
        return {
            "validators": [v.to_dict() for v in self._validators.values()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ValidatorRegistry:
        """Restore a ValidatorRegistry from a serialized dict."""
        registry = cls()
        for vd in d.get("validators", []):
            v = Validator.from_dict(vd)
            registry._validators[v.address] = v
            registry._pubkey_index[v.pubkey] = v.address
        return registry

    # ------------------------------------------------------------------ #
    #  Pool Separation (Phase 32b: Validator Centralization Fix)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deterministic_shuffle(
        validators: List[Validator],
        seed_bytes: bytes,
    ) -> List[Validator]:
        """
        Deterministic Fisher-Yates shuffle using SHA-512-derived randomness.
        Same seed always produces the same ordering.
        """
        result = list(validators)
        n = len(result)
        if n <= 1:
            return result

        # Generate enough random bytes for all swaps
        random_stream = b""
        counter = 0
        while len(random_stream) < n * 8:
            random_stream += hashlib.sha512(
                seed_bytes + counter.to_bytes(4, "big")
            ).digest()
            counter += 1

        for i in range(n - 1, 0, -1):
            offset = (n - 1 - i) * 8
            rand_val = int.from_bytes(random_stream[offset:offset + 8], "big")
            j = rand_val % (i + 1)
            result[i], result[j] = result[j], result[i]

        return result

    @staticmethod
    def _stake_weighted_select(
        validators: List[Validator],
        count: int,
        seed_bytes: bytes,
    ) -> List[Validator]:
        """
        Select *count* validators using stake-weighted random sampling
        without replacement. Deterministic given the same seed.
        """
        if count >= len(validators):
            return list(validators)

        selected: List[Validator] = []
        remaining = list(validators)

        for pick in range(count):
            # Generate fresh 512-bit random value for each pick.
            # We use the full 64-byte SHA-512 output to ensure the random
            # value can span any realistic total_stake (which may exceed 2^64
            # with large BASE_UNIT denominations).
            rand_hash = hashlib.sha512(
                seed_bytes + b"pick" + pick.to_bytes(4, "big")
            ).digest()
            rand_val = int.from_bytes(rand_hash, "big")

            total_stake = sum(v.total_stake for v in remaining)
            if total_stake == 0:
                # Equal stake fallback: uniform random
                idx = rand_val % len(remaining)
                selected.append(remaining.pop(idx))
                continue

            threshold = rand_val % total_stake

            cumulative = 0
            chosen_idx = len(remaining) - 1  # fallback to last
            for idx, v in enumerate(remaining):
                cumulative += v.total_stake
                if cumulative > threshold:
                    chosen_idx = idx
                    break

            selected.append(remaining.pop(chosen_idx))

        return selected

    def elect_epoch_roles(
        self,
        epoch_seed: int,
        validators: List[Validator],
    ) -> Optional[Dict[str, List[Validator]]]:
        """
        Elect validators into disjoint pools for an epoch.

        Args:
            epoch_seed: Deterministic seed for this epoch.
            validators: List of eligible validators to assign.

        Returns:
            dict with keys 'pool_a' (Block Producers), 'pool_b'
            (Quarantine Judges), and 'pool_c' (Governance Voters),
            or None if fewer than MIN_VALIDATORS_FOR_POOLS validators.
        """
        if len(validators) < MIN_VALIDATORS_FOR_POOLS:
            logger.debug(
                "Too few validators for pool separation: %d < %d",
                len(validators), MIN_VALIDATORS_FOR_POOLS,
            )
            return None

        seed_bytes = hashlib.sha512(
            epoch_seed.to_bytes(32, "big", signed=False)
        ).digest()

        # Pool A: top 21 by stake (existing algorithm)
        ranked = sorted(
            validators,
            key=lambda v: (-v.total_stake, v.registered_at),
        )
        pool_a = ranked[:BLOCK_PRODUCER_COUNT]

        # Remaining candidates for Pool B
        pool_a_set = set(v.address for v in pool_a)
        remaining = [v for v in validators if v.address not in pool_a_set]

        # Pool B: 7 quarantine judges, stake-weighted random from remainder
        pool_b_seed = hashlib.sha512(seed_bytes + b"pool_b").digest()
        pool_b = self._stake_weighted_select(
            remaining, QUARANTINE_JUDGE_COUNT, pool_b_seed,
        )

        # Pool C: everyone not in A or B
        pool_b_set = set(v.address for v in pool_b)
        pool_c = [
            v for v in validators
            if v.address not in pool_a_set and v.address not in pool_b_set
        ]

        logger.debug(
            "Epoch roles elected: pool_a=%d, pool_b=%d, pool_c=%d",
            len(pool_a), len(pool_b), len(pool_c),
        )

        return {
            "pool_a": pool_a,
            "pool_b": pool_b,
            "pool_c": pool_c,
        }

    @staticmethod
    def check_concentration(
        pool: List[Validator],
        known_entities: Dict[str, List[bytes]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check whether any single entity exceeds the concentration limit
        within a pool.

        Args:
            pool: List of validators in a single pool.
            known_entities: Mapping of entity_id to list of validator addresses
                            controlled by that entity.

        Returns:
            (ok, offender) — True if no entity exceeds the limit, or
            (False, entity_id) if one does.
        """
        pool_size = len(pool)
        if pool_size == 0:
            return (True, None)

        pool_addresses = set(v.address for v in pool)
        max_allowed = int(pool_size * POOL_CONCENTRATION_LIMIT)
        # Ensure at least 1 slot is allowed per entity
        max_allowed = max(max_allowed, 1)

        for entity_id, addresses in known_entities.items():
            count = sum(1 for a in addresses if a in pool_addresses)
            if count > max_allowed:
                logger.debug(
                    "Concentration limit exceeded: entity=%s has %d/%d in pool (limit=%d)",
                    entity_id, count, pool_size, max_allowed,
                )
                return (False, entity_id)

        return (True, None)

    def elect_epoch_roles_safe(
        self,
        epoch_seed: int,
        validators: List[Validator],
        known_entities: Optional[Dict[str, List[bytes]]] = None,
    ) -> Optional[Dict[str, List[Validator]]]:
        """
        Elect epoch roles with concentration limit enforcement.

        If any entity exceeds the 15% concentration limit in any pool,
        re-runs the election with the offending entity's validators excluded.

        Args:
            epoch_seed: Deterministic seed for this epoch.
            validators: List of eligible validators to assign.
            known_entities: Optional mapping of entity_id -> [validator_addresses].

        Returns:
            dict with 'pool_a', 'pool_b', 'pool_c', or None if too few validators.
        """
        if known_entities is None:
            return self.elect_epoch_roles(epoch_seed, validators)

        candidates = list(validators)
        excluded_entities: set = set()

        for _attempt in range(10):  # bounded retry
            result = self.elect_epoch_roles(epoch_seed, candidates)
            if result is None:
                return None

            violation_found = False
            for pool_name in ("pool_a", "pool_b", "pool_c"):
                ok, offender = self.check_concentration(
                    result[pool_name], known_entities,
                )
                if not ok and offender is not None:
                    logger.debug(
                        "Re-running election: excluding entity=%s", offender,
                    )
                    excluded_entities.add(offender)
                    offender_addrs = set(known_entities[offender])
                    candidates = [
                        v for v in candidates
                        if v.address not in offender_addrs
                    ]
                    violation_found = True
                    break  # restart election

            if not violation_found:
                return result

        # After 10 retries, return best-effort result
        logger.debug("Pool election: max retries reached, returning best-effort")
        return self.elect_epoch_roles(epoch_seed, candidates)

    # ------------------------------------------------------------------ #
    #  Database Persistence                                               #
    # ------------------------------------------------------------------ #

    def save_to_db(self, db) -> None:
        """Persist all validators to the database ``validators`` table.

        *db* must be a :class:`positronic.storage.database.Database` instance
        (or any object with ``execute`` / ``safe_commit`` methods).
        """
        import json as _json
        # Recreate table with TEXT stake columns to avoid SQLite INTEGER overflow
        # (stakes can exceed 2^63 in wei representation).
        try:
            db.execute("DROP TABLE IF EXISTS validators")
            db.execute("""CREATE TABLE validators (
                pubkey TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                stake TEXT DEFAULT '0',
                delegated_stake TEXT DEFAULT '0',
                is_active INTEGER DEFAULT 1,
                activation_epoch INTEGER DEFAULT 0,
                exit_epoch INTEGER DEFAULT -1,
                slashed INTEGER DEFAULT 0,
                attestation_count INTEGER DEFAULT 0,
                validator_json TEXT NOT NULL
            )""")
        except Exception:
            pass
        for v in self._validators.values():
            db.execute(
                """INSERT OR REPLACE INTO validators
                   (pubkey, address, stake, delegated_stake, is_active,
                    activation_epoch, exit_epoch, slashed,
                    attestation_count, validator_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    v.pubkey.hex(),
                    v.address.hex(),
                    str(v.stake),
                    str(v.delegated_stake),
                    1 if v.is_active else 0,
                    0,
                    -1 if v.status != ValidatorStatus.EXITED else 0,
                    v.slashing_events,
                    v.proposed_blocks,
                    _json.dumps(v.to_dict()),
                ),
            )
        db.safe_commit()

    def load_from_db(self, db) -> int:
        """Load validators from the database ``validators`` table.

        Returns the number of validators loaded.
        *db* must be a :class:`positronic.storage.database.Database` instance.
        """
        import json as _json
        rows = db.execute(
            "SELECT validator_json FROM validators"
        ).fetchall()
        loaded = 0
        for row in rows:
            try:
                vd = _json.loads(row["validator_json"] if isinstance(row, dict) or hasattr(row, '__getitem__') else row[0])
                v = Validator.from_dict(vd)
                if v.address not in self._validators:
                    self._validators[v.address] = v
                    self._pubkey_index[v.pubkey] = v.address
                    loaded += 1
            except Exception as e:
                logger.debug("Failed to load validator from DB: %s", e)
        if loaded:
            logger.info("Loaded %d validators from database", loaded)
        return loaded

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _require(self, address: bytes) -> Validator:
        """Retrieve a validator or raise."""
        v = self._validators.get(address)
        if v is None:
            raise KeyError(f"Validator not found: 0x{address.hex()}")
        return v

    def __len__(self) -> int:
        return len(self._validators)

    def __repr__(self) -> str:
        return (
            f"ValidatorRegistry(total={self.total_count}, "
            f"active={self.active_count})"
        )
