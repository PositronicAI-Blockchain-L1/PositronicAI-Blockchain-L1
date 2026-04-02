"""
Positronic - Staking and Delegation Management
Handles validator self-bonding, delegator stakes, undelegation with
unbonding period, and reward distribution accounting.
Supports optional SQLite persistence for delegation records.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.crypto.hashing import sha512
from positronic.crypto.address import address_from_pubkey
from positronic.consensus.validator import (
    Validator,
    ValidatorRegistry,
    ValidatorStatus,
)
from positronic.constants import (
    MIN_STAKE,
    MAX_VALIDATORS,
    EPOCH_DURATION,
    BASE_UNIT,
    FEE_VALIDATOR_SHARE,
    FEE_NVN_SHARE,
)

# Unbonding period: 7 epochs (~672 seconds in dev, would be days in prod)
UNBONDING_EPOCHS = 7
UNBONDING_DURATION = UNBONDING_EPOCHS * EPOCH_DURATION

# Minimum delegation amount: 10 ASF
MIN_DELEGATION = 10 * BASE_UNIT

# Maximum commission rate a validator can charge
MAX_COMMISSION_RATE = 0.50  # 50 %


@dataclass
class Delegation:
    """
    Represents a delegator's stake with a specific validator.

    Attributes:
        delegator:          20-byte address of the delegator.
        validator_address:  20-byte address of the validator.
        amount:             Amount delegated in base units.
        created_at:         Timestamp of initial delegation.
        last_reward_epoch:  Last epoch for which rewards were claimed.
    """
    delegator: bytes
    validator_address: bytes
    amount: int = 0
    created_at: float = 0.0
    last_reward_epoch: int = 0

    @property
    def delegator_hex(self) -> str:
        return "0x" + self.delegator.hex()

    @property
    def validator_hex(self) -> str:
        return "0x" + self.validator_address.hex()

    def to_dict(self) -> dict:
        return {
            "delegator": self.delegator.hex(),
            "validator_address": self.validator_address.hex(),
            "amount": self.amount,
            "created_at": self.created_at,
            "last_reward_epoch": self.last_reward_epoch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Delegation:
        return cls(
            delegator=bytes.fromhex(d["delegator"]),
            validator_address=bytes.fromhex(d["validator_address"]),
            amount=d.get("amount", 0),
            created_at=d.get("created_at", 0.0),
            last_reward_epoch=d.get("last_reward_epoch", 0),
        )

    def __repr__(self) -> str:
        return (
            f"Delegation(from={self.delegator_hex[:12]}..., "
            f"to={self.validator_hex[:12]}..., "
            f"amount={self.amount})"
        )


@dataclass
class UnbondingEntry:
    """
    A pending undelegation waiting out the unbonding period.

    Attributes:
        delegator:          Address of the delegator.
        validator_address:  Validator the stake was removed from.
        amount:             Amount being unbonded.
        completion_time:    Unix timestamp when the stake can be released.
        created_at:         When the unbonding was initiated.
    """
    delegator: bytes
    validator_address: bytes
    amount: int
    completion_time: float
    created_at: float = 0.0

    @property
    def is_mature(self) -> bool:
        """True if the unbonding period has elapsed."""
        return time.time() >= self.completion_time

    def is_mature_at(self, ts: float) -> bool:
        """True if the unbonding period has elapsed at the given timestamp."""
        return ts >= self.completion_time

    def to_dict(self) -> dict:
        return {
            "delegator": self.delegator.hex(),
            "validator_address": self.validator_address.hex(),
            "amount": self.amount,
            "completion_time": self.completion_time,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UnbondingEntry:
        return cls(
            delegator=bytes.fromhex(d["delegator"]),
            validator_address=bytes.fromhex(d["validator_address"]),
            amount=d.get("amount", 0),
            completion_time=d.get("completion_time", 0.0),
            created_at=d.get("created_at", 0.0),
        )


class StakingManager:
    """
    Manages all staking and delegation operations.

    Coordinates between the ValidatorRegistry (validator-level stake
    bookkeeping) and per-delegator tracking.
    """

    def __init__(
        self,
        registry: ValidatorRegistry,
        db_path: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._logger = logging.getLogger("positronic.consensus.staking")

        # (delegator_address, validator_address) -> Delegation
        self._delegations: Dict[Tuple[bytes, bytes], Delegation] = {}

        # Pending unbondings indexed by delegator
        self._unbonding: Dict[bytes, List[UnbondingEntry]] = {}

        # Accumulated reward pool per validator (validator_address -> amount)
        self._reward_pools: Dict[bytes, int] = {}

        # Optional SQLite persistence
        self._db: Optional[sqlite3.Connection] = None
        self._db_path = db_path
        if db_path:
            self._init_db(db_path)

    @property
    def registry(self) -> ValidatorRegistry:
        return self._registry

    # ------------------------------------------------------------------ #
    #  Validator self-stake                                                #
    # ------------------------------------------------------------------ #

    def create_validator(
        self,
        pubkey: bytes,
        stake: int,
        commission_rate: float = 0.05,
        is_nvn: bool = False,
        extra_data: bytes = b"",
    ) -> Validator:
        """
        Register a new validator with an initial self-bonded stake.
        Raises ValueError if stake < MIN_STAKE or commission too high.
        """
        if commission_rate > MAX_COMMISSION_RATE:
            raise ValueError(
                f"Commission rate {commission_rate} exceeds max {MAX_COMMISSION_RATE}"
            )
        return self._registry.register(
            pubkey=pubkey,
            stake=stake,
            commission_rate=commission_rate,
            is_nvn=is_nvn,
            extra_data=extra_data,
        )

    def add_validator_stake(self, address: bytes, amount: int) -> None:
        """Increase a validator's self-bonded stake."""
        if amount <= 0:
            raise ValueError("Stake amount must be positive")
        self._registry.add_stake(address, amount)

    # ------------------------------------------------------------------ #
    #  Delegation                                                         #
    # ------------------------------------------------------------------ #

    def delegate(
        self,
        delegator: bytes,
        validator_address: bytes,
        amount: int,
    ) -> Delegation:
        """
        Delegate *amount* to a validator.
        Creates a new Delegation or adds to an existing one.
        Raises ValueError if amount < MIN_DELEGATION or validator not found.
        """
        if amount < MIN_DELEGATION:
            raise ValueError(
                f"Delegation amount {amount} below minimum {MIN_DELEGATION}"
            )

        validator = self._registry.get(validator_address)
        if validator is None:
            raise ValueError(
                f"Validator not found: 0x{validator_address.hex()}"
            )

        if validator.status in (ValidatorStatus.EXITING, ValidatorStatus.EXITED):
            raise ValueError(
                f"Cannot delegate to validator in {validator.status.name} status"
            )

        key = (delegator, validator_address)
        delegation = self._delegations.get(key)

        if delegation is None:
            delegation = Delegation(
                delegator=delegator,
                validator_address=validator_address,
                amount=amount,
                created_at=time.time(),
            )
            self._delegations[key] = delegation
        else:
            delegation.amount += amount

        # Update the validator's delegated_stake counter
        self._registry.add_delegation(validator_address, amount)
        return delegation

    def undelegate(
        self,
        delegator: bytes,
        validator_address: bytes,
        amount: int,
        current_time: Optional[float] = None,
    ) -> UnbondingEntry:
        """
        Begin undelegation.  The funds enter an unbonding period before
        becoming available.
        """
        if current_time is None:
            current_time = time.time()

        key = (delegator, validator_address)
        delegation = self._delegations.get(key)

        if delegation is None:
            raise ValueError("No active delegation found")

        if amount <= 0 or amount > delegation.amount:
            raise ValueError(
                f"Invalid undelegate amount: {amount} "
                f"(delegated: {delegation.amount})"
            )

        # Decrease delegation record
        delegation.amount -= amount
        if delegation.amount == 0:
            del self._delegations[key]

        # Decrease validator's aggregated delegation
        self._registry.remove_delegation(validator_address, amount)

        # Create unbonding entry
        entry = UnbondingEntry(
            delegator=delegator,
            validator_address=validator_address,
            amount=amount,
            completion_time=current_time + UNBONDING_DURATION,
            created_at=current_time,
        )
        self._unbonding.setdefault(delegator, []).append(entry)
        return entry

    def complete_unbonding(
        self,
        delegator: bytes,
        current_time: Optional[float] = None,
    ) -> int:
        """
        Release all mature unbonding entries for *delegator*.
        Returns the total amount released.
        """
        if current_time is None:
            current_time = time.time()

        entries = self._unbonding.get(delegator, [])
        if not entries:
            return 0

        released = 0
        remaining: List[UnbondingEntry] = []
        for entry in entries:
            if entry.is_mature_at(current_time):
                released += entry.amount
            else:
                remaining.append(entry)

        if remaining:
            self._unbonding[delegator] = remaining
        else:
            self._unbonding.pop(delegator, None)

        return released

    # ------------------------------------------------------------------ #
    #  Queries                                                            #
    # ------------------------------------------------------------------ #

    def get_delegation(
        self, delegator: bytes, validator_address: bytes,
    ) -> Optional[Delegation]:
        return self._delegations.get((delegator, validator_address))

    def get_delegations_by_delegator(self, delegator: bytes) -> List[Delegation]:
        """All active delegations placed by *delegator*."""
        return [
            d for (dk, _), d in self._delegations.items()
            if dk == delegator
        ]

    def get_delegations_to_validator(
        self, validator_address: bytes,
    ) -> List[Delegation]:
        """All active delegations targeting *validator_address*."""
        return [
            d for (_, vk), d in self._delegations.items()
            if vk == validator_address
        ]

    def get_pending_unbondings(self, delegator: bytes) -> List[UnbondingEntry]:
        return list(self._unbonding.get(delegator, []))

    def total_delegated_to(self, validator_address: bytes) -> int:
        """Sum of all active delegations to a validator."""
        return sum(
            d.amount
            for d in self.get_delegations_to_validator(validator_address)
        )

    # ------------------------------------------------------------------ #
    #  Reward distribution                                                #
    # ------------------------------------------------------------------ #

    def deposit_reward(self, validator_address: bytes, amount: int) -> None:
        """
        Add *amount* to a validator's reward pool.
        Called by the block reward module after each block.
        """
        if amount <= 0:
            return
        self._reward_pools[validator_address] = (
            self._reward_pools.get(validator_address, 0) + amount
        )

    def distribute_rewards(
        self, validator_address: bytes,
    ) -> Dict[bytes, int]:
        """
        Distribute accumulated rewards for *validator_address* to the
        validator and its delegators, respecting the commission rate.

        Returns a mapping {address: amount} of payouts.
        """
        pool = self._reward_pools.pop(validator_address, 0)
        if pool <= 0:
            return {}

        validator = self._registry.get(validator_address)
        if validator is None:
            return {}

        delegations = self.get_delegations_to_validator(validator_address)
        total_stake = validator.total_stake
        if total_stake == 0:
            return {}

        payouts: Dict[bytes, int] = {}

        # Commission goes entirely to the validator
        commission = int(pool * validator.commission_rate)
        validator_share = commission

        # Remaining pool is split pro-rata by stake
        remaining = pool - commission

        # Validator's own share of the remaining (pro-rata)
        if total_stake > 0:
            validator_share += int(remaining * validator.stake / total_stake)
        payouts[validator_address] = validator_share

        # Delegator shares
        for d in delegations:
            share = int(remaining * d.amount / total_stake)
            if share > 0:
                payouts[d.delegator] = payouts.get(d.delegator, 0) + share

        return payouts

    def get_reward_pool(self, validator_address: bytes) -> int:
        """Current undistributed reward pool for a validator."""
        return self._reward_pools.get(validator_address, 0)

    # ------------------------------------------------------------------ #
    #  Validator exit                                                     #
    # ------------------------------------------------------------------ #

    def begin_validator_exit(self, address: bytes) -> None:
        """
        Initiate a validator exit.  Their stake will enter unbonding.
        Delegators must separately undelegate.
        """
        self._registry.begin_exit(address)

    def complete_validator_exit(
        self,
        address: bytes,
        current_time: Optional[float] = None,
    ) -> int:
        """
        Complete a validator exit and return their self-bonded stake.
        """
        return self._registry.complete_exit(address)

    # ------------------------------------------------------------------ #
    #  SQLite Persistence                                                 #
    # ------------------------------------------------------------------ #

    def _init_db(self, db_path: str):
        """Initialize SQLite for delegation persistence."""
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS delegations (
                delegator BLOB NOT NULL,
                validator BLOB NOT NULL,
                amount INTEGER NOT NULL,
                created_at REAL DEFAULT 0,
                last_reward_epoch INTEGER DEFAULT 0,
                PRIMARY KEY (delegator, validator)
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS unbonding_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                delegator BLOB NOT NULL,
                validator BLOB NOT NULL,
                amount INTEGER NOT NULL,
                completion_time REAL NOT NULL,
                created_at REAL DEFAULT 0
            )
        """)
        self._db.commit()
        self._logger.info(f"Delegation DB initialized: {db_path}")

    def save_to_db(self):
        """Persist all delegations and unbonding entries to SQLite."""
        if self._db is None:
            return
        # Clear and rewrite (simple, correct for moderate data sizes)
        self._db.execute("DELETE FROM delegations")
        self._db.execute("DELETE FROM unbonding_entries")
        for (dk, vk), d in self._delegations.items():
            self._db.execute(
                "INSERT INTO delegations VALUES (?, ?, ?, ?, ?)",
                (dk, vk, str(d.amount), d.created_at, d.last_reward_epoch),
            )
        for delegator, entries in self._unbonding.items():
            for e in entries:
                self._db.execute(
                    "INSERT INTO unbonding_entries "
                    "(delegator, validator, amount, completion_time, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (e.delegator, e.validator_address, str(e.amount),
                     e.completion_time, e.created_at),
                )
        self._db.commit()

    def load_from_db(self):
        """Load delegations and unbonding entries from SQLite."""
        if self._db is None:
            return
        rows = self._db.execute("SELECT * FROM delegations").fetchall()
        for dk, vk, amount_str, created_at, last_epoch in rows:
            d = Delegation(
                delegator=dk, validator_address=vk,
                amount=int(amount_str), created_at=created_at,
                last_reward_epoch=last_epoch,
            )
            self._delegations[(dk, vk)] = d
        ub_rows = self._db.execute("SELECT * FROM unbonding_entries").fetchall()
        for _id, dk, vk, amount_str, comp_time, created_at in ub_rows:
            e = UnbondingEntry(
                delegator=dk, validator_address=vk,
                amount=int(amount_str), completion_time=comp_time,
                created_at=created_at,
            )
            self._unbonding.setdefault(dk, []).append(e)
        self._logger.info(
            f"Loaded {len(rows)} delegations, "
            f"{len(ub_rows)} unbonding entries from DB"
        )

    def close_db(self):
        """Save and close the delegation database."""
        if self._db:
            self.save_to_db()
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "delegations": [d.to_dict() for d in self._delegations.values()],
            "unbonding": {
                addr.hex(): [e.to_dict() for e in entries]
                for addr, entries in self._unbonding.items()
            },
            "reward_pools": {
                addr.hex(): amount
                for addr, amount in self._reward_pools.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict, registry: ValidatorRegistry) -> StakingManager:
        mgr = cls(registry)

        for dd in d.get("delegations", []):
            delegation = Delegation.from_dict(dd)
            key = (delegation.delegator, delegation.validator_address)
            mgr._delegations[key] = delegation

        for addr_hex, entries in d.get("unbonding", {}).items():
            addr = bytes.fromhex(addr_hex)
            mgr._unbonding[addr] = [
                UnbondingEntry.from_dict(e) for e in entries
            ]

        for addr_hex, amount in d.get("reward_pools", {}).items():
            mgr._reward_pools[bytes.fromhex(addr_hex)] = amount

        return mgr

    def __repr__(self) -> str:
        return (
            f"StakingManager(delegations={len(self._delegations)}, "
            f"unbonding={sum(len(v) for v in self._unbonding.values())})"
        )
