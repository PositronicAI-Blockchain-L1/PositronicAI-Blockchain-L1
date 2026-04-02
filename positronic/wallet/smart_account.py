"""
Positronic - Smart Wallet (Account Abstraction)
Smart wallets with session keys, spending limits, and social recovery.
No seed phrase needed for recovery.

Audit fixes:
- SessionKey.total_spent now tracks actual spending via record_spend()
- SessionKey.daily_limit enforced in can_spend()
- to_dict() / from_dict() serialization for state persistence
- clean_expired_sessions() callable from block processing
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set
import time


@dataclass
class SessionKey:
    """Temporary key with limited permissions."""
    key: bytes
    permissions: list  # Allowed TxTypes
    expires_at: float
    max_value_per_tx: int = 0  # 0 = unlimited
    daily_limit: int = 0       # 0 = unlimited daily limit
    total_spent: int = 0
    daily_spent: int = 0
    last_daily_reset: float = 0.0
    created_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.is_expired

    def _reset_daily_if_needed(self):
        """Reset daily spending counter if 24h passed."""
        if self.last_daily_reset == 0.0:
            self.last_daily_reset = time.time()
        if time.time() - self.last_daily_reset > 86400:
            self.daily_spent = 0
            self.last_daily_reset = time.time()

    def can_spend(self, amount: int) -> bool:
        """Check if this session key can spend the given amount."""
        if self.is_expired:
            return False
        if self.max_value_per_tx > 0 and amount > self.max_value_per_tx:
            return False
        if self.daily_limit > 0:
            self._reset_daily_if_needed()
            if self.daily_spent + amount > self.daily_limit:
                return False
        return True

    def record_spend(self, amount: int):
        """Record a spend against this session key. Updates total_spent and daily tracking."""
        self.total_spent += amount
        self._reset_daily_if_needed()
        self.daily_spent += amount

    def to_dict(self) -> dict:
        return {
            "key": self.key.hex(),
            "permissions": self.permissions,
            "expires_at": self.expires_at,
            "max_value_per_tx": self.max_value_per_tx,
            "daily_limit": self.daily_limit,
            "total_spent": self.total_spent,
            "daily_spent": self.daily_spent,
            "last_daily_reset": self.last_daily_reset,
            "created_at": self.created_at,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionKey":
        return cls(
            key=bytes.fromhex(d["key"]),
            permissions=d.get("permissions", []),
            expires_at=d.get("expires_at", 0.0),
            max_value_per_tx=d.get("max_value_per_tx", 0),
            daily_limit=d.get("daily_limit", 0),
            total_spent=d.get("total_spent", 0),
            daily_spent=d.get("daily_spent", 0),
            last_daily_reset=d.get("last_daily_reset", 0.0),
            created_at=d.get("created_at", 0.0),
        )


@dataclass
class SmartWallet:
    """Smart wallet with advanced features."""
    address: bytes
    owner_key: bytes  # Primary owner public key
    session_keys: Dict[bytes, SessionKey] = field(default_factory=dict)
    recovery_guardians: Set[bytes] = field(default_factory=set)
    recovery_threshold: int = 2  # Guardians needed for recovery
    daily_spending_limit: int = 0  # 0 = unlimited
    daily_spent: int = 0
    last_spending_reset: float = 0.0
    created_at: float = 0.0
    is_locked: bool = False

    def add_session_key(self, key: bytes, permissions: list,
                       duration: float = 3600, max_value: int = 0,
                       daily_limit: int = 0) -> SessionKey:
        """Add a temporary session key."""
        sk = SessionKey(
            key=key,
            permissions=permissions,
            expires_at=time.time() + duration,
            max_value_per_tx=max_value,
            daily_limit=daily_limit,
            created_at=time.time(),
        )
        self.session_keys[key] = sk
        return sk

    def remove_session_key(self, key: bytes) -> bool:
        """Remove a session key."""
        if key in self.session_keys:
            del self.session_keys[key]
            return True
        return False

    def is_session_key_valid(self, key: bytes) -> bool:
        """Check if a session key is valid."""
        sk = self.session_keys.get(key)
        return sk is not None and sk.is_valid

    def use_session_key(self, key: bytes, amount: int) -> bool:
        """
        Use a session key for a transaction.
        Checks validity and spending limits, then records the spend.
        Returns True if allowed, False otherwise.
        """
        sk = self.session_keys.get(key)
        if sk is None or not sk.can_spend(amount):
            return False
        sk.record_spend(amount)
        return True

    def add_guardian(self, guardian: bytes) -> bool:
        """Add a recovery guardian."""
        if guardian == self.owner_key:
            return False
        self.recovery_guardians.add(guardian)
        return True

    def remove_guardian(self, guardian: bytes) -> bool:
        """Remove a recovery guardian."""
        if guardian in self.recovery_guardians:
            self.recovery_guardians.discard(guardian)
            return True
        return False

    def can_recover(self, guardian_signatures: List[bytes]) -> bool:
        """Check if enough guardians have signed for recovery."""
        valid_sigs = sum(1 for g in guardian_signatures if g in self.recovery_guardians)
        return valid_sigs >= self.recovery_threshold

    def set_spending_limit(self, daily_max: int):
        """Set daily spending limit."""
        self.daily_spending_limit = daily_max
        self.daily_spent = 0
        self.last_spending_reset = time.time()

    def can_spend(self, amount: int) -> bool:
        """Check if spending is within daily limit."""
        if self.is_locked:
            return False
        if self.daily_spending_limit == 0:
            return True
        # Reset if 24h passed
        if time.time() - self.last_spending_reset > 86400:
            self.daily_spent = 0
            self.last_spending_reset = time.time()
        return self.daily_spent + amount <= self.daily_spending_limit

    def record_spending(self, amount: int):
        """Record spending against daily limit."""
        self.daily_spent += amount

    def lock(self):
        """Lock wallet (emergency)."""
        self.is_locked = True

    def unlock(self):
        """Unlock wallet."""
        self.is_locked = False

    def clean_expired_sessions(self):
        """Remove expired session keys."""
        expired = [k for k, v in self.session_keys.items() if v.is_expired]
        for k in expired:
            del self.session_keys[k]

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "owner_key": self.owner_key.hex(),
            "session_keys": {k.hex(): v.to_dict() for k, v in self.session_keys.items()},
            "session_keys_count": len(self.session_keys),
            "active_session_keys": sum(1 for sk in self.session_keys.values() if sk.is_valid),
            "guardian_count": len(self.recovery_guardians),
            "guardians": [g.hex() for g in self.recovery_guardians],
            "recovery_threshold": self.recovery_threshold,
            "daily_spending_limit": self.daily_spending_limit,
            "daily_spent": self.daily_spent,
            "is_locked": self.is_locked,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SmartWallet":
        wallet = cls(
            address=bytes.fromhex(d["address"].removeprefix("0x")),
            owner_key=bytes.fromhex(d["owner_key"]),
            recovery_threshold=d.get("recovery_threshold", 2),
            daily_spending_limit=d.get("daily_spending_limit", 0),
            daily_spent=d.get("daily_spent", 0),
            created_at=d.get("created_at", 0.0),
            is_locked=d.get("is_locked", False),
        )
        # Restore session keys
        for key_hex, sk_data in d.get("session_keys", {}).items():
            sk = SessionKey.from_dict(sk_data)
            wallet.session_keys[sk.key] = sk
        # Restore guardians
        for g_hex in d.get("guardians", []):
            wallet.recovery_guardians.add(bytes.fromhex(g_hex))
        return wallet


class SmartWalletRegistry:
    """Registry of all smart wallets."""

    def __init__(self):
        self._wallets: Dict[bytes, SmartWallet] = {}

    def create(self, address: bytes, owner_key: bytes) -> SmartWallet:
        """Create a new smart wallet."""
        if address in self._wallets:
            return self._wallets[address]
        wallet = SmartWallet(
            address=address,
            owner_key=owner_key,
            created_at=time.time(),
        )
        self._wallets[address] = wallet
        return wallet

    def get(self, address: bytes) -> Optional[SmartWallet]:
        """Get smart wallet by address."""
        return self._wallets.get(address)

    def exists(self, address: bytes) -> bool:
        """Check if smart wallet exists."""
        return address in self._wallets

    def clean_all_expired_sessions(self):
        """Clean expired session keys from all wallets. Call during block processing."""
        for wallet in self._wallets.values():
            wallet.clean_expired_sessions()

    def get_stats(self) -> dict:
        return {
            "total_wallets": len(self._wallets),
            "locked_wallets": sum(1 for w in self._wallets.values() if w.is_locked),
            "wallets_with_guardians": sum(
                1 for w in self._wallets.values() if len(w.recovery_guardians) > 0
            ),
            "total_session_keys": sum(
                len(w.session_keys) for w in self._wallets.values()
            ),
        }
