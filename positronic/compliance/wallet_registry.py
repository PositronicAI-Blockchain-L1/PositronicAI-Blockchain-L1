"""
Positronic - Wallet Registration & Traceability System
AI-verified wallet registration for compliance.
Not traditional KYC - wallets are verified by AI as coming from known/registered sources.
"""

import time
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class WalletStatus(IntEnum):
    """Wallet registration status."""
    UNREGISTERED = 0      # Unknown wallet - restricted
    PENDING = 1           # Registration submitted, awaiting AI verification
    REGISTERED = 2        # AI-verified, full access
    FLAGGED = 3           # Suspicious activity detected
    SUSPENDED = 4         # Temporarily suspended
    BLACKLISTED = 5       # Permanently banned


class WalletTier(IntEnum):
    """Wallet trust tier based on AI analysis."""
    BASIC = 1             # New wallet, limited tx volume
    VERIFIED = 2          # Established history
    TRUSTED = 3           # Long-term good behavior
    INSTITUTIONAL = 4     # Exchange/institutional wallet


@dataclass
class WalletRegistration:
    """Registration record for a wallet."""
    address: bytes
    status: WalletStatus = WalletStatus.UNREGISTERED
    tier: WalletTier = WalletTier.BASIC
    registered_at: float = 0.0
    last_activity: float = 0.0
    total_transactions: int = 0
    total_volume: int = 0           # Total value transferred
    flagged_count: int = 0          # Number of times flagged by AI
    source_type: str = ""           # "wallet_app", "exchange", "contract", etc.
    ai_trust_score: float = 0.5     # AI-computed trust score (0-1)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "status": int(self.status),
            "tier": int(self.tier),
            "registered_at": self.registered_at,
            "last_activity": self.last_activity,
            "total_transactions": self.total_transactions,
            "total_volume": self.total_volume,
            "flagged_count": self.flagged_count,
            "source_type": self.source_type,
            "ai_trust_score": self.ai_trust_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WalletRegistration":
        return cls(
            address=bytes.fromhex(d["address"].removeprefix("0x")),
            status=WalletStatus(d.get("status", 0)),
            tier=WalletTier(d.get("tier", 1)),
            registered_at=d.get("registered_at", 0.0),
            last_activity=d.get("last_activity", 0.0),
            total_transactions=d.get("total_transactions", 0),
            total_volume=d.get("total_volume", 0),
            flagged_count=d.get("flagged_count", 0),
            source_type=d.get("source_type", ""),
            ai_trust_score=d.get("ai_trust_score", 0.5),
            metadata=d.get("metadata", {}),
        )


class WalletRegistry:
    """
    Manages wallet registration and AI-based trust verification.
    Ensures all wallets are traceable and from known sources.
    """

    # Transaction limits by tier (in base units)
    TIER_LIMITS = {
        WalletTier.BASIC: 100 * 10**18,          # 100 ASF per tx
        WalletTier.VERIFIED: 10_000 * 10**18,     # 10K ASF per tx
        WalletTier.TRUSTED: 1_000_000 * 10**18,   # 1M ASF per tx
        WalletTier.INSTITUTIONAL: 0,               # No limit
    }

    def __init__(self):
        self._registry: Dict[bytes, WalletRegistration] = {}
        self._blacklist: Set[bytes] = set()
        self._known_exchanges: Set[bytes] = set()

    def register_wallet(self, address: bytes, source_type: str = "wallet_app") -> WalletRegistration:
        """Register a new wallet."""
        if address in self._blacklist:
            raise ValueError("Address is blacklisted")

        reg = WalletRegistration(
            address=address,
            status=WalletStatus.PENDING,
            tier=WalletTier.BASIC,
            registered_at=time.time(),
            source_type=source_type,
        )
        self._registry[address] = reg
        return reg

    def verify_wallet(self, address: bytes, ai_trust_score: float) -> WalletRegistration:
        """AI verifies a wallet and updates its status."""
        reg = self._registry.get(address)
        if not reg:
            raise ValueError("Wallet not registered")

        reg.ai_trust_score = ai_trust_score
        if ai_trust_score >= 0.7:
            reg.status = WalletStatus.REGISTERED
        elif ai_trust_score >= 0.3:
            reg.status = WalletStatus.PENDING
        else:
            reg.status = WalletStatus.FLAGGED
            reg.flagged_count += 1

        return reg

    def is_registered(self, address: bytes) -> bool:
        """Check if wallet is registered and active."""
        reg = self._registry.get(address)
        if not reg:
            return False
        return reg.status == WalletStatus.REGISTERED

    def get_wallet(self, address: bytes) -> Optional[WalletRegistration]:
        """Get wallet registration info."""
        return self._registry.get(address)

    def check_transaction_allowed(self, sender: bytes, recipient: bytes, value: int) -> tuple:
        """Check if a transaction is allowed based on wallet registration.
        Returns (allowed: bool, reason: str)"""
        sender_reg = self._registry.get(sender)

        # Genesis and system addresses are always allowed
        if sender == b"\x00" * 20:
            return True, "System address"

        if sender in self._blacklist:
            return False, "Sender is blacklisted"

        if recipient in self._blacklist:
            return False, "Recipient is blacklisted"

        # Unregistered wallets get limited access
        if not sender_reg or sender_reg.status == WalletStatus.UNREGISTERED:
            return False, "Sender wallet not registered"

        if sender_reg.status == WalletStatus.SUSPENDED:
            return False, "Sender wallet is suspended"

        if sender_reg.status == WalletStatus.BLACKLISTED:
            return False, "Sender wallet is blacklisted"

        if sender_reg.status == WalletStatus.FLAGGED:
            return False, "Sender wallet is flagged for review"

        # Check tier-based limits
        limit = self.TIER_LIMITS.get(sender_reg.tier, 0)
        if limit > 0 and value > limit:
            return False, f"Transaction value exceeds tier limit ({sender_reg.tier.name})"

        return True, "Allowed"

    def ensure_registered(self, address: bytes, source_type: str = "on_chain") -> WalletRegistration:
        """Register a wallet if not already in the registry.

        Unlike ``register_wallet`` this is idempotent — it returns the
        existing record when the address is already known, and silently
        skips blacklisted addresses instead of raising.
        """
        if address in self._blacklist:
            reg = self._registry.get(address)
            if reg:
                return reg
            # Blacklisted but not in registry — return a minimal record
            return WalletRegistration(address=address, status=WalletStatus.BLACKLISTED)
        reg = self._registry.get(address)
        if reg:
            return reg
        return self.register_wallet(address, source_type=source_type)

    def update_activity(self, address: bytes, tx_value: int):
        """Update wallet activity after a transaction."""
        reg = self._registry.get(address)
        if not reg:
            # Auto-register wallets that appear in on-chain transactions
            try:
                reg = self.ensure_registered(address, source_type="on_chain")
            except ValueError:
                return  # blacklisted — skip
        if reg:
            reg.last_activity = time.time()
            reg.total_transactions += 1
            reg.total_volume += tx_value

            # Auto-upgrade tier based on activity
            if reg.total_transactions >= 1000 and reg.tier < WalletTier.TRUSTED:
                reg.tier = WalletTier.TRUSTED
            elif reg.total_transactions >= 100 and reg.tier < WalletTier.VERIFIED:
                reg.tier = WalletTier.VERIFIED

    def flag_wallet(self, address: bytes, reason: str = ""):
        """Flag a wallet for suspicious activity."""
        reg = self._registry.get(address)
        if reg:
            reg.status = WalletStatus.FLAGGED
            reg.flagged_count += 1
            reg.metadata["flag_reason"] = reason

    def suspend_wallet(self, address: bytes, reason: str = ""):
        """Suspend a wallet."""
        reg = self._registry.get(address)
        if reg:
            reg.status = WalletStatus.SUSPENDED
            reg.metadata["suspend_reason"] = reason

    def blacklist_wallet(self, address: bytes, reason: str = ""):
        """Permanently blacklist a wallet."""
        self._blacklist.add(address)
        reg = self._registry.get(address)
        if reg:
            reg.status = WalletStatus.BLACKLISTED
            reg.metadata["blacklist_reason"] = reason

    def add_known_exchange(self, address: bytes):
        """Register a known exchange wallet."""
        self._known_exchanges.add(address)
        reg = self.register_wallet(address, source_type="exchange")
        reg.status = WalletStatus.REGISTERED
        reg.tier = WalletTier.INSTITUTIONAL
        reg.ai_trust_score = 1.0

    @property
    def total_registered(self) -> int:
        return sum(1 for r in self._registry.values()
                   if r.status == WalletStatus.REGISTERED)

    @property
    def total_wallets(self) -> int:
        return len(self._registry)

    def get_stats(self) -> dict:
        return {
            "total_wallets": self.total_wallets,
            "registered": self.total_registered,
            "blacklisted": len(self._blacklist),
            "known_exchanges": len(self._known_exchanges),
        }
