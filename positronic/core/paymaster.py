"""
Positronic - Gasless Transactions (Paymaster System)
Sponsors pay gas fees so users can transact for free.
Anti-spam: minimum TRUST score required for gasless transactions.

Audit fix: MIN_TRUST_FOR_GASLESS is now imported from constants.py
and enforced in sponsor_transaction(). Previously the constant was
defined but never checked, making the anti-spam measure dead code.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import time

from positronic.constants import MIN_TRUST_FOR_GASLESS


@dataclass
class PaymasterConfig:
    """Configuration for a paymaster sponsor."""
    sponsor: bytes  # Sponsor address
    balance: int = 0  # ASF balance for paying gas
    max_gas_per_tx: int = 100000  # Max gas per sponsored TX
    daily_limit: int = 1000000  # Max gas per day
    daily_used: int = 0
    last_reset: float = 0.0
    allowed_recipients: list = field(default_factory=list)  # Empty = allow all
    active: bool = True
    created_at: float = 0.0
    total_sponsored: int = 0  # Total gas sponsored

    def can_sponsor(self, gas_needed: int) -> bool:
        """Check if this paymaster can sponsor the given gas amount."""
        if not self.active:
            return False
        if self.balance < gas_needed:
            return False
        if gas_needed > self.max_gas_per_tx:
            return False
        # Reset daily limit if 24h passed
        if time.time() - self.last_reset > 86400:
            self.daily_used = 0
            self.last_reset = time.time()
        if self.daily_used + gas_needed > self.daily_limit:
            return False
        return True

    def spend(self, gas_amount: int) -> bool:
        """Deduct gas from paymaster balance."""
        if not self.can_sponsor(gas_amount):
            return False
        self.balance -= gas_amount
        self.daily_used += gas_amount
        self.total_sponsored += gas_amount
        return True

    def deposit(self, amount: int):
        """Add funds to paymaster."""
        self.balance += amount

    def to_dict(self) -> dict:
        return {
            "sponsor": self.sponsor.hex(),
            "balance": self.balance,
            "max_gas_per_tx": self.max_gas_per_tx,
            "daily_limit": self.daily_limit,
            "daily_used": self.daily_used,
            "active": self.active,
            "total_sponsored": self.total_sponsored,
            "created_at": self.created_at,
        }


class PaymasterRegistry:
    """Registry of all paymasters on the network."""

    def __init__(self):
        self._paymasters: Dict[bytes, PaymasterConfig] = {}
        self._total_gas_sponsored: int = 0

    def register(self, sponsor: bytes, initial_balance: int = 0,
                 max_gas_per_tx: int = 100000, daily_limit: int = 1000000) -> PaymasterConfig:
        """Register a new paymaster."""
        if sponsor in self._paymasters:
            return self._paymasters[sponsor]
        pm = PaymasterConfig(
            sponsor=sponsor,
            balance=initial_balance,
            max_gas_per_tx=max_gas_per_tx,
            daily_limit=daily_limit,
            created_at=time.time(),
            last_reset=time.time(),
        )
        self._paymasters[sponsor] = pm
        return pm

    def get(self, sponsor: bytes) -> Optional[PaymasterConfig]:
        """Get paymaster by sponsor address."""
        return self._paymasters.get(sponsor)

    def find_sponsor(self, gas_needed: int, recipient: bytes = b"",
                     sender_trust_score: int = 0) -> Optional[PaymasterConfig]:
        """
        Find a paymaster that can sponsor the given gas.
        Enforces MIN_TRUST_FOR_GASLESS: sender must have sufficient TRUST.
        """
        if sender_trust_score < MIN_TRUST_FOR_GASLESS:
            return None  # Sender TRUST too low for gasless transactions

        for pm in self._paymasters.values():
            if pm.can_sponsor(gas_needed):
                if not pm.allowed_recipients or recipient in pm.allowed_recipients:
                    return pm
        return None

    def sponsor_transaction(self, sponsor: bytes, gas_amount: int,
                            sender_trust_score: int = 0) -> bool:
        """
        Use a paymaster to sponsor gas.
        Enforces MIN_TRUST_FOR_GASLESS: rejects if sender TRUST too low.
        """
        if sender_trust_score < MIN_TRUST_FOR_GASLESS:
            return False  # Anti-spam: TRUST too low

        pm = self.get(sponsor)
        if pm is None:
            return False
        if pm.spend(gas_amount):
            self._total_gas_sponsored += gas_amount
            return True
        return False

    def deposit(self, sponsor: bytes, amount: int) -> bool:
        """Deposit funds to a paymaster."""
        pm = self.get(sponsor)
        if pm is None:
            return False
        pm.deposit(amount)
        return True

    def deactivate(self, sponsor: bytes) -> bool:
        """Deactivate a paymaster."""
        pm = self.get(sponsor)
        if pm is None:
            return False
        pm.active = False
        return True

    def get_stats(self) -> dict:
        return {
            "total_paymasters": len(self._paymasters),
            "active_paymasters": sum(1 for pm in self._paymasters.values() if pm.active),
            "total_gas_sponsored": self._total_gas_sponsored,
            "min_trust_required": MIN_TRUST_FOR_GASLESS,
        }
