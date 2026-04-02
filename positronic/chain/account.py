"""
Positronic - Account State Model
Account-based state model (like Ethereum, not UTXO).
All accounts are traceable and auditable.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Account:
    """
    Represents an account in the Positronic world state.
    Can be an externally owned account (EOA) or a contract account.
    All accounts are fully traceable - no anonymous accounts.
    """
    address: bytes              # 20-byte address
    nonce: int = 0              # Transaction count / contract creation count
    balance: int = 0            # Balance in base units (1 ASF = 10^18)
    code_hash: bytes = b""     # SHA-512 hash of contract code (empty for EOA)
    storage_root: bytes = b""  # Root hash of contract storage trie

    # Staking
    staked_amount: int = 0      # Amount staked for DPoS
    delegated_to: bytes = b""   # Validator address this account delegates to
    is_validator: bool = False  # Whether this account is an active validator
    is_nvn: bool = False         # Whether this account runs a Neural Validator Node

    # AI-related
    ai_reputation: float = 1.0   # Reputation score (affected by AI validation history)
    quarantine_count: int = 0    # Number of times transactions were quarantined

    # TRUST (Soulbound Token)
    trust_score: int = 0         # Non-transferable trust score
    trust_level: int = 0         # TrustLevel enum value (0-4)

    @property
    def is_contract(self) -> bool:
        """Check if this is a contract account."""
        return len(self.code_hash) > 0 and self.code_hash != b"\x00" * 64

    @property
    def is_empty(self) -> bool:
        """Check if account is empty (can be pruned)."""
        return (
            self.nonce == 0
            and self.balance == 0
            and not self.code_hash
            and self.staked_amount == 0
        )

    @property
    def effective_balance(self) -> int:
        """Available balance (excluding staked amount)."""
        return self.balance - self.staked_amount

    @property
    def address_hex(self) -> str:
        return "0x" + self.address.hex()

    def to_dict(self) -> dict:
        d = {
            "address": self.address.hex(),
            "nonce": self.nonce,
            "balance": self.balance,
            "code_hash": self.code_hash.hex() if self.code_hash else "",
            "storage_root": self.storage_root.hex() if self.storage_root else "",
            "staked_amount": self.staked_amount,
            "delegated_to": self.delegated_to.hex() if self.delegated_to else "",
            "is_validator": self.is_validator,
            "is_nvn": self.is_nvn,
            "ai_reputation": round(self.ai_reputation, 10),
            "quarantine_count": self.quarantine_count,
            "trust_score": self.trust_score,
            "trust_level": self.trust_level,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        return cls(
            address=bytes.fromhex(d["address"].removeprefix("0x")),
            nonce=d.get("nonce", 0),
            balance=d.get("balance", 0),
            code_hash=bytes.fromhex(d["code_hash"]) if d.get("code_hash") else b"",
            storage_root=bytes.fromhex(d["storage_root"]) if d.get("storage_root") else b"",
            staked_amount=d.get("staked_amount", 0),
            delegated_to=bytes.fromhex(d["delegated_to"]) if d.get("delegated_to") else b"",
            is_validator=d.get("is_validator", False),
            is_nvn=d.get("is_nvn", False),
            ai_reputation=d.get("ai_reputation", 1.0),
            quarantine_count=d.get("quarantine_count", 0),
            trust_score=d.get("trust_score", 0),
            trust_level=d.get("trust_level", 0),
        )

    def __repr__(self) -> str:
        from positronic.utils.encoding import format_positronic
        return (
            f"Account(addr={self.address_hex[:10]}..., "
            f"balance={format_positronic(self.balance)}, "
            f"nonce={self.nonce})"
        )
