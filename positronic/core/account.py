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
    validator_pubkey: bytes = b""  # Ed25519 public key for block signing (set by STAKE TX)

    # Rewards & Unbonding
    pending_rewards: int = 0     # Accumulated staking/attestation rewards awaiting claim
    unstaking_amount: int = 0    # Amount currently in unbonding period
    unstake_available_at: float = 0.0  # Unix timestamp when unstaking completes

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
            and self.pending_rewards == 0
            and self.unstaking_amount == 0
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
            "validator_pubkey": self.validator_pubkey.hex() if self.validator_pubkey else "",
            "ai_reputation": round(self.ai_reputation, 10),
            "quarantine_count": self.quarantine_count,
            "trust_score": self.trust_score,
            "trust_level": self.trust_level,
            "pending_rewards": self.pending_rewards,
            "unstaking_amount": self.unstaking_amount,
            "unstake_available_at": self.unstake_available_at,
        }
        return d

    @staticmethod
    def _safe_hex(val) -> bytes:
        """Safely convert a hex string or bytes to bytes."""
        if not val:
            return b""
        if isinstance(val, bytes):
            return val
        if isinstance(val, str):
            clean = val.removeprefix("0x").strip()
            try:
                return bytes.fromhex(clean) if clean else b""
            except ValueError:
                return b""
        return b""

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        # Defensive address parsing: handle hex string, bytes, or raw
        raw_addr = d.get("address", b"")
        if isinstance(raw_addr, bytes):
            addr_bytes = raw_addr
        elif isinstance(raw_addr, str):
            clean = raw_addr.removeprefix("0x").strip()
            addr_bytes = bytes.fromhex(clean) if clean else b""
        else:
            addr_bytes = bytes(raw_addr) if raw_addr else b""
        return cls(
            address=addr_bytes,
            nonce=int(d.get("nonce", 0)),
            balance=int(d.get("balance", 0)),
            code_hash=cls._safe_hex(d.get("code_hash")),
            storage_root=cls._safe_hex(d.get("storage_root")),
            staked_amount=int(d.get("staked_amount", 0)),
            delegated_to=cls._safe_hex(d.get("delegated_to")),
            is_validator=d.get("is_validator", False),
            is_nvn=d.get("is_nvn", False),
            validator_pubkey=cls._safe_hex(d.get("validator_pubkey")),
            ai_reputation=d.get("ai_reputation", 1.0),
            quarantine_count=int(d.get("quarantine_count", 0)),
            trust_score=int(d.get("trust_score", 0)),
            trust_level=int(d.get("trust_level", 0)),
            pending_rewards=int(d.get("pending_rewards", 0)),
            unstaking_amount=int(d.get("unstaking_amount", 0)),
            unstake_available_at=float(d.get("unstake_available_at", 0.0)),
        )

    def __repr__(self) -> str:
        from positronic.utils.encoding import format_positronic
        return (
            f"Account(addr={self.address_hex[:10]}..., "
            f"balance={format_positronic(self.balance)}, "
            f"nonce={self.nonce})"
        )
