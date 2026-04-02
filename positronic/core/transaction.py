"""
Positronic - Transaction Data Structure
Supports: Transfer, Contract Create/Call, Stake/Unstake,
Evidence, Token Create/Transfer, NFT Mint/Transfer.
All transactions are fully traceable and auditable.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional
import time

from positronic.crypto.hashing import sha512
from positronic.crypto.keys import KeyPair
from positronic.utils.serialization import to_json_bytes


class TxType(IntEnum):
    """Transaction types supported by Positronic.

    IDs 0–4: User-signed core operations.
    ID  5:   Evidence submission (forensics / compliance).
    IDs 6–8: System-generated (no user signature required).
    IDs 9–12: Native asset operations (PRC-20 / PRC-721).
    """
    TRANSFER = 0          # Simple value transfer
    CONTRACT_CREATE = 1   # Deploy new smart contract
    CONTRACT_CALL = 2     # Call contract method
    STAKE = 3             # Stake ASF to become/support validator
    UNSTAKE = 4           # Unstake ASF
    EVIDENCE = 5          # Submit forensic evidence on-chain
    REWARD = 6            # Block reward (system-generated)
    AI_TREASURY = 7       # AI treasury allocation (system-generated)
    GAME_REWARD = 8       # Play-to-Mine game reward (system-generated)
    TOKEN_CREATE = 9      # Create a new PRC-20 token
    TOKEN_TRANSFER = 10   # Transfer PRC-20 tokens
    NFT_MINT = 11         # Mint a PRC-721 NFT
    NFT_TRANSFER = 12     # Transfer a PRC-721 NFT
    CLAIM_REWARDS = 13    # Claim accumulated staking/attestation rewards
    BRIDGE_LOCK    = 14   # Lock ASF tokens to bridge to external chain
    BRIDGE_MINT    = 15   # Mint wrapped tokens on target chain (system)
    BRIDGE_BURN    = 16   # Burn wrapped tokens to release locked ASF (system)
    BRIDGE_RELEASE = 17   # Release locked ASF after burn confirmed (system)


class TxStatus(IntEnum):
    """Transaction status after AI validation."""
    PENDING = 0       # In mempool, not yet scored
    ACCEPTED = 1      # AI score < 0.85, approved
    QUARANTINED = 2   # AI score 0.85-0.95, held for review
    REJECTED = 3      # AI score > 0.95, rejected
    CONFIRMED = 4     # Included in a finalized block
    FAILED = 5        # Execution failed


@dataclass
class Transaction:
    """A Positronic transaction. All transactions are fully traceable."""

    tx_type: TxType
    nonce: int
    sender: bytes              # 32-byte public key
    recipient: bytes           # 20-byte address (or empty for contract create)
    value: int                 # Amount in base units
    gas_price: int             # Fee per gas unit
    gas_limit: int             # Max gas for this TX
    data: bytes = b""          # Calldata / contract bytecode
    signature: bytes = b""     # Ed25519 signature (64 bytes)
    pq_signature: bytes = b""  # Optional ML-DSA-44 post-quantum signature
    timestamp: float = 0.0
    chain_id: int = 420420

    # AI Validation fields
    ai_score: float = 0.0            # Risk score from AI gate (0-1)
    ai_model_version: int = 0        # Which AI model scored this
    status: TxStatus = TxStatus.PENDING

    # Cached transaction hash (not part of dataclass fields)
    _cached_tx_hash: bytes = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        self._cached_tx_hash = None  # Ensure cache is clear on creation

    @property
    def signing_data(self) -> bytes:
        """Data to be signed (everything except signature and AI fields).
        Uses string/int representation for all fields to ensure JS/Python compatibility.
        """
        obj = {
            "chain_id": int(self.chain_id),
            "data": self.data.hex(),
            "gas_limit": int(self.gas_limit),
            "gas_price": int(self.gas_price),
            "nonce": int(self.nonce),
            "recipient": self.recipient.hex(),
            "sender": self.sender.hex(),
            "timestamp": int(self.timestamp),
            "type": int(self.tx_type),
            "value": str(self.value),
        }
        return to_json_bytes(obj)

    @property
    def tx_hash(self) -> bytes:
        """SHA-512 hash of the transaction (64 bytes). Cached after first computation."""
        if self._cached_tx_hash is None:
            self._cached_tx_hash = sha512(self.signing_data)
        return self._cached_tx_hash

    @property
    def tx_hash_hex(self) -> str:
        """Transaction hash as hex string."""
        return "0x" + self.tx_hash.hex()

    def sign(self, keypair: KeyPair) -> "Transaction":
        """Sign the transaction with a key pair."""
        self.sender = keypair.public_key_bytes
        self.signature = keypair.sign(self.signing_data)
        return self

    def dual_sign(self, keypair: KeyPair, pq_keypair) -> "Transaction":
        """Sign with both Ed25519 and ML-DSA-44 for post-quantum security."""
        self.sign(keypair)  # Ed25519 (existing)
        try:
            from positronic.crypto.post_quantum import PQ_AVAILABLE
            if PQ_AVAILABLE and pq_keypair is not None:
                self.pq_signature = pq_keypair.sign(self.signing_data)
        except Exception:
            pass  # PQ not available — Ed25519 only
        return self

    def verify_signature(self) -> bool:
        """Verify the transaction signature."""
        if self.tx_type in (TxType.REWARD, TxType.AI_TREASURY, TxType.GAME_REWARD):
            return True  # System transactions don't need signatures
        # System TXs (STAKE, UNSTAKE, CLAIM_REWARDS) created internally have
        # empty signature and zero gas. Only allow if tx_type is a known system type.
        if self.signature == b"" and self.gas_price == 0 and self.gas_limit == 0:
            if self.tx_type in (TxType.STAKE, TxType.UNSTAKE, TxType.CLAIM_REWARDS,
                                TxType.BRIDGE_MINT, TxType.BRIDGE_BURN, TxType.BRIDGE_RELEASE):
                return True
            return False  # Reject unsigned non-system TXs
        if not self.signature:
            return False
        if not KeyPair.verify(self.sender, self.signature, self.signing_data):
            return False
        # If PQ signature present, verify it too (both must pass)
        if self.pq_signature:
            try:
                from positronic.crypto.post_quantum import PQVerifier, PQ_AVAILABLE
                if PQ_AVAILABLE:
                    # Need sender's PQ public key — stored in pq_signature metadata
                    # For now: verify via PostQuantumManager lookup
                    pass  # Will be wired when PQ key registry is integrated
            except Exception:
                pass  # PQ verification non-fatal if library unavailable
        return True  # Ed25519 passed

    @property
    def sender_address(self) -> bytes:
        """Derive the sender's 20-byte address from their public key."""
        from positronic.crypto.address import address_from_pubkey
        return address_from_pubkey(self.sender)

    @property
    def intrinsic_gas(self) -> int:
        """Calculate the minimum gas required for this transaction."""
        from positronic.constants import TX_BASE_GAS, CREATE_GAS
        gas = TX_BASE_GAS
        if self.tx_type == TxType.CONTRACT_CREATE:
            gas = CREATE_GAS
        # Add gas for non-zero data bytes
        for byte in self.data:
            gas += 16 if byte != 0 else 4
        return gas

    @property
    def total_cost(self) -> int:
        """Total cost = value + gas_limit * gas_price."""
        return self.value + (self.gas_limit * self.gas_price)

    def to_bytes(self) -> bytes:
        """Serialize transaction to bytes (JSON-encoded dict).
        Used by eth_sendRawTransaction for broadcasting signed transactions."""
        return to_json_bytes(self.to_dict())

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        d = {
            "tx_type": int(self.tx_type),
            "nonce": self.nonce,
            "sender": self.sender.hex(),
            "recipient": self.recipient.hex(),
            "value": self.value,
            "gas_price": self.gas_price,
            "gas_limit": self.gas_limit,
            "data": self.data.hex(),
            "signature": self.signature.hex(),
            "pq_signature": self.pq_signature.hex() if self.pq_signature else "",
            "timestamp": self.timestamp,
            "chain_id": self.chain_id,
            "ai_score": self.ai_score,
            "ai_model_version": self.ai_model_version,
            "status": int(self.status),
            "tx_hash": self.tx_hash.hex(),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Transaction":
        """Deserialize from dictionary."""
        _s = str(d.get("sender", "")).removeprefix("0x")
        _r = str(d.get("recipient", "")).removeprefix("0x")
        _sig = str(d.get("signature", "")).removeprefix("0x")
        _data = str(d.get("data", "")).removeprefix("0x")
        tx = cls(
            tx_type=TxType(d["tx_type"]),
            nonce=d["nonce"],
            sender=bytes.fromhex(_s) if _s else b"",
            recipient=bytes.fromhex(_r) if _r else b"",
            value=int(d["value"]),
            gas_price=int(d.get("gas_price", 0)),
            gas_limit=int(d.get("gas_limit", 21000)),
            data=bytes.fromhex(_data) if _data else b"",
            signature=bytes.fromhex(_sig) if _sig else b"",
            pq_signature=bytes.fromhex(d.get("pq_signature", "") or ""),
            timestamp=d.get("timestamp", 0.0),
            chain_id=d.get("chain_id", 420420),
            ai_score=d.get("ai_score", 0.0),
            ai_model_version=d.get("ai_model_version", 0),
            status=TxStatus(d.get("status", 0)),
        )
        return tx

    def __repr__(self) -> str:
        return (
            f"Transaction(type={self.tx_type.name}, "
            f"hash={self.tx_hash_hex[:18]}..., "
            f"value={self.value}, ai_score={self.ai_score:.3f})"
        )
