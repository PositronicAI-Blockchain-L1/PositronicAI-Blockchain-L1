"""
Positronic - Block and BlockHeader Data Structures
Uses SHA-512 for all hashing (64-byte hashes).
Includes AI validation metadata in block header.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from positronic.crypto.hashing import sha512, merkle_root
from positronic.crypto.keys import KeyPair
from positronic.core.transaction import Transaction
from positronic.utils.serialization import to_json_bytes
from positronic.constants import (
    BLOCK_HEADER_VERSION,
    BLOCK_GAS_LIMIT,
    HASH_SIZE,
)


@dataclass
class BlockHeader:
    """
    Positronic block header.
    All hashes are 64 bytes (SHA-512).
    Includes AI model version and AI score root for PoNC.
    """
    version: int = BLOCK_HEADER_VERSION
    height: int = 0
    timestamp: float = 0.0
    previous_hash: bytes = b"\x00" * HASH_SIZE
    state_root: bytes = b"\x00" * HASH_SIZE
    transactions_root: bytes = b"\x00" * HASH_SIZE
    receipts_root: bytes = b"\x00" * HASH_SIZE

    # PoNC-specific fields
    ai_score_root: bytes = b"\x00" * HASH_SIZE  # Merkle root of TX AI scores
    ai_model_version: int = 1                     # Active AI model version
    ai_scores: Dict[str, int] = field(default_factory=dict)  # tx_hash_hex -> quantized score

    # Validator info
    validator_pubkey: bytes = b""    # 32-byte Ed25519 public key of proposer
    validator_signature: bytes = b"" # 64-byte Ed25519 signature

    # Consensus
    slot: int = 0
    epoch: int = 0

    # Gas
    gas_limit: int = BLOCK_GAS_LIMIT
    gas_used: int = 0

    # Extra
    extra_data: bytes = b""  # Max 256 bytes, arbitrary data
    chain_id: int = 420420

    @property
    def signing_data(self) -> bytes:
        """Data to be signed (everything except validator_signature)."""
        obj = {
            "version": self.version,
            "height": self.height,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash.hex(),
            "state_root": self.state_root.hex(),
            "transactions_root": self.transactions_root.hex(),
            "receipts_root": self.receipts_root.hex(),
            "ai_score_root": self.ai_score_root.hex(),
            "ai_model_version": self.ai_model_version,
            "validator_pubkey": self.validator_pubkey.hex(),
            "slot": self.slot,
            "epoch": self.epoch,
            "gas_limit": self.gas_limit,
            "gas_used": self.gas_used,
            "extra_data": self.extra_data.hex(),
            "chain_id": self.chain_id,
        }
        return to_json_bytes(obj)

    @property
    def hash(self) -> bytes:
        """SHA-512 hash of the block header (64 bytes)."""
        return sha512(self.signing_data)

    @property
    def hash_hex(self) -> str:
        """Block hash as hex string."""
        return "0x" + self.hash.hex()

    def sign(self, keypair: KeyPair):
        """Sign the block header."""
        self.validator_pubkey = keypair.public_key_bytes
        self.validator_signature = keypair.sign(self.signing_data)

    def verify_signature(self) -> bool:
        """Verify the block header signature."""
        if not self.validator_signature or not self.validator_pubkey:
            return False
        return KeyPair.verify(
            self.validator_pubkey,
            self.validator_signature,
            self.signing_data,
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "height": self.height,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash.hex(),
            "state_root": self.state_root.hex(),
            "transactions_root": self.transactions_root.hex(),
            "receipts_root": self.receipts_root.hex(),
            "ai_score_root": self.ai_score_root.hex(),
            "ai_model_version": self.ai_model_version,
            "validator_pubkey": self.validator_pubkey.hex(),
            "validator_signature": self.validator_signature.hex(),
            "slot": self.slot,
            "epoch": self.epoch,
            "gas_limit": self.gas_limit,
            "gas_used": self.gas_used,
            "extra_data": self.extra_data.hex(),
            "chain_id": self.chain_id,
            "hash": self.hash.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockHeader":
        return cls(
            version=d.get("version", BLOCK_HEADER_VERSION),
            height=d["height"],
            timestamp=d["timestamp"],
            previous_hash=bytes.fromhex(d["previous_hash"]),
            state_root=bytes.fromhex(d["state_root"]),
            transactions_root=bytes.fromhex(d["transactions_root"]),
            receipts_root=bytes.fromhex(d.get("receipts_root", "00" * HASH_SIZE)),
            ai_score_root=bytes.fromhex(d.get("ai_score_root", "00" * HASH_SIZE)),
            ai_model_version=d.get("ai_model_version", 1),
            ai_scores=d.get("ai_scores", {}),
            validator_pubkey=bytes.fromhex(d.get("validator_pubkey", "")),
            validator_signature=bytes.fromhex(d.get("validator_signature", "")),
            slot=d.get("slot", 0),
            epoch=d.get("epoch", 0),
            gas_limit=d.get("gas_limit", BLOCK_GAS_LIMIT),
            gas_used=d.get("gas_used", 0),
            extra_data=bytes.fromhex(d.get("extra_data", "")),
            chain_id=d.get("chain_id", 420420),
        )


@dataclass
class Block:
    """A complete Positronic block."""

    header: BlockHeader
    transactions: List[Transaction] = field(default_factory=list)
    _cached_hash: Optional[bytes] = field(default=None, repr=False)

    @property
    def hash(self) -> bytes:
        """Block hash. Uses cached value from deserialization if available,
        otherwise recomputes from header. This ensures backward compatibility
        when header fields evolve (e.g., new fields added)."""
        if self._cached_hash is not None:
            return self._cached_hash
        return self.header.hash

    @property
    def hash_hex(self) -> str:
        return self.hash.hex()

    @property
    def height(self) -> int:
        return self.header.height

    @property
    def tx_count(self) -> int:
        return len(self.transactions)

    def compute_transactions_root(self) -> bytes:
        """Compute the Merkle root of transaction hashes."""
        if not self.transactions:
            return b"\x00" * HASH_SIZE
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        return merkle_root(tx_hashes)

    def compute_ai_score_root(self) -> bytes:
        """
        Compute the Merkle root of AI scores for all transactions.
        Uses fixed-point encoding (score * 1,000,000) with 4-byte big-endian
        to ensure deterministic cross-node agreement. The higher multiplier
        preserves 6 decimal places of precision (e.g., 0.850001 != 0.850000).
        """
        if not self.transactions:
            return b"\x00" * HASH_SIZE
        score_hashes = [
            sha512(tx.tx_hash + round(tx.ai_score * 1_000_000).to_bytes(4, "big"))
            for tx in self.transactions
        ]
        return merkle_root(score_hashes)

    def finalize(self, keypair: KeyPair, state_root: bytes = None):
        """Finalize the block: compute roots and sign."""
        self.header.transactions_root = self.compute_transactions_root()
        self.header.ai_score_root = self.compute_ai_score_root()
        if state_root:
            self.header.state_root = state_root
        self.header.gas_used = sum(tx.gas_limit for tx in self.transactions)
        self.header.sign(keypair)

    def size(self) -> int:
        """Approximate block size in bytes."""
        return len(to_json_bytes(self.to_dict()))

    def to_dict(self) -> dict:
        return {
            "hash": self.hash.hex(),
            "header": self.header.to_dict(),
            "transactions": [tx.to_dict() for tx in self.transactions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        header = BlockHeader.from_dict(d["header"])
        transactions = [Transaction.from_dict(tx) for tx in d.get("transactions", [])]
        # Preserve the original hash from the producing node to ensure
        # backward compatibility when header fields evolve.
        cached_hash = None
        if "hash" in d:
            try:
                cached_hash = bytes.fromhex(d["hash"])
            except (ValueError, TypeError):
                pass
        return cls(header=header, transactions=transactions, _cached_hash=cached_hash)

    def __repr__(self) -> str:
        return (
            f"Block(height={self.height}, "
            f"txs={self.tx_count}, "
            f"hash={self.hash_hex[:18]}...)"
        )


@dataclass
class TransactionReceipt:
    """Receipt for an executed transaction."""
    tx_hash: bytes
    block_hash: bytes
    block_height: int
    tx_index: int
    status: bool           # True = success, False = reverted
    gas_used: int
    logs: List[dict] = field(default_factory=list)
    contract_address: Optional[bytes] = None
    return_data: bytes = b""
    ai_score: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        d = {
            "tx_hash": self.tx_hash.hex(),
            "block_hash": self.block_hash.hex(),
            "block_height": self.block_height,
            "tx_index": self.tx_index,
            "status": self.status,
            "gas_used": self.gas_used,
            "logs": self.logs,
            "return_data": self.return_data.hex(),
            "ai_score": self.ai_score,
            "error": self.error,
        }
        if self.contract_address:
            d["contract_address"] = self.contract_address.hex()
        return d
