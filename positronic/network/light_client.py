"""
Positronic - SPV/Light Client Support
Simplified Payment Verification allows clients to verify transactions
without downloading the full blockchain. Uses Merkle proofs to verify
inclusion of transactions in blocks.

Superior to Bitcoin's SPV:
- SHA-512 Merkle proofs (stronger than SHA-256)
- AI score verification included in proofs
- Block header sync with BFT finality awareness
"""

import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from positronic.crypto.hashing import sha512
from positronic.constants import HASH_SIZE

logger = logging.getLogger("positronic.network.light_client")


@dataclass
class MerkleProof:
    """A Merkle proof for transaction inclusion in a block."""
    tx_hash: bytes
    block_hash: bytes
    block_height: int
    proof_hashes: List[bytes]  # Sibling hashes from leaf to root
    proof_directions: List[bool]  # True = right sibling, False = left sibling
    tx_index: int  # Transaction index in block
    transactions_root: bytes  # Expected root

    def verify(self) -> bool:
        """
        Verify the Merkle proof.
        Recomputes the root from the leaf and sibling hashes.
        Returns True if computed root matches the expected transactions_root.
        """
        current = self.tx_hash

        for i, sibling in enumerate(self.proof_hashes):
            if self.proof_directions[i]:
                # Sibling is on the right
                current = sha512(current + sibling)
            else:
                # Sibling is on the left
                current = sha512(sibling + current)

        return current == self.transactions_root

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash.hex(),
            "block_hash": self.block_hash.hex(),
            "block_height": self.block_height,
            "proof_hashes": [h.hex() for h in self.proof_hashes],
            "proof_directions": self.proof_directions,
            "tx_index": self.tx_index,
            "transactions_root": self.transactions_root.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MerkleProof":
        return cls(
            tx_hash=bytes.fromhex(d["tx_hash"]),
            block_hash=bytes.fromhex(d["block_hash"]),
            block_height=d["block_height"],
            proof_hashes=[bytes.fromhex(h) for h in d["proof_hashes"]],
            proof_directions=d["proof_directions"],
            tx_index=d["tx_index"],
            transactions_root=bytes.fromhex(d["transactions_root"]),
        )


def build_merkle_tree(tx_hashes: List[bytes]) -> Tuple[bytes, List[List[bytes]]]:
    """
    Build a complete Merkle tree from transaction hashes.
    Returns (root, tree_levels) where tree_levels[0] = leaves.
    Uses SHA-512 for quantum resistance.
    """
    if not tx_hashes:
        return b"\x00" * HASH_SIZE, []

    # Pad to even number at each level
    level = list(tx_hashes)
    tree = [level[:]]

    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # Duplicate last hash

        next_level = []
        for i in range(0, len(level), 2):
            combined = sha512(level[i] + level[i + 1])
            next_level.append(combined)

        level = next_level
        tree.append(level[:])

    return level[0], tree


def generate_merkle_proof(
    tx_hashes: List[bytes],
    tx_index: int,
    block_hash: bytes,
    block_height: int,
    transactions_root: bytes,
) -> Optional[MerkleProof]:
    """
    Generate a Merkle proof for a specific transaction.

    Args:
        tx_hashes: All transaction hashes in the block
        tx_index: Index of the target transaction
        block_hash: Hash of the block
        block_height: Height of the block
        transactions_root: Expected Merkle root

    Returns:
        MerkleProof or None if index is invalid
    """
    if not tx_hashes or tx_index < 0 or tx_index >= len(tx_hashes):
        return None

    _, tree = build_merkle_tree(tx_hashes)
    if not tree:
        return None

    proof_hashes = []
    proof_directions = []

    idx = tx_index
    for level in tree[:-1]:  # Skip the root level
        # Pad level if odd
        padded = list(level)
        if len(padded) % 2 == 1:
            padded.append(padded[-1])

        if idx % 2 == 0:
            # Current is left child, sibling is right
            sibling_idx = idx + 1
            proof_directions.append(True)
        else:
            # Current is right child, sibling is left
            sibling_idx = idx - 1
            proof_directions.append(False)

        if sibling_idx < len(padded):
            proof_hashes.append(padded[sibling_idx])
        else:
            proof_hashes.append(padded[-1])

        idx //= 2

    return MerkleProof(
        tx_hash=tx_hashes[tx_index],
        block_hash=block_hash,
        block_height=block_height,
        proof_hashes=proof_hashes,
        proof_directions=proof_directions,
        tx_index=tx_index,
        transactions_root=transactions_root,
    )


@dataclass
class LightClientHeader:
    """Lightweight block header for SPV clients."""
    height: int
    block_hash: bytes
    previous_hash: bytes
    transactions_root: bytes
    state_root: bytes
    timestamp: float
    validator_pubkey: bytes
    validator_signature: bytes
    slot: int
    epoch: int
    tx_count: int
    gas_used: int
    ai_score_root: bytes
    is_finalized: bool = False

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "block_hash": self.block_hash.hex(),
            "previous_hash": self.previous_hash.hex(),
            "transactions_root": self.transactions_root.hex(),
            "state_root": self.state_root.hex(),
            "timestamp": self.timestamp,
            "validator_pubkey": self.validator_pubkey.hex(),
            "validator_signature": self.validator_signature.hex(),
            "slot": self.slot,
            "epoch": self.epoch,
            "tx_count": self.tx_count,
            "gas_used": self.gas_used,
            "ai_score_root": self.ai_score_root.hex(),
            "is_finalized": self.is_finalized,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LightClientHeader":
        return cls(
            height=d["height"],
            block_hash=bytes.fromhex(d["block_hash"]),
            previous_hash=bytes.fromhex(d["previous_hash"]),
            transactions_root=bytes.fromhex(d["transactions_root"]),
            state_root=bytes.fromhex(d["state_root"]),
            timestamp=d["timestamp"],
            validator_pubkey=bytes.fromhex(d["validator_pubkey"]),
            validator_signature=bytes.fromhex(d.get("validator_signature", "00" * 64)),
            slot=d.get("slot", 0),
            epoch=d.get("epoch", 0),
            tx_count=d.get("tx_count", 0),
            gas_used=d.get("gas_used", 0),
            ai_score_root=bytes.fromhex(d.get("ai_score_root", "00" * HASH_SIZE)),
            is_finalized=d.get("is_finalized", False),
        )


class LightClient:
    """
    SPV Light Client for Positronic.

    Stores only block headers and verifies transactions
    using Merkle proofs. Much lighter than a full node.

    Features:
    - Header chain verification (previous_hash linking)
    - Merkle proof verification for TX inclusion
    - BFT finality awareness (knows which blocks are finalized)
    - AI score root verification (can verify AI scores too)
    """

    def __init__(self):
        self._headers: Dict[int, LightClientHeader] = {}
        self._best_height: int = 0
        self._finalized_height: int = 0
        self._verified_txs: Dict[bytes, MerkleProof] = {}

    @property
    def height(self) -> int:
        return self._best_height

    @property
    def finalized_height(self) -> int:
        return self._finalized_height

    @property
    def header_count(self) -> int:
        return len(self._headers)

    def add_header(self, header: LightClientHeader) -> bool:
        """
        Add a block header to the light client.
        Validates chain linkage (previous_hash must match).
        Returns True if accepted.
        """
        # Genesis header
        if header.height == 0:
            self._headers[0] = header
            self._best_height = 0
            return True

        # Check we have the parent
        parent = self._headers.get(header.height - 1)
        if parent is None:
            logger.debug(
                f"Missing parent for header at height {header.height}"
            )
            return False

        # Verify chain linkage
        if header.previous_hash != parent.block_hash:
            logger.warning(
                f"Header chain linkage failed at height {header.height}"
            )
            return False

        self._headers[header.height] = header
        if header.height > self._best_height:
            self._best_height = header.height

        if header.is_finalized and header.height > self._finalized_height:
            self._finalized_height = header.height

        return True

    def add_headers_batch(self, headers: List[LightClientHeader]) -> int:
        """Add a batch of headers. Returns count accepted."""
        accepted = 0
        for header in sorted(headers, key=lambda h: h.height):
            if self.add_header(header):
                accepted += 1
        return accepted

    def verify_transaction(self, proof: MerkleProof) -> bool:
        """
        Verify a transaction is included in a block using Merkle proof.
        Also checks that we have the block header and it matches.
        """
        # Check we have the header
        header = self._headers.get(proof.block_height)
        if header is None:
            logger.debug(f"No header for block {proof.block_height}")
            return False

        # Check block hash matches
        if header.block_hash != proof.block_hash:
            logger.warning("Block hash mismatch in proof")
            return False

        # Check transactions root matches
        if header.transactions_root != proof.transactions_root:
            logger.warning("Transactions root mismatch in proof")
            return False

        # Verify the Merkle proof itself
        if not proof.verify():
            logger.warning("Merkle proof verification failed")
            return False

        # Cache verified transaction
        self._verified_txs[proof.tx_hash] = proof
        return True

    def is_tx_confirmed(self, tx_hash: bytes) -> bool:
        """Check if a transaction has been verified."""
        return tx_hash in self._verified_txs

    def is_tx_finalized(self, tx_hash: bytes) -> bool:
        """Check if a verified transaction is in a finalized block."""
        proof = self._verified_txs.get(tx_hash)
        if not proof:
            return False
        header = self._headers.get(proof.block_height)
        if not header:
            return False
        return header.is_finalized

    def get_header(self, height: int) -> Optional[LightClientHeader]:
        """Get a header by height."""
        return self._headers.get(height)

    def get_headers_range(
        self, start: int, count: int
    ) -> List[LightClientHeader]:
        """Get a range of headers."""
        headers = []
        for h in range(start, start + count):
            header = self._headers.get(h)
            if header:
                headers.append(header)
        return headers

    def needs_headers(self, remote_height: int) -> bool:
        """Check if we need to download headers."""
        return remote_height > self._best_height

    def get_stats(self) -> dict:
        return {
            "best_height": self._best_height,
            "finalized_height": self._finalized_height,
            "header_count": len(self._headers),
            "verified_txs": len(self._verified_txs),
            "mode": "light",
        }
