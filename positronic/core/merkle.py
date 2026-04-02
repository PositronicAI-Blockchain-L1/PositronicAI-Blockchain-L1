"""
Positronic - Merkle Tree Implementation
SHA-512 based Merkle tree with proof generation and verification.
"""

from typing import List, Optional, Tuple
from positronic.crypto.hashing import sha512
from positronic.constants import HASH_SIZE


class MerkleTree:
    """
    Binary Merkle tree using SHA-512.
    Used for transaction roots, state roots, and AI score roots.
    """

    def __init__(self, leaves: List[bytes] = None):
        self.leaves: List[bytes] = []
        self.layers: List[List[bytes]] = []
        if leaves:
            self.build(leaves)

    def build(self, leaves: List[bytes]):
        """Build the Merkle tree from a list of leaf hashes."""
        if not leaves:
            self.leaves = []
            self.layers = [[b"\x00" * 64]]
            return

        # Hash raw data if leaves are not already 64 bytes
        self.leaves = [
            leaf if len(leaf) == 64 else sha512(leaf)
            for leaf in leaves
        ]

        self.layers = [self.leaves[:]]
        current = self.leaves[:]

        while len(current) > 1:
            if len(current) % 2 != 0:
                current.append(current[-1])  # Duplicate last if odd

            next_layer = []
            for i in range(0, len(current), 2):
                combined = sha512(current[i] + current[i + 1])
                next_layer.append(combined)

            self.layers.append(next_layer)
            current = next_layer

    @property
    def root(self) -> bytes:
        """Get the Merkle root (64 bytes)."""
        if not self.layers:
            return b"\x00" * 64
        return self.layers[-1][0]

    @property
    def root_hex(self) -> str:
        return "0x" + self.root.hex()

    def get_proof(self, index: int) -> List[Tuple[bytes, bool]]:
        """
        Get Merkle proof for a leaf at the given index.
        Returns list of (sibling_hash, is_left) tuples.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range")

        proof = []
        idx = index

        for layer in self.layers[:-1]:
            # Determine sibling
            if idx % 2 == 0:
                # Sibling is on the right
                sibling_idx = idx + 1
                if sibling_idx < len(layer):
                    proof.append((layer[sibling_idx], False))
                else:
                    proof.append((layer[idx], False))  # Duplicate
            else:
                # Sibling is on the left
                proof.append((layer[idx - 1], True))

            idx //= 2

        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: bytes,
        proof: List[Tuple[bytes, bool]],
        root: bytes,
    ) -> bool:
        """
        Verify a Merkle proof.
        proof: list of (sibling_hash, is_left) tuples.
        """
        current = leaf_hash

        for sibling, is_left in proof:
            if is_left:
                current = sha512(sibling + current)
            else:
                current = sha512(current + sibling)

        return current == root

    def __len__(self) -> int:
        return len(self.leaves)

    def __repr__(self) -> str:
        return f"MerkleTree(leaves={len(self.leaves)}, root={self.root_hex[:18]}...)"
