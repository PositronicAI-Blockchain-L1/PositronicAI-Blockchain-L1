"""
Positronic - Merkle Patricia Trie (MPT)
SHA-512 based Merkle Patricia Trie for O(1) state root lookups
and cryptographic state proofs.

This is a NEW data structure for world-state roots. The existing
binary MerkleTree (positronic/core/merkle.py) is for TX roots
and remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from positronic.crypto.hashing import sha512


# ── Nibble helpers ────────────────────────────────────────────────

def bytes_to_nibbles(data: bytes) -> List[int]:
    """Convert bytes to a list of 4-bit nibbles (0-15)."""
    nibbles: List[int] = []
    for byte in data:
        nibbles.append((byte >> 4) & 0x0F)
        nibbles.append(byte & 0x0F)
    return nibbles


def nibbles_to_bytes(nibbles: List[int]) -> bytes:
    """Convert a list of nibbles back to bytes.
    If odd number of nibbles, the last nibble is padded into the high bits."""
    result = bytearray()
    for i in range(0, len(nibbles) - 1, 2):
        result.append((nibbles[i] << 4) | nibbles[i + 1])
    if len(nibbles) % 2 != 0:
        result.append(nibbles[-1] << 4)
    return bytes(result)


def common_prefix_length(a: List[int], b: List[int]) -> int:
    """Return the length of the common prefix between two nibble lists."""
    length = min(len(a), len(b))
    for i in range(length):
        if a[i] != b[i]:
            return i
    return length


# ── Node types ────────────────────────────────────────────────────

class TrieNode:
    """Base class for trie nodes."""
    pass


@dataclass
class LeafNode(TrieNode):
    """Stores a key suffix (nibbles) and a value."""
    key_nibbles: List[int]
    value: bytes


@dataclass
class ExtensionNode(TrieNode):
    """Stores a shared key prefix (nibbles) and points to a child."""
    key_nibbles: List[int]
    child: TrieNode


@dataclass
class BranchNode(TrieNode):
    """16-way branch (one slot per nibble) plus an optional value."""
    children: List[Optional[TrieNode]] = field(
        default_factory=lambda: [None] * 16
    )
    value: Optional[bytes] = None


# ── Encoding (for hashing) ───────────────────────────────────────

def _encode_node(node: Optional[TrieNode]) -> bytes:
    """Deterministic serialization of a trie node for hashing."""
    if node is None:
        return b"\x00"

    if isinstance(node, LeafNode):
        # Tag 0x01 | nibble-count (2 bytes) | nibbles-as-bytes | value-len (4 bytes) | value
        nibble_bytes = nibbles_to_bytes(node.key_nibbles)
        return (
            b"\x01"
            + len(node.key_nibbles).to_bytes(2, "big")
            + nibble_bytes
            + len(node.value).to_bytes(4, "big")
            + node.value
        )

    if isinstance(node, ExtensionNode):
        child_hash = _hash_node(node.child)
        nibble_bytes = nibbles_to_bytes(node.key_nibbles)
        return (
            b"\x02"
            + len(node.key_nibbles).to_bytes(2, "big")
            + nibble_bytes
            + child_hash
        )

    if isinstance(node, BranchNode):
        parts = [b"\x03"]
        for child in node.children:
            if child is None:
                parts.append(b"\x00" * 64)
            else:
                parts.append(_hash_node(child))
        if node.value is not None:
            parts.append(b"\x01" + len(node.value).to_bytes(4, "big") + node.value)
        else:
            parts.append(b"\x00")
        return b"".join(parts)

    raise TypeError(f"Unknown node type: {type(node)}")  # pragma: no cover


def _hash_node(node: Optional[TrieNode]) -> bytes:
    """SHA-512 hash of the encoded node (64 bytes)."""
    return sha512(_encode_node(node))


# ── StateProof ────────────────────────────────────────────────────

@dataclass
class StateProof:
    """Cryptographic proof that a key maps to a value in a given root."""
    key: bytes
    value: Optional[bytes]
    nodes: List[bytes]  # list of encoded nodes along the path

    def to_dict(self) -> dict:
        return {
            "key": self.key.hex(),
            "value": self.value.hex() if self.value is not None else None,
            "nodes": [n.hex() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateProof":
        return cls(
            key=bytes.fromhex(d["key"]),
            value=bytes.fromhex(d["value"]) if d["value"] is not None else None,
            nodes=[bytes.fromhex(n) for n in d["nodes"]],
        )


# ── MerklePatriciaTrie ───────────────────────────────────────────

class MerklePatriciaTrie:
    """
    SHA-512 Merkle Patricia Trie.

    Provides:
      - O(log n) get / put / delete
      - O(1) root_hash (cached, recomputed only when dirty)
      - Cryptographic state proofs (get_proof / verify_proof)
    """

    EMPTY_ROOT: bytes = b"\x00" * 64

    def __init__(self, store=None):
        self._root: Optional[TrieNode] = None
        self._dirty: bool = False
        self._cached_root: bytes = self.EMPTY_ROOT
        self._store = store  # Optional[TrieNodeStore]

    # ── root_hash property ──────────────────────────────────────

    @property
    def root_hash(self) -> bytes:
        """O(1) cached root hash. Recomputes only when trie has been mutated."""
        if self._dirty:
            if self._root is None:
                self._cached_root = self.EMPTY_ROOT
            else:
                self._cached_root = _hash_node(self._root)
            self._dirty = False
        return self._cached_root

    # ── persistence ──────────────────────────────────────────────

    def _persist_node(self, node: Optional[TrieNode]):
        """Recursively persist all nodes to store if available."""
        if self._store is None or node is None:
            return
        encoded = _encode_node(node)
        node_hash = sha512(encoded)
        self._store.put(node_hash, encoded)
        # Recurse into children
        if isinstance(node, ExtensionNode):
            self._persist_node(node.child)
        elif isinstance(node, BranchNode):
            for child in node.children:
                if child is not None:
                    self._persist_node(child)

    def flush(self, height: Optional[int] = None):
        """Persist all dirty nodes to the store."""
        if self._store is None:
            return
        self._persist_node(self._root)
        self._store.commit(height=height, root_hash=self.root_hash)

    # ── get ──────────────────────────────────────────────────────

    def get(self, key: bytes) -> Optional[bytes]:
        """Look up a key. Returns the value or None."""
        nibbles = bytes_to_nibbles(key)
        return self._get(self._root, nibbles)

    def _get(self, node: Optional[TrieNode], nibbles: List[int]) -> Optional[bytes]:
        if node is None:
            return None

        if isinstance(node, LeafNode):
            if node.key_nibbles == nibbles:
                return node.value
            return None

        if isinstance(node, ExtensionNode):
            prefix_len = common_prefix_length(node.key_nibbles, nibbles)
            if prefix_len < len(node.key_nibbles):
                return None
            return self._get(node.child, nibbles[prefix_len:])

        if isinstance(node, BranchNode):
            if len(nibbles) == 0:
                return node.value
            idx = nibbles[0]
            return self._get(node.children[idx], nibbles[1:])

        return None  # pragma: no cover

    # ── put ──────────────────────────────────────────────────────

    def put(self, key: bytes, value: bytes):
        """Insert or update a key-value pair."""
        nibbles = bytes_to_nibbles(key)
        self._root = self._put(self._root, nibbles, value)
        self._dirty = True

    def _put(
        self, node: Optional[TrieNode], nibbles: List[int], value: bytes
    ) -> TrieNode:
        if node is None:
            return LeafNode(key_nibbles=list(nibbles), value=value)

        if isinstance(node, LeafNode):
            return self._put_into_leaf(node, nibbles, value)

        if isinstance(node, ExtensionNode):
            return self._put_into_extension(node, nibbles, value)

        if isinstance(node, BranchNode):
            return self._put_into_branch(node, nibbles, value)

        raise TypeError(f"Unknown node type: {type(node)}")  # pragma: no cover

    def _put_into_leaf(
        self, leaf: LeafNode, nibbles: List[int], value: bytes
    ) -> TrieNode:
        cp = common_prefix_length(leaf.key_nibbles, nibbles)

        # Exact match -> update value
        if cp == len(leaf.key_nibbles) and cp == len(nibbles):
            return LeafNode(key_nibbles=list(nibbles), value=value)

        # Create a branch at the divergence point
        branch = BranchNode()

        # Remaining nibbles of existing leaf after common prefix
        old_remaining = leaf.key_nibbles[cp:]
        new_remaining = nibbles[cp:]

        if len(old_remaining) == 0:
            branch.value = leaf.value
        else:
            old_idx = old_remaining[0]
            if len(old_remaining) == 1:
                branch.children[old_idx] = LeafNode(
                    key_nibbles=[], value=leaf.value
                )
            else:
                branch.children[old_idx] = LeafNode(
                    key_nibbles=old_remaining[1:], value=leaf.value
                )

        if len(new_remaining) == 0:
            branch.value = value
        else:
            new_idx = new_remaining[0]
            if len(new_remaining) == 1:
                branch.children[new_idx] = LeafNode(key_nibbles=[], value=value)
            else:
                branch.children[new_idx] = LeafNode(
                    key_nibbles=new_remaining[1:], value=value
                )

        # Wrap in extension if there is a shared prefix
        if cp > 0:
            return ExtensionNode(key_nibbles=nibbles[:cp], child=branch)
        return branch

    def _put_into_extension(
        self, ext: ExtensionNode, nibbles: List[int], value: bytes
    ) -> TrieNode:
        cp = common_prefix_length(ext.key_nibbles, nibbles)

        if cp == len(ext.key_nibbles):
            # Full match on extension prefix -> recurse into child
            new_child = self._put(ext.child, nibbles[cp:], value)
            return ExtensionNode(key_nibbles=ext.key_nibbles, child=new_child)

        # Partial match -> split extension
        branch = BranchNode()

        # Remaining of existing extension after common prefix
        ext_remaining = ext.key_nibbles[cp:]
        ext_idx = ext_remaining[0]
        if len(ext_remaining) == 1:
            branch.children[ext_idx] = ext.child
        else:
            branch.children[ext_idx] = ExtensionNode(
                key_nibbles=ext_remaining[1:], child=ext.child
            )

        # Insert new value
        new_remaining = nibbles[cp:]
        if len(new_remaining) == 0:
            branch.value = value
        else:
            new_idx = new_remaining[0]
            if len(new_remaining) == 1:
                branch.children[new_idx] = LeafNode(key_nibbles=[], value=value)
            else:
                branch.children[new_idx] = LeafNode(
                    key_nibbles=new_remaining[1:], value=value
                )

        if cp > 0:
            return ExtensionNode(key_nibbles=nibbles[:cp], child=branch)
        return branch

    def _put_into_branch(
        self, branch: BranchNode, nibbles: List[int], value: bytes
    ) -> TrieNode:
        if len(nibbles) == 0:
            branch.value = value
            return branch

        idx = nibbles[0]
        branch.children[idx] = self._put(branch.children[idx], nibbles[1:], value)
        return branch

    # ── delete ───────────────────────────────────────────────────

    def delete(self, key: bytes):
        """Delete a key from the trie (no-op if missing)."""
        nibbles = bytes_to_nibbles(key)
        self._root = self._delete(self._root, nibbles)
        self._dirty = True

    def _delete(
        self, node: Optional[TrieNode], nibbles: List[int]
    ) -> Optional[TrieNode]:
        if node is None:
            return None

        if isinstance(node, LeafNode):
            if node.key_nibbles == nibbles:
                return None
            return node

        if isinstance(node, ExtensionNode):
            cp = common_prefix_length(node.key_nibbles, nibbles)
            if cp < len(node.key_nibbles):
                return node  # key not in this subtree
            new_child = self._delete(node.child, nibbles[cp:])
            if new_child is None:
                return None
            return self._compact_extension(
                ExtensionNode(key_nibbles=node.key_nibbles, child=new_child)
            )

        if isinstance(node, BranchNode):
            return self._delete_from_branch(node, nibbles)

        return node  # pragma: no cover

    def _delete_from_branch(
        self, branch: BranchNode, nibbles: List[int]
    ) -> Optional[TrieNode]:
        if len(nibbles) == 0:
            branch.value = None
        else:
            idx = nibbles[0]
            branch.children[idx] = self._delete(branch.children[idx], nibbles[1:])

        return self._compact_branch(branch)

    def _compact_branch(self, branch: BranchNode) -> Optional[TrieNode]:
        """After a delete, collapse a branch if it has only one remaining item."""
        non_empty = []
        for i, child in enumerate(branch.children):
            if child is not None:
                non_empty.append(i)

        if branch.value is not None:
            # Branch still holds a value
            if len(non_empty) == 0:
                # Only the value remains -> leaf
                return LeafNode(key_nibbles=[], value=branch.value)
            # Multiple children or value + children -> keep branch
            return branch

        if len(non_empty) == 0:
            return None

        if len(non_empty) == 1:
            # Only one child -> collapse
            idx = non_empty[0]
            child = branch.children[idx]
            return self._merge_into_single_child(idx, child)

        return branch

    def _merge_into_single_child(self, idx: int, child: TrieNode) -> TrieNode:
        """Merge a single-child branch into its child, prepending the index nibble."""
        if isinstance(child, LeafNode):
            return LeafNode(key_nibbles=[idx] + child.key_nibbles, value=child.value)
        if isinstance(child, ExtensionNode):
            return ExtensionNode(
                key_nibbles=[idx] + child.key_nibbles, child=child.child
            )
        # Child is a branch -> wrap in extension
        return ExtensionNode(key_nibbles=[idx], child=child)

    def _compact_extension(self, ext: ExtensionNode) -> TrieNode:
        """If an extension's child is also an extension or leaf, merge them."""
        child = ext.child
        if isinstance(child, ExtensionNode):
            return ExtensionNode(
                key_nibbles=ext.key_nibbles + child.key_nibbles, child=child.child
            )
        if isinstance(child, LeafNode):
            return LeafNode(
                key_nibbles=ext.key_nibbles + child.key_nibbles, value=child.value
            )
        return ext

    # ── proof generation ─────────────────────────────────────────

    def get_proof(self, key: bytes) -> Optional[StateProof]:
        """
        Generate a Merkle proof for an existing key.
        Returns None if the key does not exist in the trie.
        """
        if self._root is None:
            return None

        nibbles = bytes_to_nibbles(key)
        path_nodes: List[bytes] = []
        value = self._collect_proof(self._root, nibbles, path_nodes)

        if value is None:
            return None

        return StateProof(key=key, value=value, nodes=path_nodes)

    def _collect_proof(
        self,
        node: Optional[TrieNode],
        nibbles: List[int],
        path_nodes: List[bytes],
    ) -> Optional[bytes]:
        """Walk the trie, collecting encoded nodes along the path."""
        if node is None:
            return None

        path_nodes.append(_encode_node(node))

        if isinstance(node, LeafNode):
            if node.key_nibbles == nibbles:
                return node.value
            return None

        if isinstance(node, ExtensionNode):
            cp = common_prefix_length(node.key_nibbles, nibbles)
            if cp < len(node.key_nibbles):
                return None
            return self._collect_proof(node.child, nibbles[cp:], path_nodes)

        if isinstance(node, BranchNode):
            if len(nibbles) == 0:
                return node.value
            idx = nibbles[0]
            return self._collect_proof(
                node.children[idx], nibbles[1:], path_nodes
            )

        return None  # pragma: no cover

    # ── proof verification (static) ──────────────────────────────

    @staticmethod
    def verify_proof(
        root_hash: bytes,
        key: bytes,
        proof: StateProof,
    ) -> Optional[bytes]:
        """
        Verify a state proof against a given root hash.
        Returns the value if valid, None otherwise.

        Verification:
          1. The first node in the path must hash to the root.
          2. Walk the encoded nodes, following the key nibbles.
          3. The final node must contain the expected value.
        """
        if not proof.nodes:
            return None

        # Verify root matches
        first_node_hash = sha512(proof.nodes[0])
        if first_node_hash != root_hash:
            return None

        # Walk path verifying connectivity
        nibbles = bytes_to_nibbles(key)
        pos = 0  # position in nibbles

        for i, encoded in enumerate(proof.nodes):
            tag = encoded[0:1]

            if tag == b"\x01":
                # Leaf node
                nibble_count = int.from_bytes(encoded[1:3], "big")
                # Decode the nibble bytes
                nibble_byte_len = (nibble_count + 1) // 2
                nibble_bytes = encoded[3: 3 + nibble_byte_len]
                leaf_nibbles = bytes_to_nibbles(nibble_bytes)[:nibble_count]

                if leaf_nibbles == list(nibbles[pos:]):
                    val_offset = 3 + nibble_byte_len
                    val_len = int.from_bytes(encoded[val_offset: val_offset + 4], "big")
                    val = encoded[val_offset + 4: val_offset + 4 + val_len]
                    return val
                return None

            elif tag == b"\x02":
                # Extension node - consume shared prefix
                nibble_count = int.from_bytes(encoded[1:3], "big")
                nibble_byte_len = (nibble_count + 1) // 2
                nibble_bytes = encoded[3: 3 + nibble_byte_len]
                ext_nibbles = bytes_to_nibbles(nibble_bytes)[:nibble_count]

                remaining = nibbles[pos:]
                if remaining[:nibble_count] != ext_nibbles:
                    return None
                pos += nibble_count

                # Next encoded node in the proof list should be the child
                if i + 1 < len(proof.nodes):
                    child_hash_from_ext = encoded[3 + nibble_byte_len: 3 + nibble_byte_len + 64]
                    actual_child_hash = sha512(proof.nodes[i + 1])
                    if child_hash_from_ext != actual_child_hash:
                        return None
                continue

            elif tag == b"\x03":
                # Branch node
                remaining = nibbles[pos:]
                if len(remaining) == 0:
                    # Value is at this branch
                    value_section = encoded[1 + 16 * 64:]
                    if value_section[0:1] == b"\x01":
                        val_len = int.from_bytes(value_section[1:5], "big")
                        return value_section[5: 5 + val_len]
                    return None

                idx = remaining[0]
                pos += 1

                # Verify next node hash matches branch's child slot
                child_hash_start = 1 + idx * 64
                child_hash_from_branch = encoded[child_hash_start: child_hash_start + 64]

                if child_hash_from_branch == b"\x00" * 64:
                    return None

                if i + 1 < len(proof.nodes):
                    actual_child_hash = sha512(proof.nodes[i + 1])
                    if child_hash_from_branch != actual_child_hash:
                        return None
                continue

            else:
                return None

        return None
