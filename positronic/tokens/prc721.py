"""
Positronic - PRC-721 Non-Fungible Token (NFT) Standard
Equivalent to Ethereum's ERC-721. Supports dynamic NFTs and AI verification.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import time


@dataclass
class NFTMetadata:
    """Metadata for an individual NFT."""
    name: str = ""
    description: str = ""
    image_uri: str = ""
    attributes: dict = field(default_factory=dict)
    ai_verified: bool = False
    ai_score: float = 0.0
    created_at: float = 0.0
    dynamic: bool = False  # Can metadata change?

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "image_uri": self.image_uri,
            "attributes": self.attributes,
            "ai_verified": self.ai_verified,
            "ai_score": self.ai_score,
            "created_at": self.created_at,
            "dynamic": self.dynamic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NFTMetadata":
        return cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            image_uri=d.get("image_uri", ""),
            attributes=d.get("attributes", {}),
            ai_verified=d.get("ai_verified", False),
            ai_score=d.get("ai_score", 0.0),
            created_at=d.get("created_at", 0.0),
            dynamic=d.get("dynamic", False),
        )


@dataclass
class PRC721Collection:
    """PRC-721 NFT collection on the Positronic blockchain."""
    name: str
    symbol: str
    owner: bytes  # Collection creator
    created_at: float = 0.0
    collection_id: str = ""
    max_supply: int = 0  # 0 = unlimited

    # State
    _tokens: Dict[int, bytes] = field(default_factory=dict)  # token_id -> owner
    _metadata: Dict[int, NFTMetadata] = field(default_factory=dict)
    _next_token_id: int = 1
    _total_minted: int = 0
    _total_burned: int = 0
    _transfer_count: int = 0
    _approvals: Dict[int, bytes] = field(default_factory=dict)  # token_id -> approved
    _operator_approvals: Dict[tuple, bool] = field(default_factory=dict)  # (owner, op) -> bool

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def owner_of(self, token_id: int) -> Optional[bytes]:
        """Get owner of a specific token."""
        return self._tokens.get(token_id)

    def balance_of(self, address: bytes) -> int:
        """Count NFTs owned by address."""
        return sum(1 for owner in self._tokens.values() if owner == address)

    def get_metadata(self, token_id: int) -> Optional[NFTMetadata]:
        """Get metadata for a token."""
        return self._metadata.get(token_id)

    def mint(
        self,
        to: bytes,
        metadata: Optional[NFTMetadata] = None,
        token_id: Optional[int] = None,
    ) -> Optional[int]:
        """Mint a new NFT. Returns token_id or None on failure."""
        if self.max_supply > 0 and self._total_minted >= self.max_supply:
            return None

        if token_id is None:
            token_id = self._next_token_id
            self._next_token_id += 1
        elif token_id in self._tokens:
            return None  # Already exists

        self._tokens[token_id] = to
        if metadata:
            if metadata.created_at == 0.0:
                metadata.created_at = time.time()
            self._metadata[token_id] = metadata
        else:
            self._metadata[token_id] = NFTMetadata(created_at=time.time())

        self._total_minted += 1
        if token_id >= self._next_token_id:
            self._next_token_id = token_id + 1
        return token_id

    def transfer(self, from_addr: bytes, to_addr: bytes, token_id: int) -> bool:
        """Transfer NFT to another address."""
        owner = self.owner_of(token_id)
        if owner is None or owner != from_addr:
            return False
        if from_addr == to_addr:
            return False

        self._tokens[token_id] = to_addr
        # Clear approval
        self._approvals.pop(token_id, None)
        self._transfer_count += 1
        return True

    def approve(self, owner: bytes, approved: bytes, token_id: int) -> bool:
        """Approve another address to transfer a specific token."""
        if self.owner_of(token_id) != owner:
            return False
        self._approvals[token_id] = approved
        return True

    def get_approved(self, token_id: int) -> Optional[bytes]:
        """Get approved address for a token."""
        return self._approvals.get(token_id)

    def set_approval_for_all(
        self, owner: bytes, operator: bytes, approved: bool
    ) -> bool:
        """Approve/revoke operator for all of owner's tokens."""
        self._operator_approvals[(owner, operator)] = approved
        return True

    def is_approved_for_all(self, owner: bytes, operator: bytes) -> bool:
        """Check if operator is approved for all of owner's tokens."""
        return self._operator_approvals.get((owner, operator), False)

    def burn(self, token_id: int, sender: bytes) -> bool:
        """Burn (destroy) an NFT."""
        owner = self.owner_of(token_id)
        if owner is None or owner != sender:
            return False
        del self._tokens[token_id]
        self._metadata.pop(token_id, None)
        self._approvals.pop(token_id, None)
        self._total_burned += 1
        return True

    def update_metadata(
        self, token_id: int, sender: bytes, updates: dict
    ) -> bool:
        """Update dynamic NFT metadata. Only works on dynamic NFTs."""
        meta = self.get_metadata(token_id)
        if meta is None:
            return False
        if not meta.dynamic:
            return False
        owner = self.owner_of(token_id)
        if owner != sender:
            return False

        if "name" in updates:
            meta.name = updates["name"]
        if "description" in updates:
            meta.description = updates["description"]
        if "image_uri" in updates:
            meta.image_uri = updates["image_uri"]
        if "attributes" in updates:
            meta.attributes.update(updates["attributes"])
        return True

    def ai_verify(self, token_id: int, score: float) -> bool:
        """Mark NFT as AI-verified with a confidence score."""
        meta = self.get_metadata(token_id)
        if meta is None:
            return False
        meta.ai_verified = True
        meta.ai_score = score
        return True

    def get_tokens_of(self, address: bytes) -> List[int]:
        """Get all token IDs owned by an address."""
        return [tid for tid, owner in self._tokens.items() if owner == address]

    @property
    def total_supply(self) -> int:
        """Current total supply (minted - burned)."""
        return len(self._tokens)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "owner": self.owner.hex(),
            "collection_id": self.collection_id,
            "created_at": self.created_at,
            "max_supply": self.max_supply,
            "total_supply": self.total_supply,
            "total_minted": self._total_minted,
            "total_burned": self._total_burned,
            "transfer_count": self._transfer_count,
        }

    def to_full_dict(self) -> dict:
        """Serialize full collection state (including per-token data) for persistence."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "owner": self.owner.hex(),
            "collection_id": self.collection_id,
            "created_at": self.created_at,
            "max_supply": self.max_supply,
            "total_minted": self._total_minted,
            "total_burned": self._total_burned,
            "transfer_count": self._transfer_count,
            "next_token_id": self._next_token_id,
            # tokens: {str(token_id): owner_hex}
            "tokens": {
                str(tid): owner.hex()
                for tid, owner in self._tokens.items()
            },
            # metadata: {str(token_id): metadata_dict}
            "metadata": {
                str(tid): meta.to_dict()
                for tid, meta in self._metadata.items()
            },
            # approvals: {str(token_id): approved_hex}
            "approvals": {
                str(tid): addr.hex()
                for tid, addr in self._approvals.items()
            },
            # operator_approvals: {"owner_hex:operator_hex": bool}
            "operator_approvals": {
                f"{owner.hex()}:{op.hex()}": approved
                for (owner, op), approved in self._operator_approvals.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PRC721Collection":
        """Reconstruct PRC721Collection from a full_dict snapshot."""
        owner_bytes = bytes.fromhex(d["owner"])
        collection = cls(
            name=d["name"],
            symbol=d["symbol"],
            owner=owner_bytes,
            created_at=d.get("created_at", 0.0),
            collection_id=d.get("collection_id", ""),
            max_supply=d.get("max_supply", 0),
        )
        collection._total_minted = d.get("total_minted", 0)
        collection._total_burned = d.get("total_burned", 0)
        collection._transfer_count = d.get("transfer_count", 0)
        collection._next_token_id = d.get("next_token_id", 1)
        # Restore tokens
        for tid_str, owner_hex in d.get("tokens", {}).items():
            collection._tokens[int(tid_str)] = bytes.fromhex(owner_hex)
        # Restore metadata
        for tid_str, meta_dict in d.get("metadata", {}).items():
            collection._metadata[int(tid_str)] = NFTMetadata.from_dict(meta_dict)
        # Restore approvals
        for tid_str, addr_hex in d.get("approvals", {}).items():
            collection._approvals[int(tid_str)] = bytes.fromhex(addr_hex)
        # Restore operator approvals
        for key, approved in d.get("operator_approvals", {}).items():
            owner_hex, op_hex = key.split(":", 1)
            collection._operator_approvals[
                (bytes.fromhex(owner_hex), bytes.fromhex(op_hex))
            ] = approved
        return collection
