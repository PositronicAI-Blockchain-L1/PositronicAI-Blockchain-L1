"""
Positronic - Token & NFT Registry
Central registry for all PRC-20 tokens and PRC-721 NFT collections.
Handles creation, lookup, and fee collection.
"""

from typing import Dict, Optional, List
import hashlib
import json
import sqlite3
import threading
import time

from positronic.tokens.prc20 import PRC20Token
from positronic.tokens.prc721 import PRC721Collection, NFTMetadata
from positronic.constants import BASE_UNIT, CHAIN_ID


# Cost to create tokens (in base units) — burned
TOKEN_CREATION_FEE = 10 * BASE_UNIT   # 10 ASF
NFT_COLLECTION_FEE = 5 * BASE_UNIT    # 5 ASF


class TokenRegistry:
    """Central registry for all tokens and NFT collections on Positronic."""

    def __init__(self, db_path: str = None):
        self._tokens: Dict[str, PRC20Token] = {}
        self._collections: Dict[str, PRC721Collection] = {}
        self._total_tokens_created: int = 0
        self._total_collections_created: int = 0
        self._total_fees_collected: int = 0
        self._nonce: int = 0  # Internal nonce for deterministic ID generation

        # Persistence
        self._db_conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        if db_path is not None:
            self._init_db(db_path)
            self._load_from_db()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _init_db(self, db_path: str) -> None:
        """Open/create the tokens SQLite database and create tables."""
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.execute("PRAGMA synchronous=NORMAL")
        self._db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS prc20_tokens (
                token_id TEXT PRIMARY KEY,
                data     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prc721_collections (
                collection_id TEXT PRIMARY KEY,
                data          TEXT NOT NULL
            );
        """)
        self._db_conn.commit()

    def _safe_db_commit(self) -> None:
        """Commit with error suppression (never crash on persistence failure)."""
        try:
            self._db_conn.commit()
        except Exception as exc:  # pragma: no cover
            pass  # Non-fatal: in-memory state is authoritative

    def _load_from_db(self) -> None:
        """Reload all tokens and collections from the database on startup."""
        if self._db_conn is None:
            return
        with self._db_lock:
            # Load PRC-20 tokens
            cursor = self._db_conn.execute("SELECT token_id, data FROM prc20_tokens")
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    token = PRC20Token.from_dict(data)
                    self._tokens[row[0]] = token
                    self._total_tokens_created += 1
                    self._total_fees_collected += TOKEN_CREATION_FEE
                    if data.get("_nonce_hint", 0) > self._nonce:
                        self._nonce = data["_nonce_hint"]
                except Exception:
                    pass  # Skip corrupted rows

            # Load PRC-721 collections
            cursor = self._db_conn.execute(
                "SELECT collection_id, data FROM prc721_collections"
            )
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    collection = PRC721Collection.from_dict(data)
                    self._collections[row[0]] = collection
                    self._total_collections_created += 1
                    self._total_fees_collected += NFT_COLLECTION_FEE
                    if data.get("_nonce_hint", 0) > self._nonce:
                        self._nonce = data["_nonce_hint"]
                except Exception:
                    pass  # Skip corrupted rows

    def save_token(self, token_id: str) -> None:
        """Persist a single PRC-20 token to the database."""
        if self._db_conn is None:
            return
        token = self._tokens.get(token_id)
        if token is None:
            return
        data = token.to_full_dict()
        data["_nonce_hint"] = self._nonce
        with self._db_lock:
            self._db_conn.execute(
                "INSERT OR REPLACE INTO prc20_tokens (token_id, data) VALUES (?, ?)",
                (token_id, json.dumps(data)),
            )
            self._safe_db_commit()

    def save_collection(self, collection_id: str) -> None:
        """Persist a single PRC-721 collection to the database."""
        if self._db_conn is None:
            return
        collection = self._collections.get(collection_id)
        if collection is None:
            return
        data = collection.to_full_dict()
        data["_nonce_hint"] = self._nonce
        with self._db_lock:
            self._db_conn.execute(
                "INSERT OR REPLACE INTO prc721_collections (collection_id, data) VALUES (?, ?)",
                (collection_id, json.dumps(data)),
            )
            self._safe_db_commit()

    # ------------------------------------------------------------------

    def _generate_id(self, name: str, creator: bytes) -> str:
        """
        Generate deterministic unique ID for token/collection.
        Uses creator + name + nonce + chain_id — reproducible across nodes.
        (Audit fix: replaced non-deterministic time.time())
        """
        self._nonce += 1
        data = (
            creator
            + name.encode()
            + self._nonce.to_bytes(8, "big")
            + CHAIN_ID.to_bytes(4, "big")
        )
        return hashlib.sha256(data).hexdigest()[:16]

    def create_token(
        self,
        name: str,
        symbol: str,
        decimals: int,
        total_supply: int,
        owner: bytes,
    ) -> Optional[PRC20Token]:
        """Create and register a new PRC-20 token."""
        if not name or not symbol:
            return None
        if decimals < 0 or decimals > 18:
            return None
        if total_supply < 0:
            return None

        # Check symbol not already used
        for token in self._tokens.values():
            if token.symbol.upper() == symbol.upper():
                return None

        token_id = self._generate_id(name, owner)
        token = PRC20Token(
            name=name,
            symbol=symbol,
            decimals=decimals,
            total_supply=total_supply,
            owner=owner,
            token_id=token_id,
        )

        self._tokens[token_id] = token
        self._total_tokens_created += 1
        self._total_fees_collected += TOKEN_CREATION_FEE
        self.save_token(token_id)
        return token

    def create_collection(
        self,
        name: str,
        symbol: str,
        owner: bytes,
        max_supply: int = 0,
    ) -> Optional[PRC721Collection]:
        """Create and register a new PRC-721 NFT collection."""
        if not name or not symbol:
            return None

        collection_id = self._generate_id(name, owner)
        collection = PRC721Collection(
            name=name,
            symbol=symbol,
            owner=owner,
            collection_id=collection_id,
            max_supply=max_supply,
        )

        self._collections[collection_id] = collection
        self._total_collections_created += 1
        self._total_fees_collected += NFT_COLLECTION_FEE
        self.save_collection(collection_id)
        return collection

    def get_token(self, token_id: str) -> Optional[PRC20Token]:
        """Get token by ID."""
        return self._tokens.get(token_id)

    def get_token_by_symbol(self, symbol: str) -> Optional[PRC20Token]:
        """Get token by symbol."""
        for token in self._tokens.values():
            if token.symbol.upper() == symbol.upper():
                return token
        return None

    def get_collection(self, collection_id: str) -> Optional[PRC721Collection]:
        """Get NFT collection by ID."""
        return self._collections.get(collection_id)

    def list_tokens(self) -> List[dict]:
        """List all registered tokens."""
        return [t.to_dict() for t in self._tokens.values()]

    def list_collections(self) -> List[dict]:
        """List all registered NFT collections."""
        return [c.to_dict() for c in self._collections.values()]

    def get_tokens_by_owner(self, owner: bytes) -> List[dict]:
        """Get all tokens created by an owner."""
        return [
            t.to_dict() for t in self._tokens.values()
            if t.owner == owner
        ]

    def get_collections_by_owner(self, owner: bytes) -> List[dict]:
        """Get all collections created by an owner."""
        return [
            c.to_dict() for c in self._collections.values()
            if c.owner == owner
        ]

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_tokens": self._total_tokens_created,
            "total_collections": self._total_collections_created,
            "active_tokens": len(self._tokens),
            "active_collections": len(self._collections),
            "total_fees_collected": self._total_fees_collected,
            "total_fees_asf": self._total_fees_collected / BASE_UNIT,
            "token_creation_fee_asf": TOKEN_CREATION_FEE / BASE_UNIT,
            "nft_collection_fee_asf": NFT_COLLECTION_FEE / BASE_UNIT,
        }
