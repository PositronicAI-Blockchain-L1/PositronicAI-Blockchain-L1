"""
Positronic - Trie Node Persistent Store
Content-addressable storage for MPT nodes.
Each node is stored by its SHA-512 hash.
Supports transparent AES-256-GCM encryption at rest.
"""
import logging
import sqlite3
import threading
from typing import Dict, Optional
from collections import OrderedDict

from positronic.crypto.hashing import sha512
from positronic.storage.database import DatabaseEncryption

logger = logging.getLogger("positronic.storage.trie_store")


class TrieNodeStore:
    """Persistent storage for Merkle Patricia Trie nodes.

    Stores serialized trie nodes keyed by their SHA-512 hash.
    Includes an LRU cache for frequently accessed nodes.
    Supports optional AES-256-GCM encryption at rest.
    """

    DEFAULT_CACHE_SIZE = 10_000

    def __init__(
        self,
        db_path: str,
        cache_size: int = DEFAULT_CACHE_SIZE,
        encryption_password: Optional[str] = None,
    ):
        self._db_path = db_path
        self._cache_size = cache_size
        self._cache: OrderedDict[bytes, bytes] = OrderedDict()
        self._pending: Dict[bytes, bytes] = {}  # uncommitted writes
        self._lock = threading.Lock()
        self._encryption_password = encryption_password
        self._active_path = db_path
        self._init_db()

    def _init_db(self):
        # Load into memory — no plaintext on disk
        self._in_memory = False
        if self._encryption_password and DatabaseEncryption.is_encrypted(self._db_path):
            self._conn = DatabaseEncryption.load_to_memory_connection(
                self._db_path, self._encryption_password
            )
            self._in_memory = True
        else:
            self._conn = sqlite3.connect(self._active_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trie_nodes (
                node_hash BLOB PRIMARY KEY,
                data BLOB NOT NULL
            )
        """)
        # Also store the root hash for each state version
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trie_roots (
                height INTEGER PRIMARY KEY,
                root_hash BLOB NOT NULL
            )
        """)
        self._conn.commit()

    @property
    def is_encrypted(self) -> bool:
        """Check if encryption is enabled."""
        return self._encryption_password is not None

    def get(self, node_hash: bytes) -> Optional[bytes]:
        """Get a serialized node by its hash. Cache-first."""
        with self._lock:
            # Check cache first
            if node_hash in self._cache:
                self._cache.move_to_end(node_hash)
                return self._cache[node_hash]
            # Check pending writes
            if node_hash in self._pending:
                return self._pending[node_hash]

        # Fall through to DB
        row = self._conn.execute(
            "SELECT data FROM trie_nodes WHERE node_hash = ?", (node_hash,)
        ).fetchone()
        if row is None:
            return None
        data = row[0]

        with self._lock:
            self._cache_put(node_hash, data)
        return data

    def put(self, node_hash: bytes, data: bytes):
        """Stage a node for writing."""
        with self._lock:
            self._pending[node_hash] = data
            self._cache_put(node_hash, data)

    def commit(self, height: Optional[int] = None, root_hash: Optional[bytes] = None):
        """Flush pending writes to database."""
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()

        if pending:
            self._conn.executemany(
                "INSERT OR IGNORE INTO trie_nodes (node_hash, data) VALUES (?, ?)",
                list(pending.items()),
            )
        if height is not None and root_hash is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO trie_roots (height, root_hash) VALUES (?, ?)",
                (height, root_hash),
            )
        try:
            self._conn.commit()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "disk" in err_msg or "full" in err_msg or "i/o" in err_msg:
                try:
                    self._conn.rollback()
                except Exception as rb_err:
                    logger.debug("Rollback failed during disk-full handling: %s", rb_err)
                from positronic.storage import DiskFullError
                raise DiskFullError(f"Trie store disk full: {e}") from e
            raise

    def get_root_hash(self, height: int) -> Optional[bytes]:
        """Get the root hash stored for a specific block height."""
        row = self._conn.execute(
            "SELECT root_hash FROM trie_roots WHERE height = ?", (height,)
        ).fetchone()
        return row[0] if row else None

    def get_latest_root(self) -> Optional[bytes]:
        """Get the most recent root hash."""
        row = self._conn.execute(
            "SELECT root_hash FROM trie_roots ORDER BY height DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: bytes, value: bytes):
        """Add to LRU cache, evicting oldest if full."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def encrypt_at_rest(self) -> None:
        """Re-encrypt the database file on disk (call on shutdown)."""
        if not self._encryption_password:
            return
        if self._in_memory:
            DatabaseEncryption.save_from_memory_connection(
                self._conn,
                self._db_path,
                self._encryption_password,
            )
            return
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.debug("WAL checkpoint failed during encrypt_at_rest: %s", e)
        self.close()
        if self._active_path != self._db_path:
            import shutil, os
            shutil.copy2(self._active_path, self._db_path)
            try:
                os.remove(self._active_path)
            except OSError as e:
                logger.debug("Could not remove decrypted temp file: %s", e)
            self._active_path = self._db_path
        DatabaseEncryption.encrypt_file(self._db_path, self._encryption_password)

    def close(self):
        """Close database connection."""
        self._conn.close()

    def node_count(self) -> int:
        """Total number of stored nodes."""
        row = self._conn.execute("SELECT COUNT(*) FROM trie_nodes").fetchone()
        return row[0]
