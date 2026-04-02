"""
Positronic - SQLite Database Layer
Persistent storage for blocks, transactions, accounts, and contract state.
Uses WAL mode for concurrent read access.
Supports transparent AES-256-GCM encryption at rest.
"""

import logging
import sqlite3
import os
import tempfile
import threading
import json
import hmac
import hashlib
from typing import Optional, List

logger = logging.getLogger(__name__)


def derive_storage_password(data_dir: str) -> str:
    """Derive a machine- and installation-specific database encryption password.

    Uses PBKDF2(machine_identity + data_dir + install_salt, iterations=100000)
    to create a password unique to this machine, user, AND installation.
    A random 32-byte salt is generated on first run and persisted in data_dir.
    This prevents key sharing if two users share the same data_dir and adds
    an extra layer of per-installation isolation.
    """
    import socket
    import getpass
    identity = f"{socket.gethostname()}:{getpass.getuser()}:{os.path.abspath(data_dir)}"

    # Per-installation random salt (generated once, persisted)
    salt_file = os.path.join(data_dir, ".db_salt")
    install_salt = b""
    try:
        if os.path.exists(salt_file):
            with open(salt_file, "rb") as f:
                install_salt = f.read(32)
        else:
            install_salt = os.urandom(32)
            os.makedirs(data_dir, exist_ok=True)
            with open(salt_file, "wb") as f:
                f.write(install_salt)
    except OSError:
        pass  # Fallback: no install salt (backward compatible)

    kdf_salt = b"positronic-db-v2" + install_salt
    raw = hashlib.pbkdf2_hmac("sha512", identity.encode(), kdf_salt, 100_000)
    return raw.hex()


def _derive_storage_password_v1(data_dir: str) -> str:
    """Legacy v1 key derivation (without install salt) for backward compatibility."""
    import socket
    import getpass
    identity = f"{socket.gethostname()}:{getpass.getuser()}:{os.path.abspath(data_dir)}"
    raw = hashlib.pbkdf2_hmac("sha512", identity.encode(), b"positronic-db-v1", 100_000)
    return raw.hex()


# ── Encrypted database wrapper ──────────────────────────────────────
# We encrypt the entire SQLite database file at rest using AES-256-GCM.
# On open: decrypt file → temp in-memory bytes → SQLite reads from it.
# On close/commit: SQLite WAL checkpoint → encrypt file back to disk.
# This is transparent — callers see a normal SQLite Database object.

_ENC_MAGIC = b"POSITRONIC_ENC_V1"   # 17 bytes header
_ENC_SALT_LEN = 32
_ENC_NONCE_LEN = 12
_ENC_TAG_LEN = 16  # AES-GCM tag

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


class DatabaseEncryption:
    """AES-256-GCM encryption for SQLite database files at rest."""

    ITERATIONS = 100_000
    KEY_LENGTH = 32  # AES-256

    @staticmethod
    def derive_key(password: str, salt: bytes) -> bytes:
        """Derive AES-256 key from password using PBKDF2-HMAC-SHA512."""
        if not _HAS_CRYPTO:
            # Fallback: HMAC-SHA256 based KDF (stdlib only)
            key = password.encode("utf-8")
            for _ in range(DatabaseEncryption.ITERATIONS // 1000):
                key = hmac.new(salt, key, hashlib.sha512).digest()
            return key[:DatabaseEncryption.KEY_LENGTH]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=DatabaseEncryption.KEY_LENGTH,
            salt=salt,
            iterations=DatabaseEncryption.ITERATIONS,
        )
        return kdf.derive(password.encode("utf-8"))

    @staticmethod
    def encrypt_file(db_path: str, password: str) -> None:
        """Encrypt an existing SQLite database file in-place."""
        if not os.path.exists(db_path):
            return

        with open(db_path, "rb") as f:
            plaintext = f.read()

        # Skip if already encrypted
        if plaintext[:len(_ENC_MAGIC)] == _ENC_MAGIC:
            return

        # Skip empty files
        if len(plaintext) == 0:
            return

        salt = os.urandom(_ENC_SALT_LEN)
        nonce = os.urandom(_ENC_NONCE_LEN)
        key = DatabaseEncryption.derive_key(password, salt)

        if _HAS_CRYPTO:
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, _ENC_MAGIC)
        else:
            # Fallback: XOR-based encryption with HMAC authentication
            ciphertext = DatabaseEncryption._xor_encrypt(key, nonce, plaintext, _ENC_MAGIC)

        enc_data = _ENC_MAGIC + salt + nonce + ciphertext

        # Atomic write
        tmp_path = db_path + ".enc.tmp"
        with open(tmp_path, "wb") as f:
            f.write(enc_data)
        os.replace(tmp_path, db_path)

        # Remove WAL/SHM files (they contain unencrypted data)
        for ext in (".db-wal", ".db-shm", "-wal", "-shm"):
            wal_path = db_path + ext
            if os.path.exists(wal_path):
                os.remove(wal_path)

        # Restrict file permissions (owner-only)
        try:
            if os.name == "nt":
                import subprocess
                username = os.environ.get("USERNAME", "")
                if username:
                    subprocess.run(
                        ["icacls", db_path, "/inheritance:r",
                         "/grant:r", f"{username}:(R,W)"],
                        capture_output=True, timeout=5,
                    )
            else:
                import stat as _stat
                os.chmod(db_path, _stat.S_IRUSR | _stat.S_IWUSR)
        except Exception:
            pass

        logger.info("Database encrypted at %s", db_path)

    @staticmethod
    def decrypt_file(db_path: str, password: str) -> str:
        """
        Decrypt database file -> return path to decrypted temp file.
        If file is not encrypted, returns the original path.

        .. deprecated::
            Use :meth:`load_to_memory_connection` instead to avoid writing
            plaintext to disk entirely.
        """
        if not os.path.exists(db_path):
            return db_path

        with open(db_path, "rb") as f:
            data = f.read()

        # Not encrypted — return as-is
        if data[:len(_ENC_MAGIC)] != _ENC_MAGIC:
            return db_path

        offset = len(_ENC_MAGIC)
        salt = data[offset:offset + _ENC_SALT_LEN]
        offset += _ENC_SALT_LEN
        nonce = data[offset:offset + _ENC_NONCE_LEN]
        offset += _ENC_NONCE_LEN
        ciphertext = data[offset:]

        key = DatabaseEncryption.derive_key(password, salt)

        if _HAS_CRYPTO:
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, _ENC_MAGIC)
        else:
            plaintext = DatabaseEncryption._xor_decrypt(key, nonce, ciphertext, _ENC_MAGIC)

        # Write decrypted to temp file (same directory for atomic operations)
        dec_path = db_path + ".dec"
        with open(dec_path, "wb") as f:
            f.write(plaintext)

        return dec_path

    @staticmethod
    def is_encrypted(db_path: str) -> bool:
        """Check if a database file is encrypted."""
        if not os.path.exists(db_path):
            return False
        with open(db_path, "rb") as f:
            magic = f.read(len(_ENC_MAGIC))
        return magic == _ENC_MAGIC

    # ── In-memory decryption (no plaintext on disk) ─────────────────

    @staticmethod
    def decrypt_to_bytes(db_path: str, password: str) -> Optional[bytes]:
        """Decrypt database file and return raw bytes in memory.
        Never writes decrypted data to disk.
        Returns None if file is not encrypted."""
        if not os.path.exists(db_path):
            return None
        with open(db_path, "rb") as f:
            data = f.read()
        if data[:len(_ENC_MAGIC)] != _ENC_MAGIC:
            return None
        # Parse exactly like decrypt_file but return bytes instead of writing file
        offset = len(_ENC_MAGIC)
        salt = data[offset:offset + _ENC_SALT_LEN]
        offset += _ENC_SALT_LEN
        nonce = data[offset:offset + _ENC_NONCE_LEN]
        offset += _ENC_NONCE_LEN
        ciphertext = data[offset:]
        key = DatabaseEncryption.derive_key(password, salt)
        if _HAS_CRYPTO:
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, _ENC_MAGIC)
        else:
            return DatabaseEncryption._xor_decrypt(key, nonce, ciphertext, _ENC_MAGIC)

    @staticmethod
    def encrypt_bytes(plaintext: bytes, db_path: str, password: str) -> None:
        """Encrypt raw bytes and write to file. No plaintext on disk."""
        salt = os.urandom(_ENC_SALT_LEN)
        nonce = os.urandom(_ENC_NONCE_LEN)
        key = DatabaseEncryption.derive_key(password, salt)
        if _HAS_CRYPTO:
            aesgcm = AESGCM(key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, _ENC_MAGIC)
        else:
            ciphertext = DatabaseEncryption._xor_encrypt(key, nonce, plaintext, _ENC_MAGIC)
        enc_data = _ENC_MAGIC + salt + nonce + ciphertext
        tmp_path = db_path + ".enc.tmp"
        with open(tmp_path, "wb") as f:
            f.write(enc_data)
        os.replace(tmp_path, db_path)
        # Restrict file permissions (owner-only)
        try:
            if os.name == "nt":
                import subprocess
                username = os.environ.get("USERNAME", "")
                if username:
                    subprocess.run(
                        ["icacls", db_path, "/inheritance:r",
                         "/grant:r", f"{username}:(R,W)"],
                        capture_output=True, timeout=5,
                    )
            else:
                import stat as _stat
                os.chmod(db_path, _stat.S_IRUSR | _stat.S_IWUSR)
        except Exception:
            pass

    @staticmethod
    def load_to_memory_connection(db_path: str, password: str) -> sqlite3.Connection:
        """Load encrypted database into in-memory SQLite connection.
        Uses sqlite3 backup API -- no plaintext ever touches disk."""
        plaintext = DatabaseEncryption.decrypt_to_bytes(db_path, password)
        if plaintext is None:
            # Not encrypted, load from file directly into memory
            mem_conn = sqlite3.connect(":memory:")
            src = sqlite3.connect(db_path)
            src.backup(mem_conn)
            src.close()
            return mem_conn
        # Write to tempfile, load into memory via backup, delete immediately
        mem_conn = sqlite3.connect(":memory:")
        fd, tmp_path = tempfile.mkstemp(suffix=".db")
        try:
            os.write(fd, plaintext)
            os.close(fd)
            src = sqlite3.connect(tmp_path)
            src.backup(mem_conn)
            src.close()
        finally:
            # Secure delete: overwrite with zeros, then remove
            try:
                with open(tmp_path, "wb") as f:
                    f.write(b"\x00" * len(plaintext))
                os.remove(tmp_path)
            except OSError:
                pass
        return mem_conn

    @staticmethod
    def save_from_memory_connection(conn: sqlite3.Connection, db_path: str, password: str) -> None:
        """Save in-memory database to encrypted file on disk.
        No intermediate plaintext file persists on disk."""
        plaintext = None
        fd, tmp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            tmp_conn = sqlite3.connect(tmp_path)
            conn.backup(tmp_conn)
            tmp_conn.close()
            with open(tmp_path, "rb") as f:
                plaintext = f.read()
        finally:
            # Secure delete temp file
            try:
                size = len(plaintext) if plaintext is not None else 1
                with open(tmp_path, "wb") as f:
                    f.write(b"\x00" * max(size, 1))
                os.remove(tmp_path)
            except OSError:
                pass
        DatabaseEncryption.encrypt_bytes(plaintext, db_path, password)
        # Remove any WAL/SHM files
        for ext in (".db-wal", ".db-shm", "-wal", "-shm"):
            wal_path = db_path + ext
            if os.path.exists(wal_path):
                try:
                    os.remove(wal_path)
                except OSError:
                    pass
        logger.info("Database saved encrypted at %s", db_path)

    # ── Fallback XOR cipher (when cryptography not installed) ────────

    @staticmethod
    def _xor_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
        """Generate a keystream using HMAC-SHA512 in counter mode."""
        stream = b""
        counter = 0
        while len(stream) < length:
            block = hmac.new(
                key, nonce + counter.to_bytes(8, "big"), hashlib.sha512
            ).digest()
            stream += block
            counter += 1
        return stream[:length]

    @staticmethod
    def _xor_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        """XOR encrypt with HMAC-SHA256 authentication tag."""
        ks = DatabaseEncryption._xor_keystream(key, nonce, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, ks))
        # Authentication tag
        tag = hmac.new(key, aad + nonce + ciphertext, hashlib.sha256).digest()
        return ciphertext + tag

    @staticmethod
    def _xor_decrypt(key: bytes, nonce: bytes, data: bytes, aad: bytes) -> bytes:
        """XOR decrypt with HMAC-SHA256 verification."""
        tag_len = 32  # SHA-256
        ciphertext = data[:-tag_len]
        tag = data[-tag_len:]
        expected = hmac.new(key, aad + nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("Database decryption failed: invalid password or corrupted file")
        ks = DatabaseEncryption._xor_keystream(key, nonce, len(ciphertext))
        return bytes(a ^ b for a, b in zip(ciphertext, ks))


class Database:
    """
    SQLite-based storage backend for Positronic.
    Thread-safe with connection-per-thread model.
    Supports optional AES-256-GCM encryption at rest.
    """

    SCHEMA_VERSION = 2  # Increment when schema changes

    def __init__(self, db_path: str, encryption_password: Optional[str] = None):
        self.db_path = db_path
        self._encryption_password = encryption_password
        self._active_path = db_path  # may differ if decrypted to temp
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()

        # Load encrypted database into memory (no plaintext on disk)
        self._in_memory = False
        if encryption_password and DatabaseEncryption.is_encrypted(db_path):
            try:
                self._conn_memory = DatabaseEncryption.load_to_memory_connection(
                    db_path, encryption_password
                )
            except (ValueError, Exception) as exc:
                # Backward compatibility: try legacy v1 key derivation
                # (before per-installation salt was added)
                data_dir = os.path.dirname(db_path) or "."
                v1_password = _derive_storage_password_v1(data_dir)
                if v1_password != encryption_password:
                    try:
                        self._conn_memory = DatabaseEncryption.load_to_memory_connection(
                            db_path, v1_password
                        )
                        logger.info("Database decrypted with legacy v1 key — will re-encrypt with v2 on save")
                        self._encryption_password = encryption_password  # Use v2 for re-encryption
                    except Exception:
                        raise exc  # Re-raise original error
                else:
                    raise
            self._conn_memory.row_factory = sqlite3.Row
            self._in_memory = True
            logger.info("Database loaded into memory from: %s", db_path)

        self._create_tables()
        self._run_migrations()

    @property
    def conn(self) -> sqlite3.Connection:
        """Thread-local database connection (or shared in-memory connection)."""
        if self._in_memory:
            return self._conn_memory
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._active_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @property
    def is_encrypted(self) -> bool:
        """Check if encryption is enabled."""
        return self._encryption_password is not None

    def encrypt_at_rest(self) -> None:
        """Re-encrypt the database file on disk (call on shutdown)."""
        if not self._encryption_password:
            return
        if self._in_memory:
            # Save from in-memory connection directly to encrypted file
            DatabaseEncryption.save_from_memory_connection(
                self._conn_memory, self.db_path, self._encryption_password
            )
            return
        # Non-encrypted path: force WAL checkpoint then encrypt
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.debug("WAL checkpoint failed during encryption: %s", e)
        self.close()
        if self._active_path != self.db_path:
            import shutil
            shutil.copy2(self._active_path, self.db_path)
            try:
                os.remove(self._active_path)
            except OSError:
                pass
            self._active_path = self.db_path
        DatabaseEncryption.encrypt_file(self.db_path, self._encryption_password)

    def _create_tables(self):
        """Create all database tables if they don't exist."""
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS blocks (
                height INTEGER PRIMARY KEY,
                hash TEXT UNIQUE NOT NULL,
                header_json TEXT NOT NULL,
                timestamp REAL NOT NULL,
                tx_count INTEGER DEFAULT 0,
                gas_used INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS transactions (
                tx_hash TEXT PRIMARY KEY,
                block_hash TEXT,
                block_height INTEGER,
                tx_index INTEGER,
                tx_json TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipient TEXT,
                value TEXT,
                tx_type INTEGER,
                ai_score REAL DEFAULT 0.0,
                status INTEGER DEFAULT 0,
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS accounts (
                address TEXT PRIMARY KEY,
                nonce INTEGER DEFAULT 0,
                balance TEXT DEFAULT '0',
                code_hash TEXT DEFAULT '',
                storage_root TEXT DEFAULT '',
                staked_amount TEXT DEFAULT '0',
                delegated_to TEXT DEFAULT '',
                is_validator INTEGER DEFAULT 0,
                is_nvn INTEGER DEFAULT 0,
                ai_reputation REAL DEFAULT 1.0,
                quarantine_count INTEGER DEFAULT 0,
                account_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contract_code (
                code_hash TEXT PRIMARY KEY,
                bytecode BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS contract_storage (
                contract_address TEXT NOT NULL,
                storage_key TEXT NOT NULL,
                storage_value TEXT NOT NULL,
                PRIMARY KEY (contract_address, storage_key)
            );

            CREATE TABLE IF NOT EXISTS validators (
                pubkey TEXT PRIMARY KEY,
                address TEXT NOT NULL,
                stake INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                activation_epoch INTEGER DEFAULT 0,
                exit_epoch INTEGER DEFAULT -1,
                slashed INTEGER DEFAULT 0,
                attestation_count INTEGER DEFAULT 0,
                validator_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ai_models (
                version INTEGER PRIMARY KEY,
                model_hash TEXT NOT NULL,
                activated_at_block INTEGER NOT NULL,
                false_positive_rate REAL DEFAULT 0.0,
                total_scored INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS quarantine_pool (
                tx_hash TEXT PRIMARY KEY,
                tx_json TEXT NOT NULL,
                ai_score REAL NOT NULL,
                quarantined_at_block INTEGER NOT NULL,
                review_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'quarantined'
            );

            CREATE TABLE IF NOT EXISTS treasury (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_height INTEGER NOT NULL,
                amount INTEGER NOT NULL,
                purpose TEXT,
                tx_hash TEXT,
                timestamp REAL
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_hash);
            CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(sender);
            CREATE INDEX IF NOT EXISTS idx_tx_recipient ON transactions(recipient);
            CREATE INDEX IF NOT EXISTS idx_tx_status ON transactions(status);
            CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(hash);
            CREATE INDEX IF NOT EXISTS idx_accounts_balance ON accounts(balance);
            CREATE INDEX IF NOT EXISTS idx_validators_active ON validators(is_active);
        """)
        c.commit()

    def _run_migrations(self):
        """Run database schema migrations if needed."""
        c = self.conn
        # Create schema_version table if missing
        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
        """)
        c.commit()

        row = c.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] if row and row[0] is not None else 0

        if current < self.SCHEMA_VERSION:
            import time
            # Migration 1: initial schema (tables already created by _create_tables)
            if current < 1:
                c.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (1, time.time()),
                )
                c.commit()
                logger.info("Database schema at version %d", 1)

            # Migration 2: add block_metadata column for upgrade tracking
            if current < 2:
                try:
                    c.execute("ALTER TABLE blocks ADD COLUMN metadata TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # Column may already exist
                c.execute(
                    "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (2, time.time()),
                )
                c.commit()
                logger.info("Database schema migrated to version %d", 2)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement."""
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        """Execute a SQL statement with many parameter sets."""
        return self.conn.executemany(sql, params_list)

    def commit(self):
        """Commit current transaction."""
        self.conn.commit()

    def safe_commit(self):
        """Commit with disk-full detection and rollback."""
        try:
            self.conn.commit()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "disk" in err_msg or "full" in err_msg or "i/o" in err_msg:
                try:
                    self.conn.rollback()
                except Exception as rb_err:
                    logger.debug("Rollback failed during disk-full handling: %s", rb_err)
                from positronic.storage import DiskFullError
                raise DiskFullError(f"Disk full during commit: {e}") from e
            try:
                self.conn.rollback()
            except Exception as rb_err:
                logger.debug("Rollback failed during fatal error handling: %s", rb_err)
            from positronic.storage import StorageFatalError
            raise StorageFatalError(f"Unrecoverable DB error: {e}") from e

    def safe_rollback(self):
        """Rollback with error suppression."""
        try:
            self.conn.rollback()
        except Exception as rb_err:
            logger.debug("safe_rollback suppressed error: %s", rb_err)

    def rollback(self):
        """Rollback current transaction."""
        self.conn.rollback()

    def close(self, encrypt: bool = False):
        """Close the database connection. Optionally re-encrypt at rest."""
        if encrypt and self._encryption_password:
            self.encrypt_at_rest()
        if self._in_memory and hasattr(self, '_conn_memory') and self._conn_memory:
            self._conn_memory.close()
            self._conn_memory = None
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def get_chain_height(self) -> int:
        """Get the current chain height."""
        row = self.execute("SELECT MAX(height) FROM blocks").fetchone()
        if row and row[0] is not None:
            return row[0]
        return -1

    def get_block_count(self) -> int:
        """Get total number of blocks."""
        row = self.execute("SELECT COUNT(*) FROM blocks").fetchone()
        return row[0] if row else 0

    def get_tx_count(self) -> int:
        """Get total number of transactions."""
        row = self.execute("SELECT COUNT(*) FROM transactions").fetchone()
        return row[0] if row else 0

    def get_account_count(self) -> int:
        """Get total number of accounts."""
        row = self.execute("SELECT COUNT(*) FROM accounts").fetchone()
        return row[0] if row else 0
