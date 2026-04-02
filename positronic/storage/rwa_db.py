"""
Positronic - RWA Tokenization Persistence Layer (Phase 30)
SQLite storage for RWA tokens, holders, compliance, and dividends.
Supports transparent AES-256-GCM encryption at rest.
"""

import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from positronic.utils.logging import get_logger
from positronic.storage.database import DatabaseEncryption

logger = get_logger("rwa_db")

RWA_DB_VERSION = 1


class RWADatabase:
    """
    SQLite persistence for all RWA tokenization data.
    Supports optional AES-256-GCM encryption at rest.

    Tables:
    - rwa_tokens: Registered RWA token metadata
    - rwa_holders: Token holder balances
    - rwa_kyc: KYC records linked to DID credentials
    - rwa_dividends: Dividend distribution history
    - rwa_meta: Key-value metadata store
    """

    def __init__(
        self,
        db_path: str = "rwa_data.db",
        encryption_password: Optional[str] = None,
    ):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._encryption_password = encryption_password
        self._active_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        self._in_memory = False
        if self._encryption_password and DatabaseEncryption.is_encrypted(self.db_path):
            self._conn = DatabaseEncryption.load_to_memory_connection(
                self.db_path, self._encryption_password
            )
            self._in_memory = True
        else:
            self._conn = sqlite3.connect(self._active_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rwa_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rwa_tokens (
                token_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                decimals INTEGER DEFAULT 18,
                total_supply TEXT DEFAULT '0',
                issuer_hex TEXT NOT NULL,
                asset_type INTEGER DEFAULT 0,
                status INTEGER DEFAULT 0,
                description TEXT DEFAULT '',
                jurisdiction TEXT DEFAULT '',
                legal_doc_hash TEXT DEFAULT '',
                valuation TEXT DEFAULT '0',
                allowed_jurisdictions_json TEXT DEFAULT '[]',
                min_kyc_level INTEGER DEFAULT 2,
                max_holders INTEGER DEFAULT 100000,
                ai_risk_score REAL DEFAULT 0.0,
                votes_for INTEGER DEFAULT 0,
                votes_against INTEGER DEFAULT 0,
                voters_json TEXT DEFAULT '[]',
                holder_count INTEGER DEFAULT 0,
                total_transfers INTEGER DEFAULT 0,
                total_dividends_distributed TEXT DEFAULT '0',
                created_at REAL DEFAULT 0,
                approved_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS rwa_holders (
                token_id TEXT NOT NULL,
                holder_hex TEXT NOT NULL,
                balance TEXT DEFAULT '0',
                updated_at REAL DEFAULT 0,
                PRIMARY KEY (token_id, holder_hex),
                FOREIGN KEY (token_id) REFERENCES rwa_tokens(token_id)
            );

            CREATE TABLE IF NOT EXISTS rwa_kyc (
                address_hex TEXT PRIMARY KEY,
                kyc_level INTEGER DEFAULT 0,
                jurisdiction TEXT DEFAULT '',
                credential_id TEXT DEFAULT '',
                verified_at REAL DEFAULT 0,
                expires_at REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS rwa_dividends (
                dividend_id TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                total_amount TEXT DEFAULT '0',
                amount_per_token TEXT DEFAULT '0',
                holder_count INTEGER DEFAULT 0,
                issuer_hex TEXT DEFAULT '',
                distributed_at REAL DEFAULT 0,
                payouts_json TEXT DEFAULT '{}',
                FOREIGN KEY (token_id) REFERENCES rwa_tokens(token_id)
            );

            CREATE INDEX IF NOT EXISTS idx_rwa_holders_token
                ON rwa_holders(token_id);
            CREATE INDEX IF NOT EXISTS idx_rwa_dividends_token
                ON rwa_dividends(token_id);
        """)

        self._conn.execute(
            "INSERT OR IGNORE INTO rwa_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(RWA_DB_VERSION)),
        )
        self._safe_commit()
        logger.info("rwa_db_initialized", extra={"path": self._active_path})

    def _safe_commit(self):
        """Commit with disk-full detection."""
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
                raise DiskFullError(f"RWA DB disk full: {e}") from e
            raise

    @property
    def is_encrypted(self) -> bool:
        return self._encryption_password is not None

    # ---- Token Persistence ----

    def save_token(self, token_data: dict):
        """Save or update an RWA token record."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO rwa_tokens (
                    token_id, name, symbol, decimals, total_supply,
                    issuer_hex, asset_type, status, description,
                    jurisdiction, legal_doc_hash, valuation,
                    allowed_jurisdictions_json, min_kyc_level, max_holders,
                    ai_risk_score, votes_for, votes_against, voters_json,
                    holder_count, total_transfers, total_dividends_distributed,
                    created_at, approved_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                token_data.get("token_id", ""),
                token_data.get("name", ""),
                token_data.get("symbol", ""),
                token_data.get("decimals", 18),
                str(token_data.get("total_supply", 0)),
                token_data.get("issuer_hex", ""),
                token_data.get("asset_type", 0),
                token_data.get("status", 0),
                token_data.get("description", ""),
                token_data.get("jurisdiction", ""),
                token_data.get("legal_doc_hash", ""),
                str(token_data.get("valuation", 0)),
                json.dumps(token_data.get("allowed_jurisdictions", [])),
                token_data.get("min_kyc_level", 2),
                token_data.get("max_holders", 100000),
                token_data.get("ai_risk_score", 0.0),
                token_data.get("votes_for", 0),
                token_data.get("votes_against", 0),
                json.dumps(token_data.get("voters", [])),
                token_data.get("holder_count", 0),
                token_data.get("total_transfers", 0),
                str(token_data.get("total_dividends_distributed", 0)),
                token_data.get("created_at", 0),
                token_data.get("approved_at", 0),
            ))
            self._safe_commit()

    def load_all_tokens(self) -> List[dict]:
        """Load all RWA tokens from database."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM rwa_tokens")
            columns = [desc[0] for desc in cursor.description]
            rows = []
            for row in cursor.fetchall():
                d = dict(zip(columns, row))
                d["allowed_jurisdictions"] = json.loads(
                    d.pop("allowed_jurisdictions_json", "[]")
                )
                d["voters"] = json.loads(d.pop("voters_json", "[]"))
                rows.append(d)
            return rows

    def get_token(self, token_id: str) -> Optional[dict]:
        """Get a single token by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM rwa_tokens WHERE token_id = ?", (token_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(zip(columns, row))
            d["allowed_jurisdictions"] = json.loads(
                d.pop("allowed_jurisdictions_json", "[]")
            )
            d["voters"] = json.loads(d.pop("voters_json", "[]"))
            return d

    # ---- Holder Persistence ----

    def save_holders(self, token_id: str, holders: Dict[str, int]):
        """Save all holder balances for a token."""
        with self._lock:
            for holder_hex, balance in holders.items():
                self._conn.execute("""
                    INSERT OR REPLACE INTO rwa_holders
                    (token_id, holder_hex, balance, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (token_id, holder_hex, str(balance), time.time()))
            self._safe_commit()

    def get_holders(self, token_id: str) -> List[dict]:
        """Get all holders for a token."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT holder_hex, balance FROM rwa_holders "
                "WHERE token_id = ? AND CAST(balance AS INTEGER) > 0 "
                "ORDER BY CAST(balance AS INTEGER) DESC",
                (token_id,),
            )
            return [
                {"address": row[0], "balance": int(row[1])}
                for row in cursor.fetchall()
            ]

    # ---- KYC Persistence ----

    def save_kyc(self, kyc_data: dict):
        """Save KYC record."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO rwa_kyc
                (address_hex, kyc_level, jurisdiction, credential_id,
                 verified_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                kyc_data.get("address_hex", ""),
                kyc_data.get("kyc_level", 0),
                kyc_data.get("jurisdiction", ""),
                kyc_data.get("credential_id", ""),
                kyc_data.get("verified_at", 0),
                kyc_data.get("expires_at", 0),
            ))
            self._safe_commit()

    def get_kyc(self, address_hex: str) -> Optional[dict]:
        """Get KYC record by address."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM rwa_kyc WHERE address_hex = ?",
                (address_hex,),
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None

    # ---- Dividend Persistence ----

    def save_dividend(self, dividend_data: dict):
        """Save dividend distribution record."""
        with self._lock:
            payouts = dividend_data.get("payouts", {})
            # Convert bytes keys to hex if needed
            payouts_serializable = {}
            for k, v in payouts.items():
                key = k.hex() if isinstance(k, bytes) else str(k)
                payouts_serializable[key] = v

            self._conn.execute("""
                INSERT OR REPLACE INTO rwa_dividends
                (dividend_id, token_id, total_amount, amount_per_token,
                 holder_count, issuer_hex, distributed_at, payouts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dividend_data.get("dividend_id", ""),
                dividend_data.get("token_id", ""),
                str(dividend_data.get("total_amount", 0)),
                str(dividend_data.get("amount_per_token", 0)),
                dividend_data.get("holder_count", 0),
                dividend_data.get("issuer_hex", ""),
                dividend_data.get("distributed_at", 0),
                json.dumps(payouts_serializable),
            ))
            self._safe_commit()

    def get_token_dividends(self, token_id: str, limit: int = 20) -> List[dict]:
        """Get dividend history for a token."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM rwa_dividends WHERE token_id = ? "
                "ORDER BY distributed_at DESC LIMIT ?",
                (token_id, limit),
            )
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                d = dict(zip(columns, row))
                d["payouts"] = json.loads(d.pop("payouts_json", "{}"))
                results.append(d)
            return results

    # ---- Stats ----

    def get_stats(self) -> dict:
        with self._lock:
            tokens_count = self._conn.execute(
                "SELECT COUNT(*) FROM rwa_tokens"
            ).fetchone()[0]
            active_count = self._conn.execute(
                "SELECT COUNT(*) FROM rwa_tokens WHERE status = 6"
            ).fetchone()[0]
            kyc_count = self._conn.execute(
                "SELECT COUNT(*) FROM rwa_kyc"
            ).fetchone()[0]
            dividends_count = self._conn.execute(
                "SELECT COUNT(*) FROM rwa_dividends"
            ).fetchone()[0]
            total_valuation = self._conn.execute(
                "SELECT COALESCE(SUM(CAST(valuation AS INTEGER)), 0) "
                "FROM rwa_tokens WHERE status = 6"
            ).fetchone()[0]
            return {
                "tokens_stored": tokens_count,
                "active_tokens": active_count,
                "kyc_records": kyc_count,
                "dividends_stored": dividends_count,
                "total_valuation": total_valuation,
                "db_path": self._active_path,
                "encrypted": self.is_encrypted,
            }

    def encrypt_at_rest(self) -> None:
        if not self._encryption_password:
            return
        if self._in_memory:
            DatabaseEncryption.save_from_memory_connection(
                self._conn,
                self.db_path,
                self._encryption_password,
            )
            return
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.debug("WAL checkpoint failed during encrypt_at_rest: %s", e)
        self.close()
        if self._active_path != self.db_path:
            import shutil, os
            shutil.copy2(self._active_path, self.db_path)
            try:
                os.remove(self._active_path)
            except OSError as e:
                logger.debug("Could not remove decrypted temp file: %s", e)
            self._active_path = self.db_path
        DatabaseEncryption.encrypt_file(self.db_path, self._encryption_password)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
