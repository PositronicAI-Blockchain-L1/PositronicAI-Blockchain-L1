"""
Positronic - Game Data Persistence Layer
SQLite storage for game registry, sessions, mining stats, and token mappings.
Ensures game data survives node restarts.
Supports transparent AES-256-GCM encryption at rest.
"""

import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Tuple

from positronic.utils.logging import get_logger
from positronic.storage.database import DatabaseEncryption

logger = get_logger("game_db")

# Schema version for future migrations
GAME_DB_VERSION = 1


class GameDatabase:
    """
    SQLite persistence for all game-related data.
    Supports optional AES-256-GCM encryption at rest.

    Tables:
    - games: Registered game info (from GameRegistry)
    - game_sessions: Session history (from SessionValidator)
    - game_emissions: Daily emission tracking (from MiningRateController)
    - game_tokens: Game→token mappings (from GameTokenBridge)
    - game_collections: Game→NFT collection mappings
    - game_meta: Key-value metadata store
    """

    def __init__(
        self,
        db_path: str = "game_data.db",
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
            CREATE TABLE IF NOT EXISTS game_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                developer_hex TEXT NOT NULL,
                game_type INTEGER NOT NULL DEFAULT 3,
                status INTEGER NOT NULL DEFAULT 0,
                description TEXT DEFAULT '',
                api_key_hash TEXT DEFAULT '',
                daily_emission_cap INTEGER DEFAULT 0,
                reward_multiplier REAL DEFAULT 1.0,
                player_daily_cap INTEGER DEFAULT 0,
                total_emitted INTEGER DEFAULT 0,
                total_sessions INTEGER DEFAULT 0,
                active_players INTEGER DEFAULT 0,
                total_players INTEGER DEFAULT 0,
                cheat_detections INTEGER DEFAULT 0,
                trust_score INTEGER DEFAULT 100,
                ai_risk_score REAL DEFAULT 0.0,
                registered_at REAL DEFAULT 0,
                approved_at REAL DEFAULT 0,
                daily_emitted INTEGER DEFAULT 0,
                daily_reset_time REAL DEFAULT 0,
                extra_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                player_hex TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                started_at REAL NOT NULL,
                ended_at REAL DEFAULT 0,
                duration REAL DEFAULT 0,
                reward INTEGER DEFAULT 0,
                ai_cheat_score REAL DEFAULT 0,
                metrics_json TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS game_emissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                player_hex TEXT NOT NULL,
                amount INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS game_token_mappings (
                game_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                created_at REAL DEFAULT 0,
                PRIMARY KEY (game_id, token_id)
            );

            CREATE TABLE IF NOT EXISTS game_collection_mappings (
                game_id TEXT NOT NULL,
                collection_id TEXT NOT NULL,
                created_at REAL DEFAULT 0,
                PRIMARY KEY (game_id, collection_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_game
                ON game_sessions(game_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_player
                ON game_sessions(player_hex);
            CREATE INDEX IF NOT EXISTS idx_emissions_game
                ON game_emissions(game_id);
            CREATE INDEX IF NOT EXISTS idx_emissions_player
                ON game_emissions(player_hex);
        """)

        # Set schema version
        self._conn.execute(
            "INSERT OR IGNORE INTO game_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(GAME_DB_VERSION)),
        )
        self._safe_commit()
        logger.info("game_db_initialized", extra={"path": self._active_path})

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
                raise DiskFullError(f"Game DB disk full: {e}") from e
            raise

    @property
    def is_encrypted(self) -> bool:
        """Check if encryption is enabled."""
        return self._encryption_password is not None

    # ---- Game Registry Persistence ----

    def save_game(self, game_data: dict):
        """Save or update a registered game."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO games (
                    game_id, name, developer_hex, game_type, status,
                    description, api_key_hash, daily_emission_cap,
                    reward_multiplier, player_daily_cap, total_emitted,
                    total_sessions, active_players, total_players,
                    cheat_detections, trust_score, ai_risk_score,
                    registered_at, approved_at, daily_emitted, daily_reset_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_data.get("game_id", ""),
                game_data.get("name", ""),
                game_data.get("developer", ""),
                game_data.get("game_type", 3),
                game_data.get("status", 0),
                game_data.get("description", ""),
                game_data.get("api_key_hash", ""),
                game_data.get("daily_emission_cap", 0),
                game_data.get("reward_multiplier", 1.0),
                game_data.get("player_daily_cap", 0),
                game_data.get("total_emitted", 0),
                game_data.get("total_sessions", 0),
                game_data.get("active_players", 0),
                game_data.get("total_players", 0),
                game_data.get("cheat_detections", 0),
                game_data.get("trust_score", 100),
                game_data.get("ai_risk_score", 0.0),
                game_data.get("registered_at", 0),
                game_data.get("approved_at", 0),
                game_data.get("daily_emitted", 0),
                game_data.get("daily_reset_time", 0),
            ))
            self._safe_commit()

    def load_all_games(self) -> List[dict]:
        """Load all registered games from database."""
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM games")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def get_game(self, game_id: str) -> Optional[dict]:
        """Load a single game by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM games WHERE game_id = ?", (game_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None

    # ---- Session Persistence ----

    def save_session(self, session_data: dict):
        """Save a completed game session."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO game_sessions (
                    session_id, game_id, player_hex, status,
                    started_at, ended_at, duration, reward,
                    ai_cheat_score, metrics_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_data.get("session_id", ""),
                session_data.get("game_id", ""),
                session_data.get("player_hex", ""),
                session_data.get("status", "ACTIVE"),
                session_data.get("started_at", 0),
                session_data.get("ended_at", 0),
                session_data.get("duration", 0),
                session_data.get("reward", 0),
                session_data.get("ai_cheat_score", 0),
                json.dumps(session_data.get("metrics", {})),
                time.time(),
            ))
            self._safe_commit()

    def get_player_sessions(
        self, player_hex: str, limit: int = 20
    ) -> List[dict]:
        """Get recent sessions for a player."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM game_sessions WHERE player_hex = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (player_hex, limit),
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_game_sessions(
        self, game_id: str, limit: int = 50
    ) -> List[dict]:
        """Get recent sessions for a game."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM game_sessions WHERE game_id = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (game_id, limit),
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ---- Emission Recording ----

    def record_emission(
        self, game_id: str, player_hex: str, amount: int
    ):
        """Record a token emission event."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO game_emissions (game_id, player_hex, amount, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (game_id, player_hex, amount, time.time()),
            )
            self._safe_commit()

    def get_total_emissions(self) -> int:
        """Get total emissions across all games."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM game_emissions"
            ).fetchone()
            return row[0]

    # ---- Token/Collection Mapping Persistence ----

    def save_token_mapping(self, game_id: str, token_id: str):
        """Save game→token mapping."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO game_token_mappings "
                "(game_id, token_id, created_at) VALUES (?, ?, ?)",
                (game_id, token_id, time.time()),
            )
            self._safe_commit()

    def save_collection_mapping(self, game_id: str, collection_id: str):
        """Save game→collection mapping."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO game_collection_mappings "
                "(game_id, collection_id, created_at) VALUES (?, ?, ?)",
                (game_id, collection_id, time.time()),
            )
            self._safe_commit()

    def get_game_token_ids(self, game_id: str) -> List[str]:
        """Get all token IDs for a game."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT token_id FROM game_token_mappings WHERE game_id = ?",
                (game_id,),
            ).fetchall()
            return [r[0] for r in rows]

    def get_game_collection_ids(self, game_id: str) -> List[str]:
        """Get all collection IDs for a game."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT collection_id FROM game_collection_mappings "
                "WHERE game_id = ?",
                (game_id,),
            ).fetchall()
            return [r[0] for r in rows]

    # ---- Stats ----

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._lock:
            games = self._conn.execute(
                "SELECT COUNT(*) FROM games"
            ).fetchone()[0]
            sessions = self._conn.execute(
                "SELECT COUNT(*) FROM game_sessions"
            ).fetchone()[0]
            emissions = self._conn.execute(
                "SELECT COUNT(*) FROM game_emissions"
            ).fetchone()[0]
            total_emitted = self._conn.execute(
                "SELECT COALESCE(SUM(amount), 0) FROM game_emissions"
            ).fetchone()[0]
            return {
                "games_stored": games,
                "sessions_stored": sessions,
                "emission_records": emissions,
                "total_emitted_recorded": total_emitted,
                "db_path": self._active_path,
                "encrypted": self.is_encrypted,
            }

    def encrypt_at_rest(self) -> None:
        """Re-encrypt the database file on disk (call on shutdown)."""
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
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
