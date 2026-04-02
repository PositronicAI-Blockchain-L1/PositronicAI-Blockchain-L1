"""
Positronic - AI Agent Marketplace Persistence Layer (Phase 29)
SQLite storage for agent registry, tasks, ratings, and stats.
Supports transparent AES-256-GCM encryption at rest.
"""

import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from positronic.utils.logging import get_logger
from positronic.storage.database import DatabaseEncryption

logger = get_logger("agent_db")

AGENT_DB_VERSION = 1


class AgentDatabase:
    """
    SQLite persistence for all AI agent marketplace data.
    Supports optional AES-256-GCM encryption at rest.

    Tables:
    - agents: Registered agent info
    - agent_tasks: Task submission and result history
    - agent_ratings: User ratings for agents
    - agent_meta: Key-value metadata store
    """

    def __init__(
        self,
        db_path: str = "agent_data.db",
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
            CREATE TABLE IF NOT EXISTS agent_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                owner_hex TEXT NOT NULL,
                category INTEGER NOT NULL DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                description TEXT DEFAULT '',
                endpoint_url TEXT DEFAULT '',
                model_hash TEXT DEFAULT '',
                task_fee INTEGER DEFAULT 0,
                api_key_hash TEXT DEFAULT '',
                quality_score INTEGER DEFAULT 7500,
                trust_score INTEGER DEFAULT 100,
                ai_risk_score REAL DEFAULT 0.0,
                tasks_completed INTEGER DEFAULT 0,
                tasks_failed INTEGER DEFAULT 0,
                total_earned INTEGER DEFAULT 0,
                total_ratings INTEGER DEFAULT 0,
                rating_sum INTEGER DEFAULT 0,
                registered_at REAL DEFAULT 0,
                approved_at REAL DEFAULT 0,
                last_task_at REAL DEFAULT 0,
                extra_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS agent_tasks (
                task_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                requester_hex TEXT NOT NULL,
                input_data TEXT DEFAULT '',
                fee_paid INTEGER DEFAULT 0,
                status INTEGER NOT NULL DEFAULT 0,
                result_data TEXT DEFAULT '',
                result_hash TEXT DEFAULT '',
                ai_quality_score INTEGER DEFAULT 0,
                agent_reward INTEGER DEFAULT 0,
                platform_fee INTEGER DEFAULT 0,
                burn_amount INTEGER DEFAULT 0,
                submitted_at REAL DEFAULT 0,
                assigned_at REAL DEFAULT 0,
                completed_at REAL DEFAULT 0,
                timeout_at REAL DEFAULT 0,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS agent_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                rater_hex TEXT NOT NULL,
                score INTEGER NOT NULL,
                comment TEXT DEFAULT '',
                created_at REAL DEFAULT 0,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent
                ON agent_tasks(agent_id);
            CREATE INDEX IF NOT EXISTS idx_agent_tasks_requester
                ON agent_tasks(requester_hex);
            CREATE INDEX IF NOT EXISTS idx_agent_ratings_agent
                ON agent_ratings(agent_id);
        """)

        self._conn.execute(
            "INSERT OR IGNORE INTO agent_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(AGENT_DB_VERSION)),
        )
        self._safe_commit()
        logger.info("agent_db_initialized", extra={"path": self._active_path})

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
                raise DiskFullError(f"Agent DB disk full: {e}") from e
            raise

    @property
    def is_encrypted(self) -> bool:
        return self._encryption_password is not None

    # ---- Agent Persistence ----

    def save_agent(self, agent_data: dict):
        """Save or update an agent record."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO agents (
                    agent_id, name, owner_hex, category, status,
                    description, endpoint_url, model_hash, task_fee,
                    api_key_hash, quality_score, trust_score, ai_risk_score,
                    tasks_completed, tasks_failed, total_earned,
                    total_ratings, rating_sum,
                    registered_at, approved_at, last_task_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                agent_data.get("agent_id", ""),
                agent_data.get("name", ""),
                agent_data.get("owner_hex", ""),
                agent_data.get("category", 0),
                agent_data.get("status", 0),
                agent_data.get("description", ""),
                agent_data.get("endpoint_url", ""),
                agent_data.get("model_hash", ""),
                agent_data.get("task_fee", 0),
                agent_data.get("api_key_hash", ""),
                agent_data.get("quality_score", 7500),
                agent_data.get("trust_score", 100),
                agent_data.get("ai_risk_score", 0.0),
                agent_data.get("tasks_completed", 0),
                agent_data.get("tasks_failed", 0),
                agent_data.get("total_earned", 0),
                agent_data.get("total_ratings", 0),
                agent_data.get("rating_sum", 0),
                agent_data.get("registered_at", 0),
                agent_data.get("approved_at", 0),
                agent_data.get("last_task_at", 0),
            ))
            self._safe_commit()

    def load_all_agents(self) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM agents")
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_agent(self, agent_id: str) -> Optional[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agents WHERE agent_id = ?", (agent_id,)
            )
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            return dict(zip(columns, row)) if row else None

    # ---- Task Persistence ----

    def save_task(self, task_data: dict):
        """Save or update a task record."""
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO agent_tasks (
                    task_id, agent_id, requester_hex, input_data, fee_paid,
                    status, result_data, result_hash, ai_quality_score,
                    agent_reward, platform_fee, burn_amount,
                    submitted_at, assigned_at, completed_at, timeout_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                task_data.get("task_id", ""),
                task_data.get("agent_id", ""),
                task_data.get("requester_hex", ""),
                task_data.get("input_data", ""),
                task_data.get("fee_paid", 0),
                task_data.get("status", 0),
                task_data.get("result_data", ""),
                task_data.get("result_hash", ""),
                task_data.get("ai_quality_score", 0),
                task_data.get("agent_reward", 0),
                task_data.get("platform_fee", 0),
                task_data.get("burn_amount", 0),
                task_data.get("submitted_at", 0),
                task_data.get("assigned_at", 0),
                task_data.get("completed_at", 0),
                task_data.get("timeout_at", 0),
            ))
            self._safe_commit()

    def get_agent_tasks(self, agent_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agent_tasks WHERE agent_id = ? "
                "ORDER BY submitted_at DESC LIMIT ?",
                (agent_id, limit),
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ---- Rating Persistence ----

    def save_rating(self, agent_id: str, rater_hex: str, score: int, comment: str = ""):
        with self._lock:
            self._conn.execute(
                "INSERT INTO agent_ratings (agent_id, rater_hex, score, comment, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (agent_id, rater_hex, score, comment, time.time()),
            )
            self._safe_commit()

    def get_agent_ratings(self, agent_id: str, limit: int = 20) -> List[dict]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agent_ratings WHERE agent_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (agent_id, limit),
            )
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    # ---- Stats ----

    def get_stats(self) -> dict:
        with self._lock:
            agents_count = self._conn.execute(
                "SELECT COUNT(*) FROM agents"
            ).fetchone()[0]
            tasks_count = self._conn.execute(
                "SELECT COUNT(*) FROM agent_tasks"
            ).fetchone()[0]
            completed = self._conn.execute(
                "SELECT COUNT(*) FROM agent_tasks WHERE status = 3"
            ).fetchone()[0]
            total_earned = self._conn.execute(
                "SELECT COALESCE(SUM(agent_reward), 0) FROM agent_tasks WHERE status = 3"
            ).fetchone()[0]
            total_burned = self._conn.execute(
                "SELECT COALESCE(SUM(burn_amount), 0) FROM agent_tasks WHERE status = 3"
            ).fetchone()[0]
            return {
                "agents_stored": agents_count,
                "tasks_stored": tasks_count,
                "tasks_completed": completed,
                "total_earned": total_earned,
                "total_burned": total_burned,
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
