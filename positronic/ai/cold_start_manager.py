"""
Positronic - Cold Start Manager (Phase 32)

Solves the AI cold start problem in new networks where unusual-but-legitimate
transactions all look anomalous to freshly trained models.

Three-phase calibration system based on block height:
  Phase A (blocks 0 to 100,000):       LEARNING MODE
    - Score every transaction but quarantine none
    - Kill switch always disabled
    - Collect false positive data for calibration

  Phase B (blocks 100,001 to 500,000): CALIBRATION MODE
    - Thresholds tighten linearly toward production values
    - Kill switch triggers at 0.5% FP rate (10x stricter)
    - Quarantine enabled with 2x review timeout

  Phase C (blocks 500,001+):           PRODUCTION MODE
    - Full thresholds: accept_q=8500, quarantine_q=9500
    - Kill switch triggers at 5% FP rate
    - Online learning enabled
"""

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

from positronic.utils.logging import get_logger
from positronic.constants import (
    COLD_START_PHASE_A_END,
    COLD_START_PHASE_B_END,
    COLD_START_PHASE_A_ACCEPT_Q,
    COLD_START_PHASE_A_QUARANTINE_Q,
    COLD_START_PHASE_B_START_ACCEPT_Q,
    COLD_START_PHASE_B_START_QUARANTINE_Q,
    COLD_START_PHASE_B_KS_FP_RATE,
    COLD_START_FP_RATE_LIMIT_PER_HOUR,
    AI_ACCEPT_THRESHOLD_Q,
    AI_QUARANTINE_THRESHOLD_Q,
    AI_KILL_SWITCH_FP_RATE,
)

logger = get_logger(__name__)


class ColdStartManager:
    """
    Manages AI validation thresholds during network cold start.

    Provides graduated threshold relaxation so that new networks do not
    trigger kill switches on unusual-but-legitimate transactions before
    the AI models have enough training data to calibrate properly.
    """

    # Phase boundaries (block heights)
    PHASE_A_END = COLD_START_PHASE_A_END      # 100,000
    PHASE_B_END = COLD_START_PHASE_B_END      # 500,000

    def __init__(self, db_path: str = None):
        """Initialize with optional DB path for persistence.

        Args:
            db_path: Path to SQLite database file. If None, state is
                     held in memory only (no persistence).
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

        # False positive tracking
        self._fp_reports: List[dict] = []
        self._total_scored: int = 0

        # Rate limiting: {reporter_hex: [timestamp, ...]}
        self._rate_limits: Dict[str, List[float]] = {}

        # Initialize database if path provided
        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database with WAL mode."""
        os.makedirs(
            os.path.dirname(self._db_path) if os.path.dirname(self._db_path) else ".",
            exist_ok=True,
        )
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cold_start_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.commit()

    def get_current_phase(self, block_height: int) -> str:
        """Returns 'A', 'B', or 'C' based on block height.

        Args:
            block_height: Current block height.

        Returns:
            Phase identifier string.
        """
        if block_height <= self.PHASE_A_END:
            return "A"
        if block_height <= self.PHASE_B_END:
            return "B"
        return "C"

    def get_thresholds(self, block_height: int) -> Tuple[int, int]:
        """Returns (accept_bp, quarantine_bp) for given block height.

        Phase A: fully relaxed (accept all, quarantine none)
        Phase B: linear interpolation from lenient to production
        Phase C: full production thresholds

        Args:
            block_height: Current block height.

        Returns:
            Tuple of (accept_q, quarantine_q) in integer basis points.
        """
        phase = self.get_current_phase(block_height)

        if phase == "A":
            return (COLD_START_PHASE_A_ACCEPT_Q, COLD_START_PHASE_A_QUARANTINE_Q)

        if phase == "C":
            return (AI_ACCEPT_THRESHOLD_Q, AI_QUARANTINE_THRESHOLD_Q)

        # Phase B: linear interpolation
        # progress = 0.0 at block 100,001 ... 1.0 at block 500,000
        phase_b_start = self.PHASE_A_END + 1      # 100,001
        phase_b_range = self.PHASE_B_END - phase_b_start  # 399,999
        progress = (block_height - phase_b_start) / phase_b_range

        accept_q = int(
            COLD_START_PHASE_B_START_ACCEPT_Q
            + (AI_ACCEPT_THRESHOLD_Q - COLD_START_PHASE_B_START_ACCEPT_Q) * progress
        )
        quarantine_q = int(
            COLD_START_PHASE_B_START_QUARANTINE_Q
            + (AI_QUARANTINE_THRESHOLD_Q - COLD_START_PHASE_B_START_QUARANTINE_Q) * progress
        )

        return (accept_q, quarantine_q)

    def record_false_positive(
        self, tx_hash: bytes, model_name: str, reporter: bytes
    ) -> bool:
        """Record a false positive report. Rate limited per reporter.

        Args:
            tx_hash: Hash of the falsely flagged transaction.
            model_name: Name of the model that produced the false positive.
            reporter: Address of the reporter (20 bytes).

        Returns:
            True if recorded, False if rate limited.
        """
        now = time.time()
        reporter_hex = reporter.hex()

        # Clean up expired timestamps and check rate limit
        if reporter_hex in self._rate_limits:
            # Remove timestamps older than 1 hour
            cutoff = now - 3600
            self._rate_limits[reporter_hex] = [
                t for t in self._rate_limits[reporter_hex] if t > cutoff
            ]

            # Check rate limit
            if len(self._rate_limits[reporter_hex]) >= COLD_START_FP_RATE_LIMIT_PER_HOUR:
                logger.debug(
                    "FP report rate-limited for reporter %s (max %d/hour)",
                    reporter_hex[:16],
                    COLD_START_FP_RATE_LIMIT_PER_HOUR,
                )
                return False

        # Record the report
        self._fp_reports.append({
            "tx_hash": tx_hash.hex(),
            "model_name": model_name,
            "reporter": reporter_hex,
            "timestamp": now,
        })

        # Update rate limit tracker
        if reporter_hex not in self._rate_limits:
            self._rate_limits[reporter_hex] = []
        self._rate_limits[reporter_hex].append(now)

        logger.debug(
            "FP report recorded: tx=%s model=%s reporter=%s",
            tx_hash.hex()[:16],
            model_name,
            reporter_hex[:16],
        )
        return True

    def should_trigger_kill_switch(self, block_height: int) -> Tuple[bool, str]:
        """Check whether the kill switch should trigger at this block height.

        Phase A: always disabled
        Phase B: triggers at 0.5% FP rate
        Phase C: triggers at 5% FP rate

        Args:
            block_height: Current block height.

        Returns:
            Tuple of (should_trigger, reason_string).
        """
        phase = self.get_current_phase(block_height)

        if phase == "A":
            return (False, "Phase A: kill switch disabled")

        fp_rate = self.get_fp_rate()

        if phase == "B":
            if fp_rate > COLD_START_PHASE_B_KS_FP_RATE:
                return (
                    True,
                    f"Phase B: FP rate {fp_rate:.4f} exceeds 0.5% threshold",
                )
            return (
                False,
                f"Phase B: FP rate {fp_rate:.4f} within 0.5% threshold",
            )

        # Phase C
        if fp_rate > AI_KILL_SWITCH_FP_RATE:
            return (
                True,
                f"Phase C: FP rate {fp_rate:.4f} exceeds 5% threshold",
            )
        return (
            False,
            f"Phase C: FP rate {fp_rate:.4f} within 5% threshold",
        )

    def get_fp_rate(self) -> float:
        """Current false positive rate.

        Returns:
            FP rate as a float (0.0 to 1.0). Returns 0.0 if no
            transactions have been scored.
        """
        if self._total_scored == 0:
            return 0.0
        return len(self._fp_reports) / self._total_scored

    def export_training_data(self) -> list:
        """Export collected false positive data for model retraining.

        Returns:
            List of dicts containing FP report data.
        """
        return list(self._fp_reports)

    def get_status(self, block_height: int) -> dict:
        """Returns dict suitable for RPC response with full status.

        Args:
            block_height: Current block height.

        Returns:
            Dictionary with phase, thresholds, FP stats, and kill switch state.
        """
        phase = self.get_current_phase(block_height)
        accept_q, quarantine_q = self.get_thresholds(block_height)

        # Calculate blocks to next phase
        if phase == "A":
            blocks_to_next = (self.PHASE_A_END + 1) - block_height
        elif phase == "B":
            blocks_to_next = (self.PHASE_B_END + 1) - block_height
        else:
            blocks_to_next = 0

        # Kill switch enabled in phases B and C
        kill_switch_enabled = phase != "A"

        return {
            "phase": phase,
            "block_height": block_height,
            "accept_threshold": accept_q,
            "quarantine_threshold": quarantine_q,
            "fp_count": len(self._fp_reports),
            "fp_rate": self.get_fp_rate(),
            "kill_switch_enabled": kill_switch_enabled,
            "blocks_to_next_phase": blocks_to_next,
        }

    def save_state(self):
        """Persist state to SQLite."""
        if self._conn is None:
            return

        state = {
            "fp_reports": self._fp_reports,
            "total_scored": self._total_scored,
        }

        self._conn.execute(
            "INSERT OR REPLACE INTO cold_start_state (key, value) VALUES (?, ?)",
            ("state", json.dumps(state)),
        )
        self._conn.commit()
        logger.debug("Cold start state saved (%d FP reports)", len(self._fp_reports))

    def load_state(self):
        """Load state from SQLite."""
        if self._conn is None:
            return

        row = self._conn.execute(
            "SELECT value FROM cold_start_state WHERE key = ?",
            ("state",),
        ).fetchone()

        if row is None:
            return

        state = json.loads(row[0])
        self._fp_reports = state.get("fp_reports", [])
        self._total_scored = state.get("total_scored", 0)
        logger.debug(
            "Cold start state loaded (%d FP reports, %d scored)",
            len(self._fp_reports),
            self._total_scored,
        )
