"""
Positronic - Quarantine Pool Manager
Manages transactions that are neither accepted nor rejected.
Quarantined TXs are reviewed periodically and can be appealed.

Persistence: QuarantinePool can optionally save/load from the SQLite database
so quarantined transactions survive node restarts.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from positronic.core.transaction import Transaction, TxStatus
from positronic.constants import QUARANTINE_REVIEW_INTERVAL, MAX_QUARANTINE_TIME

logger = logging.getLogger("positronic.ai.quarantine")


@dataclass
class QuarantineEntry:
    """A quarantined transaction entry."""
    tx: Transaction
    ai_score: float
    quarantined_at_block: int
    quarantined_at_time: float
    review_count: int = 0
    appeal_votes_for: int = 0
    appeal_votes_against: int = 0
    appeal_voters: set = field(default_factory=set)  # Track voters for dedup
    status: str = "quarantined"  # quarantined, released, expired, rejected


class QuarantinePool:
    """
    Manages quarantined transactions.

    Rules:
    - Quarantined TXs are reviewed every QUARANTINE_REVIEW_INTERVAL blocks
    - If AI score improves on re-evaluation, TX is released
    - Users can appeal via governance voting
    - TXs expire after MAX_QUARANTINE_TIME blocks

    Persistence:
    - Call save_to_db() to persist current state to SQLite
    - Call load_from_db() on startup to restore quarantined TXs
    - Both methods are safe to call without a database (no-op)
    """

    def __init__(self, db=None):
        self.entries: Dict[bytes, QuarantineEntry] = {}  # tx_hash -> entry
        self.released_count: int = 0
        self.expired_count: int = 0
        self.appeal_count: int = 0
        self._db = db  # Optional Database instance for persistence

    def set_db(self, db):
        """Set the database for persistence."""
        self._db = db

    def add(
        self,
        tx: Transaction,
        ai_score: float,
        current_block: int,
    ):
        """Add a transaction to quarantine."""
        entry = QuarantineEntry(
            tx=tx,
            ai_score=ai_score,
            quarantined_at_block=current_block,
            quarantined_at_time=time.time(),
        )
        self.entries[tx.tx_hash] = entry

        # Persist to DB
        self._persist_entry(entry)

    def remove(self, tx_hash: bytes) -> Optional[Transaction]:
        """Remove a transaction from quarantine."""
        entry = self.entries.pop(tx_hash, None)
        if entry:
            self._remove_from_db(tx_hash)
            return entry.tx
        return None

    def get(self, tx_hash: bytes) -> Optional[QuarantineEntry]:
        """Get a quarantine entry."""
        return self.entries.get(tx_hash)

    def review(
        self,
        current_block: int,
        re_evaluate_fn=None,
    ) -> Tuple[List[Transaction], List[Transaction]]:
        """
        Review quarantined transactions.
        Returns (released_txs, expired_txs).

        re_evaluate_fn: Optional callback that takes a Transaction
        and returns a new AI score.
        """
        released = []
        expired = []
        to_remove = []

        for tx_hash, entry in self.entries.items():
            blocks_in_quarantine = current_block - entry.quarantined_at_block

            # Check if expired
            if blocks_in_quarantine >= MAX_QUARANTINE_TIME:
                entry.status = "expired"
                expired.append(entry.tx)
                to_remove.append(tx_hash)
                self.expired_count += 1
                continue

            # Check if due for review
            if blocks_in_quarantine % QUARANTINE_REVIEW_INTERVAL != 0:
                continue

            entry.review_count += 1

            # Re-evaluate with AI if callback provided
            if re_evaluate_fn:
                new_score = re_evaluate_fn(entry.tx)
                if new_score < 0.85:
                    # Score improved, release
                    entry.status = "released"
                    entry.tx.ai_score = new_score
                    entry.tx.status = TxStatus.ACCEPTED
                    released.append(entry.tx)
                    to_remove.append(tx_hash)
                    self.released_count += 1
                else:
                    entry.ai_score = new_score

            # Check appeal votes
            if entry.appeal_votes_for > entry.appeal_votes_against * 2:
                # Supermajority appeal, release
                entry.status = "released"
                entry.tx.status = TxStatus.ACCEPTED
                released.append(entry.tx)
                to_remove.append(tx_hash)
                self.released_count += 1
                self.appeal_count += 1

        # Clean up
        for tx_hash in to_remove:
            self.entries.pop(tx_hash, None)
            self._remove_from_db(tx_hash)

        return released, expired

    def vote_appeal(self, tx_hash: bytes, vote_for: bool,
                    voter_id: bytes = None) -> bool:
        """Vote on a quarantine appeal.

        Security fix: voter_id enables deduplication — each voter can only
        vote once per transaction. If voter_id is None (legacy), dedup is
        skipped for backward compatibility.
        """
        entry = self.entries.get(tx_hash)
        if not entry:
            return False

        # Voter deduplication
        if voter_id is not None:
            if voter_id in entry.appeal_voters:
                return False  # Already voted
            entry.appeal_voters.add(voter_id)

        if vote_for:
            entry.appeal_votes_for += 1
        else:
            entry.appeal_votes_against += 1

        # Update in DB
        self._update_entry_in_db(entry)
        return True

    @property
    def size(self) -> int:
        """Number of currently quarantined transactions."""
        return len(self.entries)

    def get_all_quarantined(self) -> List[QuarantineEntry]:
        """Get all currently quarantined transactions."""
        return list(self.entries.values())

    # === Persistence Methods ===

    def save_to_db(self):
        """Save all quarantine entries to the database."""
        if not self._db:
            return
        try:
            for tx_hash, entry in self.entries.items():
                self._persist_entry(entry)
            self._db.commit()
            logger.debug(f"Saved {len(self.entries)} quarantine entries to DB")
        except Exception as e:
            logger.warning(f"Failed to save quarantine pool: {e}")

    def load_from_db(self):
        """Load quarantine entries from the database on startup."""
        if not self._db:
            return
        try:
            rows = self._db.execute(
                "SELECT tx_json, ai_score, quarantined_at_block, review_count, status "
                "FROM quarantine_pool WHERE status = 'quarantined'"
            ).fetchall()

            loaded = 0
            for row in rows:
                try:
                    tx_data = json.loads(row[0] if isinstance(row, tuple) else row["tx_json"])
                    tx = Transaction.from_dict(tx_data)
                    ai_score = row[1] if isinstance(row, tuple) else row["ai_score"]
                    block = row[2] if isinstance(row, tuple) else row["quarantined_at_block"]
                    review = row[3] if isinstance(row, tuple) else row["review_count"]

                    entry = QuarantineEntry(
                        tx=tx,
                        ai_score=ai_score,
                        quarantined_at_block=block,
                        quarantined_at_time=time.time(),  # Reset timer
                        review_count=review,
                    )
                    self.entries[tx.tx_hash] = entry
                    loaded += 1
                except Exception as e:
                    logger.debug(f"Failed to load quarantine entry: {e}")

            if loaded > 0:
                logger.info(f"Loaded {loaded} quarantine entries from DB")

        except Exception as e:
            logger.warning(f"Failed to load quarantine pool: {e}")

    def _persist_entry(self, entry: QuarantineEntry):
        """Persist a single entry to the database."""
        if not self._db:
            return
        try:
            tx_json = json.dumps(entry.tx.to_dict())
            tx_hash_hex = entry.tx.tx_hash.hex()
            self._db.execute(
                "INSERT OR REPLACE INTO quarantine_pool "
                "(tx_hash, tx_json, ai_score, quarantined_at_block, review_count, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tx_hash_hex, tx_json, entry.ai_score,
                 entry.quarantined_at_block, entry.review_count, entry.status),
            )
            self._db.commit()
        except Exception as e:
            logger.debug(f"Failed to persist quarantine entry: {e}")

    def _update_entry_in_db(self, entry: QuarantineEntry):
        """Update an existing entry in the database."""
        if not self._db:
            return
        try:
            tx_hash_hex = entry.tx.tx_hash.hex()
            self._db.execute(
                "UPDATE quarantine_pool SET review_count=?, status=?, ai_score=? "
                "WHERE tx_hash=?",
                (entry.review_count, entry.status, entry.ai_score, tx_hash_hex),
            )
            self._db.commit()
        except Exception as e:
            logger.debug(f"Failed to update quarantine entry: {e}")

    def _remove_from_db(self, tx_hash: bytes):
        """Remove an entry from the database."""
        if not self._db:
            return
        try:
            self._db.execute(
                "DELETE FROM quarantine_pool WHERE tx_hash=?",
                (tx_hash.hex(),),
            )
            self._db.commit()
        except Exception as e:
            logger.debug(f"Failed to remove quarantine entry from DB: {e}")

    def get_stats(self) -> dict:
        return {
            "pool_size": self.size,
            "released_count": self.released_count,
            "expired_count": self.expired_count,
            "appeal_count": self.appeal_count,
            "avg_score": (
                sum(e.ai_score for e in self.entries.values()) / max(self.size, 1)
            ),
            "persistent": self._db is not None,
        }
