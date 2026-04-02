"""
Positronic - Database Archiver & Pruner

Prevents unbounded growth of blocks, transactions, and quarantine tables.
Keeps the most recent N blocks and prunes older data.

Usage::

    pruner = DatabasePruner(database, keep_blocks=10_000)
    stats = pruner.prune()
    # stats => {"blocks_pruned": 500, "transactions_pruned": 12300, ...}
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("positronic.storage.archiver")


@dataclass
class PruneStats:
    """Statistics from a single prune run."""
    blocks_pruned: int = 0
    transactions_pruned: int = 0
    quarantine_pruned: int = 0
    treasury_pruned: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "blocks_pruned": self.blocks_pruned,
            "transactions_pruned": self.transactions_pruned,
            "quarantine_pruned": self.quarantine_pruned,
            "treasury_pruned": self.treasury_pruned,
            "duration_ms": round(self.duration_ms, 2),
        }


class DatabasePruner:
    """
    Prunes old data from the blockchain database.

    Keeps the most recent ``keep_blocks`` blocks and their transactions.
    Older blocks are deleted along with their associated transactions.
    Resolved quarantine entries and old treasury logs are also cleaned.

    Parameters
    ----------
    db : Database
        The Positronic database instance.
    keep_blocks : int
        Number of most-recent blocks to retain (default 10 000).
    keep_quarantine_days : int
        Days to keep resolved quarantine entries (default 7).
    keep_treasury_days : int
        Days to keep treasury log entries (default 90).
    """

    DEFAULT_KEEP_BLOCKS = 10_000
    DEFAULT_KEEP_QUARANTINE_DAYS = 7
    DEFAULT_KEEP_TREASURY_DAYS = 90

    def __init__(
        self,
        db,
        keep_blocks: int = DEFAULT_KEEP_BLOCKS,
        keep_quarantine_days: int = DEFAULT_KEEP_QUARANTINE_DAYS,
        keep_treasury_days: int = DEFAULT_KEEP_TREASURY_DAYS,
    ):
        self._db = db
        self._keep_blocks = keep_blocks
        self._keep_quarantine_days = keep_quarantine_days
        self._keep_treasury_days = keep_treasury_days

    def prune(self, vacuum: bool = True) -> PruneStats:
        """
        Run a full prune cycle.

        Returns
        -------
        PruneStats
            Counts of rows deleted from each table.
        """
        t0 = time.monotonic()
        stats = PruneStats()

        chain_height = self._get_chain_height()
        if chain_height < 0:
            logger.debug("No blocks to prune")
            return stats

        cutoff_height = chain_height - self._keep_blocks
        if cutoff_height > 0:
            stats.transactions_pruned = self._prune_transactions(cutoff_height)
            stats.blocks_pruned = self._prune_blocks(cutoff_height)

        stats.quarantine_pruned = self._prune_quarantine()
        stats.treasury_pruned = self._prune_treasury()

        if vacuum and (stats.blocks_pruned + stats.transactions_pruned) > 0:
            self._vacuum()

        stats.duration_ms = (time.monotonic() - t0) * 1000
        self._db.commit()

        logger.info(
            "Prune complete: %d blocks, %d txs, %d quarantine, %d treasury (%.0f ms)",
            stats.blocks_pruned,
            stats.transactions_pruned,
            stats.quarantine_pruned,
            stats.treasury_pruned,
            stats.duration_ms,
        )
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_chain_height(self) -> int:
        """Current chain height from database."""
        row = self._db.execute("SELECT MAX(height) FROM blocks").fetchone()
        if row and row[0] is not None:
            return row[0]
        return -1

    def _prune_blocks(self, cutoff_height: int) -> int:
        """Delete blocks below cutoff_height."""
        cursor = self._db.execute(
            "DELETE FROM blocks WHERE height < ?", (cutoff_height,)
        )
        count = cursor.rowcount
        if count > 0:
            logger.debug("Pruned %d blocks below height %d", count, cutoff_height)
        return count

    def _prune_transactions(self, cutoff_height: int) -> int:
        """Delete transactions from pruned blocks."""
        cursor = self._db.execute(
            "DELETE FROM transactions WHERE block_height < ?", (cutoff_height,)
        )
        count = cursor.rowcount
        if count > 0:
            logger.debug("Pruned %d transactions below height %d", count, cutoff_height)
        return count

    def _prune_quarantine(self) -> int:
        """Delete resolved quarantine entries older than threshold."""
        cutoff_ts = time.time() - (self._keep_quarantine_days * 86400)
        # Only prune entries that have been resolved (not still quarantined)
        cursor = self._db.execute(
            "DELETE FROM quarantine_pool WHERE status != 'quarantined' "
            "AND quarantined_at_block < (SELECT MAX(height) - ? FROM blocks)",
            (self._keep_blocks,),
        )
        count = cursor.rowcount
        if count > 0:
            logger.debug("Pruned %d resolved quarantine entries", count)
        return count

    def _prune_treasury(self) -> int:
        """Delete old treasury log entries."""
        cutoff_ts = time.time() - (self._keep_treasury_days * 86400)
        cursor = self._db.execute(
            "DELETE FROM treasury WHERE timestamp < ?", (cutoff_ts,)
        )
        count = cursor.rowcount
        if count > 0:
            logger.debug("Pruned %d treasury entries older than %d days",
                         count, self._keep_treasury_days)
        return count

    def _vacuum(self):
        """Reclaim disk space after large deletes."""
        try:
            self._db.execute("VACUUM")
            logger.debug("VACUUM completed")
        except Exception as e:
            # VACUUM can fail inside a transaction — that's OK
            logger.debug("VACUUM skipped: %s", e)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current database size metrics."""
        height = self._get_chain_height()
        block_count = self._db.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        tx_count = self._db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        quarantine_count = self._db.execute("SELECT COUNT(*) FROM quarantine_pool").fetchone()[0]

        return {
            "chain_height": height,
            "block_count": block_count,
            "transaction_count": tx_count,
            "quarantine_count": quarantine_count,
            "keep_blocks": self._keep_blocks,
            "prunable_blocks": max(0, block_count - self._keep_blocks),
        }
