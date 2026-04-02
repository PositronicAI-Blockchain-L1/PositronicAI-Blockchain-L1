"""
Positronic - Compact Block Relay

Efficient block propagation using only block header + transaction hashes.
Receiving peers reconstruct the full block from their local mempool,
drastically reducing bandwidth for block announcements.

Phase 17 GOD CHAIN addition.

**Fail-open**: If compact block reconstruction fails (missing TXs),
the peer requests a full block instead. No block is ever lost.
"""

from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CompactBlock:
    """A compact block representation for efficient relay.

    Contains only the block header and ordered transaction hashes.
    Receiving peers use their mempool to reconstruct the full block.

    Attributes:
        header: Full block header dictionary.
        tx_hashes: Ordered list of transaction hashes in the block.
        prefilled_txs: Transactions that peers are unlikely to have
            (e.g., coinbase/reward TXs). Sent in full to avoid
            reconstruction failures.
    """
    header: dict
    tx_hashes: List[str]
    prefilled_txs: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for wire transmission."""
        return {
            "header": self.header,
            "tx_hashes": self.tx_hashes,
            "prefilled_txs": self.prefilled_txs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompactBlock":
        """Deserialize from dictionary."""
        return cls(
            header=d.get("header", {}),
            tx_hashes=d.get("tx_hashes", []),
            prefilled_txs=d.get("prefilled_txs", []),
        )

    @classmethod
    def from_block(cls, block) -> "CompactBlock":
        """Create a compact block from a full block object.

        Reward and system transactions are prefilled since peers
        won't have them in their mempool.

        Args:
            block: A Block object with ``header`` and ``transactions``.

        Returns:
            CompactBlock with header, TX hashes, and prefilled system TXs.
        """
        try:
            header = {}
            if hasattr(block, "header") and hasattr(block.header, "to_dict"):
                header = block.header.to_dict()
            elif hasattr(block, "to_dict"):
                bd = block.to_dict()
                header = bd.get("header", bd)

            tx_hashes = []
            prefilled = []

            for tx in getattr(block, "transactions", []):
                tx_hash = ""
                if hasattr(tx, "hash"):
                    h = tx.hash
                    tx_hash = h.hex() if isinstance(h, bytes) else str(h)
                elif hasattr(tx, "tx_hash"):
                    h = tx.tx_hash
                    tx_hash = h.hex() if isinstance(h, bytes) else str(h)

                tx_hashes.append(tx_hash)

                # Prefill system/reward TXs (not in mempool)
                tx_type = ""
                if hasattr(tx, "tx_type"):
                    tt = tx.tx_type
                    tx_type = tt.name if hasattr(tt, "name") else str(tt)

                if tx_type.upper() in ("REWARD", "AI_TREASURY", "GAME_REWARD"):
                    tx_dict = tx.to_dict() if hasattr(tx, "to_dict") else {}
                    prefilled.append(tx_dict)

            return cls(
                header=header,
                tx_hashes=tx_hashes,
                prefilled_txs=prefilled,
            )
        except Exception as e:
            # Fail-safe: return empty compact block
            logger.warning("Failed to create compact block from block: %s", e)
            return cls(header={}, tx_hashes=[], prefilled_txs=[])

    @property
    def height(self) -> int:
        """Block height from header."""
        return self.header.get("height", 0)

    @property
    def tx_count(self) -> int:
        """Number of transactions in the block."""
        return len(self.tx_hashes)

    @property
    def bandwidth_savings(self) -> float:
        """Estimated bandwidth savings ratio (0.0 - 1.0).

        Compact blocks typically save 90%+ bandwidth since they only
        transmit ~32 bytes per TX hash vs full TX data.
        """
        if self.tx_count == 0:
            return 0.0
        prefilled_ratio = len(self.prefilled_txs) / self.tx_count
        return max(0.0, 1.0 - prefilled_ratio - 0.05)  # ~5% overhead for header


class CompactBlockHandler:
    """Handles compact block reconstruction from mempool.

    When a compact block arrives:
    1. Check which TX hashes exist in the local mempool.
    2. If all present → reconstruct immediately.
    3. If some missing → return the missing hashes for explicit request.
    4. Prefilled TXs (rewards) are used directly without mempool lookup.

    Example::

        handler = CompactBlockHandler(mempool)
        can_build, missing = handler.can_reconstruct(compact_block)
        if can_build:
            full_txs = handler.reconstruct_txs(compact_block)
    """

    def __init__(self, mempool=None):
        self._mempool = mempool
        # Stats tracking (Fix 5.2: real stats)
        self._blocks_compacted: int = 0
        self._blocks_reconstructed: int = 0
        self._reconstruction_failures: int = 0
        self._total_bytes_saved: int = 0

    def record_compaction(self, compact: CompactBlock):
        """Record that a block was compacted for relay."""
        self._blocks_compacted += 1
        # Estimate savings: ~500 bytes per TX vs ~32 bytes per hash
        non_prefilled = compact.tx_count - len(compact.prefilled_txs)
        self._total_bytes_saved += max(0, non_prefilled * (500 - 32))

    def can_reconstruct(self, compact: CompactBlock) -> Tuple[bool, List[str]]:
        """Check if all transactions are available for reconstruction.

        Args:
            compact: The compact block to evaluate.

        Returns:
            Tuple of (can_reconstruct, missing_tx_hashes).
        """
        if not compact.tx_hashes:
            return True, []

        # Build set of prefilled TX hashes
        prefilled_hashes: Set[str] = set()
        for ptx in compact.prefilled_txs:
            h = ptx.get("hash", ptx.get("tx_hash", ""))
            if h:
                prefilled_hashes.add(h if isinstance(h, str) else h.hex())

        missing = []
        for tx_hash in compact.tx_hashes:
            if tx_hash in prefilled_hashes:
                continue
            if not self._has_tx_in_mempool(tx_hash):
                missing.append(tx_hash)

        return len(missing) == 0, missing

    def reconstruct_txs(self, compact: CompactBlock) -> List[dict]:
        """Reconstruct full transaction list from compact block + mempool.

        Args:
            compact: The compact block with TX hashes and prefilled TXs.

        Returns:
            Ordered list of transaction dictionaries. Returns empty list
            on reconstruction failure.
        """
        try:
            # Index prefilled TXs
            prefilled_by_hash: Dict[str, dict] = {}
            for ptx in compact.prefilled_txs:
                h = ptx.get("hash", ptx.get("tx_hash", ""))
                key = h if isinstance(h, str) else (h.hex() if h else "")
                if key:
                    prefilled_by_hash[key] = ptx

            txs = []
            for tx_hash in compact.tx_hashes:
                if tx_hash in prefilled_by_hash:
                    txs.append(prefilled_by_hash[tx_hash])
                else:
                    tx = self._get_tx_from_mempool(tx_hash)
                    if tx is not None:
                        txs.append(tx)
                    else:
                        # Missing TX — reconstruction failed
                        self._reconstruction_failures += 1
                        return []

            self._blocks_reconstructed += 1
            return txs

        except Exception as e:
            logger.warning("Compact block reconstruction failed: %s", e)
            self._reconstruction_failures += 1
            return []

    def _has_tx_in_mempool(self, tx_hash: str) -> bool:
        """Check if a transaction exists in the mempool."""
        if self._mempool is None:
            return False
        try:
            # Check pending pool for the TX hash
            if hasattr(self._mempool, "pending"):
                return tx_hash in self._mempool.pending
            if hasattr(self._mempool, "has_tx"):
                return self._mempool.has_tx(tx_hash)
            return False
        except Exception as e:
            logger.debug("Mempool TX lookup failed: %s", e)
            return False

    def _get_tx_from_mempool(self, tx_hash: str) -> Optional[dict]:
        """Get a transaction from the mempool by hash."""
        if self._mempool is None:
            return None
        try:
            if hasattr(self._mempool, "pending"):
                tx = self._mempool.pending.get(tx_hash)
                if tx and hasattr(tx, "to_dict"):
                    return tx.to_dict()
                return tx
            if hasattr(self._mempool, "get_tx"):
                tx = self._mempool.get_tx(tx_hash)
                if tx and hasattr(tx, "to_dict"):
                    return tx.to_dict()
                return tx
            return None
        except Exception as e:
            logger.debug("Failed to get tx from mempool: %s", e)
            return None

    def get_stats(self) -> dict:
        """Return handler statistics."""
        return {
            "mempool_available": self._mempool is not None,
            "blocks_compacted": self._blocks_compacted,
            "blocks_reconstructed": self._blocks_reconstructed,
            "reconstruction_failures": self._reconstruction_failures,
            "total_bytes_saved": self._total_bytes_saved,
        }
