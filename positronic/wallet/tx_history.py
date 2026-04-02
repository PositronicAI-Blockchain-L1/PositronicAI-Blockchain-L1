"""
Positronic - Transaction History Tracker

Tracks and queries transaction history for watched wallet addresses.
Scans confirmed blocks and maintains an indexed history of sent,
received, and contract interactions.

Phase 17 GOD CHAIN addition.
"""

import time
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TxHistoryEntry:
    """A single transaction history record.

    Attributes:
        tx_hash: Transaction hash (hex string).
        tx_type: Transaction type name (transfer, contract_call, etc.).
        direction: ``"sent"``, ``"received"``, ``"contract"``, or ``"self"``.
        counterparty: The other address involved.
        value: Transaction value in base units.
        gas_used: Gas consumed by the transaction.
        gas_price: Gas price offered.
        fee: Total fee paid (gas_used * gas_price).
        block_height: Block height where TX was confirmed.
        timestamp: Unix timestamp of the block.
        status: ``"confirmed"``, ``"failed"``, or ``"pending"``.
        ai_score: AI risk score assigned to this transaction.
    """
    tx_hash: str
    tx_type: str = "transfer"
    direction: str = "sent"
    counterparty: str = ""
    value: int = 0
    gas_used: int = 0
    gas_price: int = 0
    fee: int = 0
    block_height: int = 0
    timestamp: float = 0.0
    status: str = "confirmed"
    ai_score: float = 0.0


class TxHistoryTracker:
    """Tracks transaction history for wallet addresses.

    Watches specified addresses and records all transactions involving
    them. Supports filtering by type, direction, and date range.

    Example::

        tracker = TxHistoryTracker()
        tracker.watch_address("0x1234...")
        tracker.record_transaction(tx, block_height=100)
        history = tracker.get_history("0x1234...", limit=20)
    """

    def __init__(self, max_history_per_address: int = 10_000):
        self._watched: Set[str] = set()
        self._history: Dict[str, List[TxHistoryEntry]] = {}
        self._max_per_address = max_history_per_address

    def watch_address(self, address: str):
        """Start watching an address for transactions.

        Args:
            address: Address to watch (hex string, with or without 0x).
        """
        addr = self._normalize_addr(address)
        self._watched.add(addr)
        if addr not in self._history:
            self._history[addr] = []

    def unwatch_address(self, address: str):
        """Stop watching an address (keeps existing history).

        Args:
            address: Address to unwatch.
        """
        addr = self._normalize_addr(address)
        self._watched.discard(addr)

    @property
    def watched_addresses(self) -> Set[str]:
        """Set of currently watched addresses."""
        return self._watched.copy()

    def record_transaction(
        self,
        tx,
        block_height: int = 0,
        block_timestamp: float = 0.0,
        status: str = "confirmed",
    ):
        """Record a transaction if it involves any watched address.

        Args:
            tx: Transaction object with sender, recipient, value, etc.
            block_height: Block height where TX was included.
            block_timestamp: Block timestamp.
            status: Transaction status ("confirmed" or "failed").
        """
        try:
            sender = self._get_addr(tx, "sender")
            recipient = self._get_addr(tx, "recipient")

            tx_hash = ""
            if hasattr(tx, "hash"):
                h = tx.hash
                tx_hash = h.hex() if isinstance(h, bytes) else str(h)
            elif hasattr(tx, "tx_hash"):
                h = tx.tx_hash
                tx_hash = h.hex() if isinstance(h, bytes) else str(h)

            tx_type = "transfer"
            if hasattr(tx, "tx_type"):
                tt = tx.tx_type
                tx_type = tt.name.lower() if hasattr(tt, "name") else str(tt).lower()

            value = getattr(tx, "value", 0) or 0
            gas_used = getattr(tx, "gas_used", 0) or getattr(tx, "gas_limit", 0) or 0
            gas_price = getattr(tx, "gas_price", 0) or 0
            fee = gas_used * gas_price
            ai_score = getattr(tx, "ai_score", 0.0) or 0.0
            timestamp = block_timestamp or getattr(tx, "timestamp", time.time())

            # Check if sender is watched
            if sender in self._watched:
                direction = "self" if recipient == sender else "sent"
                entry = TxHistoryEntry(
                    tx_hash=tx_hash,
                    tx_type=tx_type,
                    direction=direction,
                    counterparty=recipient,
                    value=value,
                    gas_used=gas_used,
                    gas_price=gas_price,
                    fee=fee,
                    block_height=block_height,
                    timestamp=timestamp,
                    status=status,
                    ai_score=ai_score,
                )
                self._add_entry(sender, entry)

            # Check if recipient is watched (and different from sender)
            if recipient in self._watched and recipient != sender:
                entry = TxHistoryEntry(
                    tx_hash=tx_hash,
                    tx_type=tx_type,
                    direction="received",
                    counterparty=sender,
                    value=value,
                    gas_used=gas_used,
                    gas_price=gas_price,
                    fee=fee,
                    block_height=block_height,
                    timestamp=timestamp,
                    status=status,
                    ai_score=ai_score,
                )
                self._add_entry(recipient, entry)

        except Exception as e:
            logger.debug("Failed to record transaction in history: %s", e)

    def get_history(
        self,
        address: str,
        direction: Optional[str] = None,
        tx_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TxHistoryEntry]:
        """Get transaction history for an address.

        Args:
            address: Address to query.
            direction: Filter by direction ("sent", "received", "contract").
            tx_type: Filter by transaction type ("transfer", "contract_call").
            limit: Maximum entries to return.
            offset: Skip this many entries (for pagination).

        Returns:
            List of TxHistoryEntry, most recent first.
        """
        addr = self._normalize_addr(address)
        entries = self._history.get(addr, [])

        # Apply filters
        if direction:
            entries = [e for e in entries if e.direction == direction]
        if tx_type:
            entries = [e for e in entries if e.tx_type == tx_type]

        # Sort by block height descending (most recent first)
        entries = sorted(entries, key=lambda e: e.block_height, reverse=True)

        return entries[offset:offset + limit]

    def get_balance_changes(self, address: str) -> List[Tuple[int, int]]:
        """Get cumulative balance changes over time.

        Args:
            address: Address to query.

        Returns:
            List of (block_height, cumulative_change) tuples.
        """
        addr = self._normalize_addr(address)
        entries = sorted(
            self._history.get(addr, []),
            key=lambda e: e.block_height,
        )

        cumulative = 0
        result = []
        for entry in entries:
            if entry.direction == "received":
                cumulative += entry.value
            elif entry.direction == "sent":
                cumulative -= (entry.value + entry.fee)
            result.append((entry.block_height, cumulative))

        return result

    def get_stats(self, address: Optional[str] = None) -> Dict:
        """Get history statistics."""
        if address:
            addr = self._normalize_addr(address)
            entries = self._history.get(addr, [])
            sent = sum(1 for e in entries if e.direction == "sent")
            received = sum(1 for e in entries if e.direction == "received")
            return {
                "total": len(entries),
                "sent": sent,
                "received": received,
            }
        return {
            "watched_addresses": len(self._watched),
            "total_entries": sum(len(v) for v in self._history.values()),
        }

    def _add_entry(self, address: str, entry: TxHistoryEntry):
        """Add an entry to an address's history, with capacity limit."""
        if address not in self._history:
            self._history[address] = []
        self._history[address].append(entry)
        if len(self._history[address]) > self._max_per_address:
            self._history[address] = self._history[address][-self._max_per_address:]

    @staticmethod
    def _normalize_addr(address: str) -> str:
        """Normalize address to lowercase with 0x prefix."""
        if isinstance(address, bytes):
            return "0x" + address.hex()
        addr = address.lower()
        if not addr.startswith("0x"):
            addr = "0x" + addr
        return addr

    @staticmethod
    def _get_addr(tx, field: str) -> str:
        """Extract and normalize an address field from a transaction."""
        val = getattr(tx, field, None)
        if val is None:
            return ""
        if isinstance(val, bytes):
            return "0x" + val.hex()
        return str(val).lower()
