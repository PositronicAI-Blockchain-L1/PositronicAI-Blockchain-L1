"""
Positronic - Transaction Mempool
Transaction pool with AI pre-screening and priority ordering.
All transactions are fully traceable.
AI validation runs at mempool entry to reject malicious TXs early.
"""

import time
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from positronic.core.transaction import Transaction, TxType, TxStatus
from positronic.crypto.address import address_from_pubkey

logger = logging.getLogger("positronic.network.mempool")


class Mempool:
    """
    Transaction mempool with AI-enhanced ordering.
    Transactions are sorted by gas price (priority) and AI score (trust).

    AI enforcement at mempool level:
    - Transactions are scored by AIValidationGate on entry
    - Rejected TXs (score > 0.95) are dropped immediately
    - Quarantined TXs (0.85-0.95) are routed to QuarantinePool
    - Accepted TXs (score < 0.85) enter the normal pending pool

    Phase 17: Transaction priority lanes (SYSTEM > FAST > STANDARD > HEAVY).
    Lane ordering is advisory — all valid TXs are always included.
    """

    MAX_SIZE = 10_000
    MAX_PER_ACCOUNT = 64

    def __init__(self):
        """Initialize mempool with default configuration."""
        self.pending: Dict[bytes, Transaction] = {}  # tx_hash -> tx
        self.by_sender: Dict[bytes, List[bytes]] = {}  # sender_addr -> [tx_hashes]
        self._sorted_cache: Optional[List[Transaction]] = None
        self._blocked_addresses: Set[bytes] = set()  # Blocked by immune system
        self._quarantined_hashes: Set[bytes] = set()  # Track already-quarantined TXs

        # AI integration (set by Node when AI is available)
        self._ai_gate = None
        self._quarantine_pool = None
        self._immune_system = None
        self._state_manager = None  # Set by Node for nonce validation

        # Phase 17: Transaction priority lane classifier (fail-open)
        self._tx_classifier = None
        try:
            from positronic.chain.tx_pipeline import TxPriorityClassifier
            self._tx_classifier = TxPriorityClassifier()
        except Exception as e:
            logger.debug("TxPriorityClassifier unavailable (fail-open): %s", e)
            pass  # Fail-open: if import fails, use legacy ordering

        # AI mempool stats
        self.ai_rejected: int = 0
        self.ai_quarantined: int = 0
        self.ai_accepted: int = 0

    def set_ai_gate(self, ai_gate):
        """Set the AI validation gate for mempool-level screening."""
        self._ai_gate = ai_gate

    def set_quarantine_pool(self, quarantine_pool):
        """Set the quarantine pool for borderline transactions."""
        self._quarantine_pool = quarantine_pool

    def set_immune_system(self, immune_system):
        """Set the neural immune system for threat monitoring."""
        self._immune_system = immune_system

    def add(self, tx: Transaction, current_block_height: int = 0) -> bool:
        """
        Add a transaction to the mempool.
        Runs AI validation if available.
        Returns True if accepted, False if rejected/quarantined.
        """
        tx_hash = tx.tx_hash

        # Security fix: reject system transaction types from mempool
        # System TXs (REWARD, AI_TREASURY, GAME_REWARD) are created during
        # block production, never submitted externally via RPC/P2P.
        if tx.tx_type in (TxType.REWARD, TxType.AI_TREASURY, TxType.GAME_REWARD):
            logger.warning(
                f"Rejected system TX type {tx.tx_type.name} from mempool: "
                f"{tx_hash.hex()[:16]}"
            )
            return False

        # Check duplicates
        if tx_hash in self.pending:
            return False

        # FIX 2: Prevent quarantine re-entry — reject if already quarantined
        if tx_hash in self._quarantined_hashes:
            return False

        # Check pool size
        if len(self.pending) >= self.MAX_SIZE:
            if not self._evict_lowest_priority():
                return False

        # Check per-sender limit
        sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""

        # Check if sender is blocked by immune system
        if sender_addr in self._blocked_addresses:
            return False

        # AI validation at mempool entry
        if self._ai_gate is not None and tx.tx_type not in (TxType.REWARD, TxType.AI_TREASURY):
            try:
                result = self._ai_gate.validate_transaction(
                    tx,
                    mempool_size=len(self.pending),
                )
                if result.status == TxStatus.REJECTED:
                    # Rejected by AI - drop immediately
                    self.ai_rejected += 1
                    logger.debug(
                        f"TX {tx_hash.hex()[:16]} rejected by AI "
                        f"(score={result.final_score:.3f})"
                    )
                    # Report to immune system
                    if self._immune_system:
                        self._immune_system.report_anomaly(
                            sender_addr,
                            result.final_score,
                            f"TX rejected at mempool: score={result.final_score:.3f}",
                            block_height=current_block_height,
                        )
                    return False

                if result.status == TxStatus.QUARANTINED:
                    # Quarantined - route to quarantine pool
                    self.ai_quarantined += 1
                    self._quarantined_hashes.add(tx_hash)  # FIX 2: track quarantined
                    if self._quarantine_pool:
                        self._quarantine_pool.add(
                            tx, result.final_score, current_block_height
                        )
                        logger.debug(
                            f"TX {tx_hash.hex()[:16]} quarantined "
                            f"(score={result.final_score:.3f})"
                        )
                    return False  # Not in mempool (in quarantine instead)

                self.ai_accepted += 1
            except Exception as e:
                # Security fix: fail-closed — reject TX on AI gate errors
                # to prevent malicious TXs from bypassing AI validation.
                logger.warning(f"AI validation error at mempool (rejected): {e}")
                return False

        sender_txs = self.by_sender.get(sender_addr, [])
        if len(sender_txs) >= self.MAX_PER_ACCOUNT:
            return False

        # Security fix: reject duplicate nonce from same sender (anti-double-spend)
        for existing_hash in sender_txs:
            existing_tx = self.pending.get(existing_hash)
            if existing_tx and existing_tx.nonce == tx.nonce:
                logger.debug(
                    "Rejected duplicate nonce %d from %s (double-spend attempt)",
                    tx.nonce, sender_addr.hex()[:12]
                )
                return False

        # Security fix: reject nonce lower than current state nonce (anti-replay)
        if self._state_manager is not None:
            current_nonce = self._state_manager.get_nonce(sender_addr)
            if tx.nonce < current_nonce:
                logger.debug(
                    "Rejected stale nonce %d < %d from %s (replay attempt)",
                    tx.nonce, current_nonce, sender_addr.hex()[:12]
                )
                return False

        # Add to pool
        self.pending[tx_hash] = tx
        if sender_addr not in self.by_sender:
            self.by_sender[sender_addr] = []
        self.by_sender[sender_addr].append(tx_hash)
        self._sorted_cache = None

        return True

    def remove(self, tx_hash: bytes):
        """Remove a transaction from the mempool."""
        tx = self.pending.pop(tx_hash, None)
        if tx:
            sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""
            if sender_addr in self.by_sender:
                self.by_sender[sender_addr] = [
                    h for h in self.by_sender[sender_addr] if h != tx_hash
                ]
                if not self.by_sender[sender_addr]:
                    del self.by_sender[sender_addr]

            self._sorted_cache = None

    def get(self, tx_hash: bytes) -> Optional[Transaction]:
        return self.pending.get(tx_hash)

    def contains(self, tx_hash: bytes) -> bool:
        return tx_hash in self.pending

    def get_pending_transactions(
        self, max_count: int = 1000, max_gas: int = 30_000_000
    ) -> List[Transaction]:
        """Get transactions sorted by priority lanes then gas price.

        Phase 17: Uses TxPriorityClassifier for lane-based ordering
        (SYSTEM > FAST > STANDARD > HEAVY). Falls back to legacy
        gas-price ordering if the classifier is unavailable.
        """
        if self._sorted_cache is None:
            # Phase 17: lane-based ordering (fail-open)
            try:
                if self._tx_classifier is not None:
                    self._sorted_cache = self._tx_classifier.get_lane_ordering(
                        list(self.pending.values())
                    )
                else:
                    raise ValueError("No classifier")
            except Exception as e:
                # Fallback: legacy gas-price ordering
                logger.debug("Lane-based ordering unavailable, using legacy ordering: %s", e)
                self._sorted_cache = sorted(
                    self.pending.values(),
                    key=lambda tx: (-tx.gas_price, tx.timestamp),
                )

        result = []
        total_gas = 0
        for tx in self._sorted_cache:
            if len(result) >= max_count:
                break
            if total_gas + tx.gas_limit > max_gas:
                continue
            result.append(tx)
            total_gas += tx.gas_limit

        return result

    def get_transactions_by_sender(self, sender_addr: bytes) -> List[Transaction]:
        """Get all pending transactions from a sender."""
        tx_hashes = self.by_sender.get(sender_addr, [])
        return [self.pending[h] for h in tx_hashes if h in self.pending]

    def on_block_added(self, block_txs: List[Transaction]):
        """Remove transactions that were included in a new block."""
        for tx in block_txs:
            self.remove(tx.tx_hash)

    def get_all_pending(self) -> List[Transaction]:
        """Get all pending transactions."""
        return list(self.pending.values())

    def _evict_lowest_priority(self) -> bool:
        """Evict the transaction with lowest gas price."""
        if not self.pending:
            return False
        worst = min(self.pending.values(), key=lambda tx: tx.gas_price)
        self.remove(worst.tx_hash)
        return True

    @property
    def size(self) -> int:
        return len(self.pending)

    def block_address(self, address: bytes):
        """Block an address from submitting transactions (immune system)."""
        self._blocked_addresses.add(address)
        # Remove all pending TXs from this address
        tx_hashes = self.by_sender.get(address, []).copy()
        for tx_hash in tx_hashes:
            self.remove(tx_hash)

    def unblock_address(self, address: bytes):
        """Unblock an address."""
        self._blocked_addresses.discard(address)

    def sync_blocked_addresses(self, blocked: set):
        """Sync blocked addresses from the immune system."""
        self._blocked_addresses = blocked

    def get_stats(self) -> dict:
        stats = {
            "size": self.size,
            "senders": len(self.by_sender),
            "blocked_addresses": len(self._blocked_addresses),
            "ai_rejected": self.ai_rejected,
            "ai_quarantined": self.ai_quarantined,
            "ai_accepted": self.ai_accepted,
            "ai_enforcement": self._ai_gate is not None,
            "lane_ordering": self._tx_classifier is not None,
        }
        # Phase 17: Include lane distribution if classifier is available
        try:
            if self._tx_classifier is not None and self.pending:
                stats["lane_stats"] = self._tx_classifier.get_lane_stats(
                    list(self.pending.values())
                )
        except Exception as e:
            logger.debug("Failed to get lane stats (fail-open): %s", e)
        return stats
