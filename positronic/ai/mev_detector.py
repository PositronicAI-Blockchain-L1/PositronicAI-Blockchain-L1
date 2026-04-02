"""
Positronic - MEV & Sandwich Attack Detector (MSAD)
Detects front-running, back-running, and sandwich attacks in the mempool.
Uses temporal pattern analysis on transaction sequences.
"""

import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from positronic.utils.logging import get_logger
from positronic.core.transaction import Transaction, TxType

logger = get_logger(__name__)


@dataclass
class MempoolSnapshot:
    """A snapshot of mempool state for sequence analysis."""
    timestamp: float
    tx_hash: bytes
    sender: bytes
    recipient: bytes
    value: int
    gas_price: int
    tx_type: TxType


class MEVDetector:
    """
    Detects MEV (Maximal Extractable Value) attacks by analyzing
    transaction sequences in the mempool.

    Detection strategies:
    1. Sandwich Detection: A-B-A pattern where attacker brackets a victim
    2. Front-running: Same contract call with higher gas price right after victim
    3. Back-running: Profitable TX immediately after a state-changing TX
    4. Gas Price Manipulation: Abnormal gas price sequences
    """

    def __init__(self):
        self.mempool_history: List[MempoolSnapshot] = []
        self.sender_sequences: Dict[bytes, List[MempoolSnapshot]] = defaultdict(list)
        self.contract_interactions: Dict[bytes, List[MempoolSnapshot]] = defaultdict(list)
        self.max_history = 500  # Keep last 500 TXs for analysis
        self.MAX_TRACKED = 10000

        # Neural model for temporal pattern detection
        try:
            from positronic.ai.models.temporal_attention import TemporalAttentionNet
            self._temporal_net = TemporalAttentionNet(tx_feature_dim=10, d_model=64)
            self._neural_available = True
        except ImportError:
            self._temporal_net = None
            self._neural_available = False
        self._use_neural = False
        self._neural_threshold = 500
        self._neural_samples = 0
        self._train_buffer: List = []
        self._neural_errors: int = 0
        self._consecutive_neural_failures: int = 0

    def analyze_transaction(
        self,
        tx: Transaction,
        pending_txs: List[Transaction] = None,
    ) -> float:
        """
        Analyze a transaction for MEV attack patterns.
        Returns risk score 0.0 (safe) to 1.0 (likely MEV attack).
        """
        score = 0.0

        # Create snapshot
        snapshot = MempoolSnapshot(
            timestamp=time.time(),
            tx_hash=tx.tx_hash,
            sender=tx.sender,
            recipient=tx.recipient,
            value=tx.value,
            gas_price=tx.gas_price,
            tx_type=tx.tx_type,
        )

        # Check for sandwich attack pattern
        sandwich_score = self._detect_sandwich(snapshot, pending_txs or [])
        score = max(score, sandwich_score)

        # Check for front-running
        frontrun_score = self._detect_frontrun(snapshot, pending_txs or [])
        score = max(score, frontrun_score)

        # Check for gas price manipulation
        gas_score = self._detect_gas_manipulation(snapshot)
        score = max(score, gas_score)

        # Neural augmentation: if active, compute neural score and take max
        if self._use_neural and self._temporal_net is not None:
            try:
                from positronic.ai.models.temporal_attention import TemporalAttentionNet
                seq_features = TemporalAttentionNet.extract_sequence_features(
                    tx, pending_txs or []
                )
                neural_score = self._temporal_net.score(seq_features)
                score = max(score, neural_score)
                self._consecutive_neural_failures = 0
            except Exception as e:
                logger.debug("Temporal attention scoring error (neural fallback): %s", e)
                self._neural_errors += 1  # Neural fallback: temporal scoring
                self._consecutive_neural_failures += 1
                if self._consecutive_neural_failures > 10:
                    self._use_neural = False  # Deactivate neural on degradation

        # Update history
        self._update_history(snapshot)

        # Track samples and activate neural model after threshold
        self._neural_samples += 1
        if self._neural_samples >= self._neural_threshold and self._neural_available:
            self._use_neural = True

        return min(score, 1.0)

    def _detect_sandwich(
        self,
        tx_snapshot: MempoolSnapshot,
        pending: List[Transaction],
    ) -> float:
        """
        Detect sandwich attacks: Attacker places TXs before and after victim.
        Pattern: Attacker buys -> Victim buys (price up) -> Attacker sells

        Signals:
        - Same sender has 2+ pending TXs to same contract
        - One TX before and one after a third-party TX
        - Large gas price difference to ensure ordering
        """
        score = 0.0

        # Check if this sender has other pending TXs to same recipient
        same_sender_to_target = [
            t for t in pending
            if t.sender == tx_snapshot.sender
            and t.recipient == tx_snapshot.recipient
            and t.tx_hash != tx_snapshot.tx_hash
        ]

        if len(same_sender_to_target) >= 1:
            # Check for a victim TX in between
            victims = [
                t for t in pending
                if t.sender != tx_snapshot.sender
                and t.recipient == tx_snapshot.recipient
            ]

            if victims:
                # Potential sandwich: check gas price ordering
                for attacker_tx in same_sender_to_target:
                    for victim_tx in victims:
                        if (
                            attacker_tx.gas_price > victim_tx.gas_price * 1.5
                            or tx_snapshot.gas_price > victim_tx.gas_price * 1.5
                        ):
                            score = max(score, 0.8)
                        elif attacker_tx.gas_price > victim_tx.gas_price:
                            score = max(score, 0.5)

        return score

    def _detect_frontrun(
        self,
        tx_snapshot: MempoolSnapshot,
        pending: List[Transaction],
    ) -> float:
        """
        Detect front-running: Someone copies a pending TX with higher gas.

        Signals:
        - Same recipient and similar calldata
        - Much higher gas price
        - Submitted right after the original
        """
        score = 0.0

        # Look for pending TXs to same recipient with lower gas
        similar_pending = [
            t for t in pending
            if t.recipient == tx_snapshot.recipient
            and t.sender != tx_snapshot.sender
            and t.gas_price < tx_snapshot.gas_price
        ]

        for other_tx in similar_pending:
            gas_ratio = tx_snapshot.gas_price / max(other_tx.gas_price, 1)

            # Same recipient, much higher gas = potential front-run
            if gas_ratio > 5:
                score = max(score, 0.7)
            elif gas_ratio > 2:
                score = max(score, 0.4)

            # If contract calls with similar value, even more suspicious
            if (
                tx_snapshot.tx_type == TxType.CONTRACT_CALL
                and other_tx.tx_type == TxType.CONTRACT_CALL
                and abs(tx_snapshot.value - other_tx.value) < tx_snapshot.value * 0.1
            ):
                score = max(score, 0.8)

        return score

    def _detect_gas_manipulation(self, tx_snapshot: MempoolSnapshot) -> float:
        """
        Detect gas price manipulation patterns.

        Signals:
        - Extreme gas prices (>100x median)
        - Rapid gas price escalation from same sender
        """
        score = 0.0

        # Check sender's gas price history
        sender_history = self.sender_sequences.get(tx_snapshot.sender, [])
        if sender_history:
            recent_gas = [s.gas_price for s in sender_history[-10:]]
            avg_gas = sum(recent_gas) / len(recent_gas)

            # Current gas much higher than sender's average
            if avg_gas > 0 and tx_snapshot.gas_price > avg_gas * 10:
                score = max(score, 0.5)

            # Rapid escalation pattern
            if len(recent_gas) >= 3:
                increasing = all(
                    recent_gas[i] < recent_gas[i + 1]
                    for i in range(len(recent_gas) - 1)
                )
                if increasing and recent_gas[-1] > recent_gas[0] * 5:
                    score = max(score, 0.6)

        return score

    def _update_history(self, snapshot: MempoolSnapshot):
        """Update internal history buffers."""
        self.mempool_history.append(snapshot)
        if len(self.mempool_history) > self.max_history:
            self.mempool_history = self.mempool_history[-self.max_history:]

        self.sender_sequences[snapshot.sender].append(snapshot)
        if len(self.sender_sequences[snapshot.sender]) > 50:
            self.sender_sequences[snapshot.sender] = \
                self.sender_sequences[snapshot.sender][-50:]

        if snapshot.tx_type in (TxType.CONTRACT_CALL, TxType.CONTRACT_CREATE):
            self.contract_interactions[snapshot.recipient].append(snapshot)
            if len(self.contract_interactions[snapshot.recipient]) > 50:
                self.contract_interactions[snapshot.recipient] = \
                    self.contract_interactions[snapshot.recipient][-50:]

        self._evict_if_needed()

    def _evict_if_needed(self):
        if len(self.sender_sequences) > self.MAX_TRACKED:
            oldest_key = next(iter(self.sender_sequences))
            del self.sender_sequences[oldest_key]
        if len(self.contract_interactions) > self.MAX_TRACKED:
            oldest_key = next(iter(self.contract_interactions))
            del self.contract_interactions[oldest_key]

    def get_stats(self) -> dict:
        return {
            "history_size": len(self.mempool_history),
            "tracked_senders": len(self.sender_sequences),
            "tracked_contracts": len(self.contract_interactions),
            "neural_active": self._use_neural,
            "neural_errors": self._neural_errors,
        }
