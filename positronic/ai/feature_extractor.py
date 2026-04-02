"""
Positronic - Transaction Feature Extractor
Extracts 35 features from each transaction for AI validation.
Features capture sender behavior, value patterns, timing, network context,
and Phase 16 graph/behavioral/flow analysis features.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from positronic.core.transaction import Transaction, TxType
from positronic.core.account import Account
from positronic.constants import AI_FEATURE_DIM


@dataclass
class TransactionFeatures:
    """Feature vector for a single transaction."""

    # === Sender Behavior Features (8) ===
    sender_balance_ratio: float = 0.0       # value / sender_balance
    sender_nonce: int = 0                    # Transaction count history
    sender_avg_value: float = 0.0           # Average historical TX value
    sender_value_deviation: float = 0.0      # How much this TX deviates from avg
    sender_tx_frequency: float = 0.0        # TXs per hour recently
    sender_age: float = 0.0                 # Account age in hours
    sender_unique_recipients: int = 0        # Unique addresses sent to
    sender_reputation: float = 1.0           # AI reputation score

    # === Value Features (5) ===
    value_log: float = 0.0                  # log10(value + 1)
    value_is_round: int = 0                 # Is value a round number
    gas_price_ratio: float = 0.0            # gas_price / network_avg_gas
    total_cost_ratio: float = 0.0           # total_cost / sender_balance
    value_percentile: float = 0.0           # Percentile in recent TX values

    # === Timing Features (4) ===
    time_since_last_tx: float = 0.0         # Seconds since sender's last TX
    hour_of_day: float = 0.0               # 0-23, normalized to 0-1
    is_burst: int = 0                       # Part of a rapid burst of TXs
    mempool_wait_time: float = 0.0          # Time spent in mempool

    # === Transaction Type Features (4) ===
    tx_type: int = 0                        # Transaction type enum
    has_data: int = 0                       # Contains calldata
    data_size: int = 0                      # Size of calldata
    is_contract_interaction: int = 0         # Calls a contract

    # === Network Context Features (5) ===
    mempool_size: int = 0                   # Current mempool size
    recent_block_fullness: float = 0.0      # Avg gas used / gas limit
    network_tx_rate: float = 0.0            # Network-wide TXs per second
    gas_price_vs_median: float = 0.0        # Gas price relative to median
    pending_from_sender: int = 0             # Pending TXs from same sender

    # === Phase 16: Graph/Behavioral/Flow Features (9) ===
    sender_recipient_tx_count: float = 0.0  # How many times sender->recipient before
    sender_cluster_diversity: float = 0.0   # Unique recipients / total TXs
    recipient_popularity: float = 0.0       # Unique senders to this recipient
    sender_tx_regularity: float = 0.0       # Std dev of inter-TX intervals (normalized)
    contract_call_ratio: float = 0.0        # % of sender's TXs that are contract calls
    value_entropy: float = 0.0             # Shannon entropy of sender's value distribution
    incoming_outgoing_ratio: float = 0.0    # Received / Sent ratio for sender
    value_velocity: float = 0.0            # Rate of value change over time
    gas_efficiency: float = 0.0            # gas_used / gas_limit

    def to_vector(self) -> List[float]:
        """Convert features to a flat numeric vector for model input (35 features)."""
        return [
            self.sender_balance_ratio,
            float(self.sender_nonce),
            self.sender_avg_value,
            self.sender_value_deviation,
            self.sender_tx_frequency,
            self.sender_age,
            float(self.sender_unique_recipients),
            self.sender_reputation,
            self.value_log,
            float(self.value_is_round),
            self.gas_price_ratio,
            self.total_cost_ratio,
            self.value_percentile,
            self.time_since_last_tx,
            self.hour_of_day,
            float(self.is_burst),
            self.mempool_wait_time,
            float(self.tx_type),
            float(self.has_data),
            float(self.data_size),
            float(self.is_contract_interaction),
            float(self.mempool_size),
            self.recent_block_fullness,
            self.network_tx_rate,
            self.gas_price_vs_median,
            float(self.pending_from_sender),
            # Phase 16: 9 new features
            self.sender_recipient_tx_count,
            self.sender_cluster_diversity,
            self.recipient_popularity,
            self.sender_tx_regularity,
            self.contract_call_ratio,
            self.value_entropy,
            self.incoming_outgoing_ratio,
            self.value_velocity,
            self.gas_efficiency,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "sender_balance_ratio", "sender_nonce", "sender_avg_value",
            "sender_value_deviation", "sender_tx_frequency", "sender_age",
            "sender_unique_recipients", "sender_reputation",
            "value_log", "value_is_round", "gas_price_ratio",
            "total_cost_ratio", "value_percentile",
            "time_since_last_tx", "hour_of_day", "is_burst", "mempool_wait_time",
            "tx_type", "has_data", "data_size", "is_contract_interaction",
            "mempool_size", "recent_block_fullness", "network_tx_rate",
            "gas_price_vs_median", "pending_from_sender",
            # Phase 16
            "sender_recipient_tx_count", "sender_cluster_diversity",
            "recipient_popularity", "sender_tx_regularity",
            "contract_call_ratio", "value_entropy",
            "incoming_outgoing_ratio", "value_velocity", "gas_efficiency",
        ]

    @staticmethod
    def vector_size() -> int:
        return AI_FEATURE_DIM  # 35

    def to_extended_vector(self) -> List[float]:
        """Extended feature vector with additional computed features (39 features)."""
        base = self.to_vector()
        # Add computed features
        base.append(self.sender_balance_ratio * self.value_log)  # interaction
        base.append(float(self.is_burst) * self.sender_tx_frequency)  # burst intensity
        base.append(self.gas_price_ratio * float(self.is_contract_interaction))  # contract gas
        base.append(self.total_cost_ratio * float(self.sender_nonce > 0))  # experienced sender cost
        return base

    @staticmethod
    def extended_vector_size() -> int:
        return AI_FEATURE_DIM + 4  # 39


class FeatureExtractor:
    """
    Extracts features from transactions using account history and network state.
    Maintains rolling statistics for normalization.
    """

    MAX_TRACKED_SENDERS = 10000

    def __init__(self):
        # Rolling statistics
        self.recent_values: List[int] = []          # Last 1000 TX values
        self.recent_gas_prices: List[int] = []      # Last 1000 gas prices
        self.sender_history: Dict[bytes, List[float]] = {}  # sender -> [timestamps]
        self.sender_values: Dict[bytes, List[int]] = {}     # sender -> [values]
        self.sender_recipients: Dict[bytes, set] = {}       # sender -> {recipients}
        self.avg_gas_price: float = 1.0
        self.median_gas_price: float = 1.0
        self.network_tx_rate: float = 0.0
        self.avg_block_fullness: float = 0.5

        # Deterministic sequence counter (replaces wall-clock time for cross-node determinism)
        self._tx_counter: int = 0

        # NEW: Sequence buffer for temporal models
        self._sequence_buffer: List[List[float]] = []
        self._max_sequence: int = 50

        # Phase 16: Additional tracking for new features
        self._recipient_incoming: Dict[bytes, Set[bytes]] = {}   # recipient -> {senders}
        self._sender_total_sent: Dict[bytes, int] = {}           # sender -> cumulative value sent
        self._sender_total_received: Dict[bytes, int] = {}       # address -> cumulative value received
        self._sender_contract_calls: Dict[bytes, int] = {}       # sender -> contract call count
        self._sender_total_txs: Dict[bytes, int] = {}            # sender -> total tx count
        self._sender_recipient_pairs: Dict[bytes, Dict[bytes, int]] = {}  # sender -> {recipient: count}

    def extract(
        self,
        tx: Transaction,
        sender_account: Optional[Account] = None,
        mempool_size: int = 0,
    ) -> TransactionFeatures:
        """Extract features from a transaction."""
        features = TransactionFeatures()

        # Sender info
        balance = sender_account.balance if sender_account else 0
        nonce = sender_account.nonce if sender_account else 0
        reputation = sender_account.ai_reputation if sender_account else 1.0

        # === Sender Behavior ===
        features.sender_balance_ratio = (
            tx.value / max(balance, 1) if balance > 0 else 1.0
        )
        features.sender_nonce = nonce
        features.sender_reputation = reputation

        # Historical values for sender
        sender_key = tx.sender
        if sender_key in self.sender_values and self.sender_values[sender_key]:
            vals = self.sender_values[sender_key]
            features.sender_avg_value = sum(vals) / len(vals)
            if features.sender_avg_value > 0:
                features.sender_value_deviation = (
                    abs(tx.value - features.sender_avg_value)
                    / features.sender_avg_value
                )

        # TX frequency
        if sender_key in self.sender_history and self.sender_history[sender_key]:
            timestamps = self.sender_history[sender_key]
            if len(timestamps) >= 2:
                span = timestamps[-1] - timestamps[0]
                if span > 0:
                    features.sender_tx_frequency = len(timestamps) / (span / 3600)
            features.time_since_last_tx = float(self._tx_counter - timestamps[-1])
            features.sender_age = float(self._tx_counter - timestamps[0]) / 3600

        # Unique recipients
        features.sender_unique_recipients = len(
            self.sender_recipients.get(sender_key, set())
        )

        # === Value Features ===
        features.value_log = math.log10(tx.value + 1) if tx.value >= 0 else 0
        features.value_is_round = int(
            tx.value > 0 and tx.value % (10 ** 18) == 0
        )
        features.gas_price_ratio = tx.gas_price / max(self.avg_gas_price, 1)
        features.total_cost_ratio = (
            tx.total_cost / max(balance, 1) if balance > 0 else 1.0
        )

        # Value percentile
        if self.recent_values:
            below = sum(1 for v in self.recent_values if v <= tx.value)
            features.value_percentile = below / len(self.recent_values)

        # === Timing ===
        # Use block-aligned timestamp for deterministic cross-node agreement.
        # tx.timestamp is set at block proposal time (same across all validators).
        # Fall back to deterministic counter if no timestamp available.
        tx_timestamp = getattr(tx, 'timestamp', 0)
        if tx_timestamp > 0:
            import datetime
            dt = datetime.datetime.fromtimestamp(tx_timestamp, tz=datetime.timezone.utc)
            features.hour_of_day = dt.hour / 24.0
        else:
            features.hour_of_day = (self._tx_counter % 24) / 24.0

        # Burst detection: more than 5 TXs in last 10 counter ticks
        if sender_key in self.sender_history:
            recent = [
                t for t in self.sender_history[sender_key]
                if self._tx_counter - t < 10
            ]
            features.is_burst = int(len(recent) >= 5)
            features.pending_from_sender = len(recent)

        # === TX Type ===
        features.tx_type = int(tx.tx_type)
        features.has_data = int(len(tx.data) > 0)
        features.data_size = len(tx.data)
        features.is_contract_interaction = int(
            tx.tx_type in (TxType.CONTRACT_CREATE, TxType.CONTRACT_CALL)
        )

        # === Network Context ===
        features.mempool_size = mempool_size
        features.recent_block_fullness = self.avg_block_fullness
        features.network_tx_rate = self.network_tx_rate
        features.gas_price_vs_median = tx.gas_price / max(self.median_gas_price, 1)

        # === Phase 16: Graph/Behavioral/Flow Features ===
        recipient_key = tx.recipient

        # 1. sender_recipient_tx_count: how many times sender->recipient before
        pair_counts = self._sender_recipient_pairs.get(sender_key, {})
        features.sender_recipient_tx_count = float(pair_counts.get(recipient_key, 0))

        # 2. sender_cluster_diversity: unique recipients / total TXs
        total_txs = self._sender_total_txs.get(sender_key, 0)
        unique_recips = len(self.sender_recipients.get(sender_key, set()))
        features.sender_cluster_diversity = (
            unique_recips / max(total_txs, 1)
        )

        # 3. recipient_popularity: how many unique senders to this recipient
        features.recipient_popularity = float(
            len(self._recipient_incoming.get(recipient_key, set()))
        )

        # 4. sender_tx_regularity: std dev of inter-TX intervals (normalized)
        if sender_key in self.sender_history and len(self.sender_history[sender_key]) >= 3:
            timestamps = self.sender_history[sender_key]
            intervals = [
                timestamps[i+1] - timestamps[i]
                for i in range(len(timestamps) - 1)
            ]
            mean_interval = sum(intervals) / len(intervals)
            if mean_interval > 0:
                variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(intervals)
                features.sender_tx_regularity = math.sqrt(variance) / mean_interval
            else:
                features.sender_tx_regularity = 0.0

        # 5. contract_call_ratio: % of sender's TXs that are contract calls
        cc_count = self._sender_contract_calls.get(sender_key, 0)
        features.contract_call_ratio = cc_count / max(total_txs, 1)

        # 6. value_entropy: Shannon entropy of sender's value distribution
        if sender_key in self.sender_values and len(self.sender_values[sender_key]) >= 2:
            vals = self.sender_values[sender_key]
            total_val = sum(abs(v) for v in vals)
            if total_val > 0:
                entropy = 0.0
                for v in vals:
                    p = abs(v) / total_val
                    if p > 0:
                        entropy -= p * math.log2(p)
                features.value_entropy = entropy

        # 7. incoming_outgoing_ratio: received / sent ratio
        total_sent = self._sender_total_sent.get(sender_key, 0)
        total_received = self._sender_total_received.get(sender_key, 0)
        features.incoming_outgoing_ratio = (
            total_received / max(total_sent, 1)
        )

        # 8. value_velocity: rate of value change over time
        if sender_key in self.sender_values and sender_key in self.sender_history:
            vals = self.sender_values[sender_key]
            timestamps = self.sender_history[sender_key]
            if len(vals) >= 2 and len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                if time_span > 0:
                    value_range = max(vals) - min(vals)
                    features.value_velocity = math.log10(value_range + 1) / time_span

        # 9. gas_efficiency: gas_used / gas_limit
        if tx.gas_limit > 0:
            gas_used = getattr(tx, 'gas_used', tx.gas_limit)
            features.gas_efficiency = gas_used / tx.gas_limit

        # Increment deterministic counter
        self._tx_counter += 1

        # NEW: Buffer for sequence models
        self._sequence_buffer.append(features.to_vector())
        if len(self._sequence_buffer) > self._max_sequence:
            self._sequence_buffer = self._sequence_buffer[-self._max_sequence:]

        return features

    def get_recent_sequence(self, max_len: int = 50) -> List[List[float]]:
        """Get recent feature sequence for temporal models."""
        return self._sequence_buffer[-max_len:]

    def update_stats(self, tx: Transaction):
        """Update rolling statistics with a new transaction."""
        sender_key = tx.sender
        recipient_key = tx.recipient

        # Value history
        self.recent_values.append(tx.value)
        if len(self.recent_values) > 1000:
            self.recent_values = self.recent_values[-1000:]

        # Gas price history
        self.recent_gas_prices.append(tx.gas_price)
        if len(self.recent_gas_prices) > 1000:
            self.recent_gas_prices = self.recent_gas_prices[-1000:]
        self.avg_gas_price = sum(self.recent_gas_prices) / len(self.recent_gas_prices)
        sorted_prices = sorted(self.recent_gas_prices)
        self.median_gas_price = sorted_prices[len(sorted_prices) // 2]

        # Sender history
        if sender_key not in self.sender_history:
            self.sender_history[sender_key] = []
        self.sender_history[sender_key].append(self._tx_counter)
        if len(self.sender_history[sender_key]) > 100:
            self.sender_history[sender_key] = self.sender_history[sender_key][-100:]

        # Sender values
        if sender_key not in self.sender_values:
            self.sender_values[sender_key] = []
        self.sender_values[sender_key].append(tx.value)
        if len(self.sender_values[sender_key]) > 100:
            self.sender_values[sender_key] = self.sender_values[sender_key][-100:]

        # Recipients
        if sender_key not in self.sender_recipients:
            self.sender_recipients[sender_key] = set()
        self.sender_recipients[sender_key].add(recipient_key)

        # Phase 16: Track recipient incoming senders
        if recipient_key not in self._recipient_incoming:
            self._recipient_incoming[recipient_key] = set()
        self._recipient_incoming[recipient_key].add(sender_key)

        # Phase 16: Track cumulative value sent/received
        self._sender_total_sent[sender_key] = (
            self._sender_total_sent.get(sender_key, 0) + tx.value
        )
        self._sender_total_received[recipient_key] = (
            self._sender_total_received.get(recipient_key, 0) + tx.value
        )

        # Phase 16: Track contract calls and total TXs
        self._sender_total_txs[sender_key] = (
            self._sender_total_txs.get(sender_key, 0) + 1
        )
        if tx.tx_type in (TxType.CONTRACT_CREATE, TxType.CONTRACT_CALL):
            self._sender_contract_calls[sender_key] = (
                self._sender_contract_calls.get(sender_key, 0) + 1
            )

        # Phase 16: Track sender->recipient pair counts
        if sender_key not in self._sender_recipient_pairs:
            self._sender_recipient_pairs[sender_key] = {}
        self._sender_recipient_pairs[sender_key][recipient_key] = (
            self._sender_recipient_pairs[sender_key].get(recipient_key, 0) + 1
        )

        # LRU eviction: remove oldest entries when dicts exceed MAX_TRACKED_SENDERS
        if len(self.sender_history) > self.MAX_TRACKED_SENDERS:
            oldest_key = next(iter(self.sender_history))
            del self.sender_history[oldest_key]
            self.sender_values.pop(oldest_key, None)
            self.sender_recipients.pop(oldest_key, None)
            self._sender_total_sent.pop(oldest_key, None)
            self._sender_contract_calls.pop(oldest_key, None)
            self._sender_total_txs.pop(oldest_key, None)
            self._sender_recipient_pairs.pop(oldest_key, None)

        # LRU eviction for recipient tracking
        if len(self._recipient_incoming) > self.MAX_TRACKED_SENDERS:
            oldest_key = next(iter(self._recipient_incoming))
            del self._recipient_incoming[oldest_key]
            self._sender_total_received.pop(oldest_key, None)

    def update_network_stats(
        self, tx_rate: float, block_fullness: float
    ):
        """Update network-level statistics."""
        self.network_tx_rate = tx_rate
        self.avg_block_fullness = block_fullness
