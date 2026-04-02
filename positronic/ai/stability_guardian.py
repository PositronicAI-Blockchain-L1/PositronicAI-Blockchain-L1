"""
Positronic - Economic Stability Guardian (ESG)
Monitors network-wide metrics for coordinated manipulation attempts.
Detects flash loan attacks, pump & dump, and coordinated spam.
"""

import time
import math
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NetworkMetrics:
    """Current network health metrics."""
    timestamp: float = 0.0
    tx_rate: float = 0.0              # Transactions per second
    avg_value: float = 0.0            # Average TX value
    total_volume: int = 0             # Total volume in time window
    unique_senders: int = 0           # Unique senders in window
    unique_recipients: int = 0        # Unique recipients in window
    avg_gas_price: float = 0.0        # Average gas price
    mempool_size: int = 0             # Current mempool depth
    block_fullness: float = 0.0       # Block gas utilization
    validator_participation: float = 0.0  # Active validators ratio


class StabilityGuardian:
    """
    Monitors network-wide economic metrics using an LSTM-like
    time series analysis to detect coordinated attacks.

    Detection targets:
    1. Flash Loan patterns: Sudden large borrows + trades + repay in one TX
    2. Pump & Dump: Coordinated buying followed by selling
    3. Spam attacks: Flood of small transactions
    4. Network congestion attacks: Gas price manipulation
    5. Sybil-like patterns: Many new accounts acting in concert
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # Time series of network metrics
        self.metrics_history: deque = deque(maxlen=window_size)

        # Rolling counters
        self.tx_count_window: deque = deque(maxlen=window_size)
        self.value_window: deque = deque(maxlen=window_size)
        self.gas_window: deque = deque(maxlen=window_size)

        # Pattern tracking
        self.sender_activity: Dict[bytes, deque] = {}
        self.new_accounts_window: deque = deque(maxlen=window_size)

        # LSTM-like state (simplified with exponential moving averages)
        self.ema_tx_rate: float = 0.0
        self.ema_value: float = 0.0
        self.ema_gas: float = 0.0
        self.ema_alpha: float = 0.1  # Smoothing factor

        # Alert thresholds
        self.alert_level: float = 0.0

        # Neural model for time-series network risk prediction
        try:
            from positronic.ai.models.lstm_attention import LSTMAttentionNet
            self._lstm_net = LSTMAttentionNet(input_size=9, hidden_size=32)
            self._neural_available = True
        except ImportError:
            self._lstm_net = None
            self._neural_available = False
        self._use_neural = False
        self._neural_threshold = 200
        self._neural_samples = 0
        self._neural_errors: int = 0
        self._consecutive_neural_failures: int = 0

    def update_metrics(self, metrics: NetworkMetrics):
        """Update with new network metrics snapshot."""
        self.metrics_history.append(metrics)

        # Update EMAs
        self.ema_tx_rate = (
            self.ema_alpha * metrics.tx_rate
            + (1 - self.ema_alpha) * self.ema_tx_rate
        )
        self.ema_value = (
            self.ema_alpha * metrics.avg_value
            + (1 - self.ema_alpha) * self.ema_value
        )
        self.ema_gas = (
            self.ema_alpha * metrics.avg_gas_price
            + (1 - self.ema_alpha) * self.ema_gas
        )

    def assess_network_risk(self) -> float:
        """
        Assess current network-wide risk level.
        Returns 0.0 (stable) to 1.0 (under attack).
        """
        if len(self.metrics_history) < 3:
            return 0.0

        scores = [
            self._detect_volume_spike(),
            self._detect_gas_manipulation(),
            self._detect_spam_flood(),
            self._detect_coordinated_activity(),
            self._detect_congestion_attack(),
        ]

        # Take weighted combination
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        rule_score = sum(s * w for s, w in zip(scores, weights))

        # Neural scoring: if active, blend LSTM prediction with rule-based
        if self._use_neural and self._lstm_net is not None:
            try:
                from positronic.ai.models.lstm_attention import LSTMAttentionNet
                metrics_array = LSTMAttentionNet.metrics_to_array(
                    list(self.metrics_history)[-50:]
                )
                neural_score = self._lstm_net.score(metrics_array)
                # Blend: 70% neural + 30% rules for stability
                self.alert_level = min(0.7 * neural_score + 0.3 * rule_score, 1.0)
                self._consecutive_neural_failures = 0
            except Exception as e:
                logger.debug("LSTM scoring error (neural fallback): %s", e)
                self.alert_level = min(rule_score, 1.0)
                self._neural_errors += 1  # Neural fallback: LSTM scoring
                self._consecutive_neural_failures += 1
                if self._consecutive_neural_failures > 10:
                    self._use_neural = False  # Deactivate neural on degradation
        else:
            self.alert_level = min(rule_score, 1.0)

        # Track samples and activate neural model after threshold
        self._neural_samples += 1
        if self._neural_samples >= self._neural_threshold and self._neural_available:
            self._use_neural = True

        return self.alert_level

    def get_transaction_risk_modifier(self) -> float:
        """
        Get a risk modifier based on current network state.
        Applied to individual transaction scores.
        During an attack, all transactions get slightly higher scores.
        """
        return self.alert_level * 0.3  # Max 30% boost during attack

    def _detect_volume_spike(self) -> float:
        """Detect sudden volume spikes (potential flash loan / pump)."""
        if len(self.metrics_history) < 10:
            return 0.0

        recent = list(self.metrics_history)[-5:]
        older = list(self.metrics_history)[-10:-5]

        recent_vol = sum(m.total_volume for m in recent)
        older_vol = sum(m.total_volume for m in older) or 1

        ratio = recent_vol / older_vol

        if ratio > 10:
            return 0.9
        elif ratio > 5:
            return 0.6
        elif ratio > 3:
            return 0.3
        return 0.0

    def _detect_gas_manipulation(self) -> float:
        """Detect gas price manipulation patterns."""
        if len(self.metrics_history) < 5:
            return 0.0

        recent_gas = [m.avg_gas_price for m in list(self.metrics_history)[-5:]]

        if not recent_gas or self.ema_gas == 0:
            return 0.0

        current_gas = recent_gas[-1]
        ratio = current_gas / max(self.ema_gas, 1)

        # Check for rapid increase
        if ratio > 20:
            return 0.8
        elif ratio > 10:
            return 0.5
        elif ratio > 5:
            return 0.3
        return 0.0

    def _detect_spam_flood(self) -> float:
        """Detect transaction spam/flood attacks."""
        if len(self.metrics_history) < 5:
            return 0.0

        recent = list(self.metrics_history)[-5:]
        tx_rates = [m.tx_rate for m in recent]
        avg_rate = sum(tx_rates) / len(tx_rates) if tx_rates else 0

        # Compare to EMA
        if self.ema_tx_rate == 0:
            return 0.0

        ratio = avg_rate / max(self.ema_tx_rate, 1)

        # Check for low-value spam
        avg_vals = [m.avg_value for m in recent]
        low_value = all(v < 100 for v in avg_vals)  # Very low average value

        if ratio > 10 and low_value:
            return 0.9  # Spam attack
        elif ratio > 5:
            return 0.4
        elif ratio > 3:
            return 0.2
        return 0.0

    def _detect_coordinated_activity(self) -> float:
        """Detect coordinated activity from multiple accounts."""
        if len(self.metrics_history) < 5:
            return 0.0

        recent = list(self.metrics_history)[-3:]

        # High TX rate but low unique senders = coordinated
        for m in recent:
            if m.tx_rate > 0 and m.unique_senders > 0:
                tx_per_sender = m.tx_rate / m.unique_senders
                if tx_per_sender > 10:
                    return 0.7  # Few senders, many TXs

        # Many new accounts suddenly active
        new_account_count = sum(self.new_accounts_window)
        if new_account_count > 50:
            return 0.5

        return 0.0

    def _detect_congestion_attack(self) -> float:
        """Detect deliberate congestion (filling blocks with garbage)."""
        if len(self.metrics_history) < 5:
            return 0.0

        recent = list(self.metrics_history)[-5:]
        fullness = [m.block_fullness for m in recent]
        avg_fullness = sum(fullness) / len(fullness) if fullness else 0

        mempool = [m.mempool_size for m in recent]
        avg_mempool = sum(mempool) / len(mempool) if mempool else 0

        # Blocks full + large mempool = congestion
        if avg_fullness > 0.95 and avg_mempool > 5000:
            return 0.7
        elif avg_fullness > 0.9 and avg_mempool > 1000:
            return 0.4
        return 0.0

    def record_new_account(self):
        """Record that a new account was created."""
        self.new_accounts_window.append(1)

    def get_stats(self) -> dict:
        return {
            "alert_level": self.alert_level,
            "ema_tx_rate": self.ema_tx_rate,
            "ema_value": self.ema_value,
            "ema_gas": self.ema_gas,
            "history_length": len(self.metrics_history),
            "neural_active": self._use_neural,
            "neural_errors": self._neural_errors,
        }
