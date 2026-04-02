"""
Positronic - Prometheus-Compatible Metrics

Exposes blockchain metrics in Prometheus text exposition format.
No external dependencies -- implements the text format directly.

Endpoint: GET /metrics on the RPC server port.
"""

import time
import threading
from typing import Dict, Optional
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Metric:
    """A single metric with optional labels."""

    def __init__(self, name: str, help_text: str, metric_type: MetricType):
        self.name = name
        self.help_text = help_text
        self.type = metric_type
        self._values: Dict[str, float] = {}  # label_str -> value
        self._lock = threading.Lock()

    def set(self, value: float, labels: Dict[str, str] = None):
        """Set gauge value."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, amount: float = 1.0, labels: Dict[str, str] = None):
        """Increment counter."""
        key = self._labels_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0) + amount

    def _labels_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def expose(self) -> str:
        """Render in Prometheus text format."""
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} {self.type.value}",
        ]
        with self._lock:
            for label_key, value in self._values.items():
                if label_key:
                    lines.append(f"{self.name}{{{label_key}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class MetricsRegistry:
    """Central registry for all metrics."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def gauge(self, name: str, help_text: str) -> Metric:
        return self._register(name, help_text, MetricType.GAUGE)

    def counter(self, name: str, help_text: str) -> Metric:
        return self._register(name, help_text, MetricType.COUNTER)

    def _register(self, name: str, help_text: str, metric_type: MetricType) -> Metric:
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(name, help_text, metric_type)
            return self._metrics[name]

    def expose_all(self) -> str:
        """Render all metrics in Prometheus text format."""
        with self._lock:
            metrics = list(self._metrics.values())
        parts = [m.expose() for m in metrics]
        return "\n\n".join(parts) + "\n"


# Global registry
REGISTRY = MetricsRegistry()

# -- Pre-defined metrics -------------------------------------------------------

# Chain
chain_height = REGISTRY.gauge(
    "positronic_chain_height",
    "Current blockchain height",
)
chain_block_time = REGISTRY.gauge(
    "positronic_chain_block_time_seconds",
    "Average block time in seconds",
)

# Peers
peers_connected = REGISTRY.gauge(
    "positronic_peers_connected",
    "Number of connected peers",
)
peers_inbound = REGISTRY.gauge(
    "positronic_peers_inbound",
    "Number of inbound peer connections",
)
peers_outbound = REGISTRY.gauge(
    "positronic_peers_outbound",
    "Number of outbound peer connections",
)
peers_avg_latency = REGISTRY.gauge(
    "positronic_peers_avg_latency_ms",
    "Average peer latency in milliseconds",
)

# Consensus
validators_active = REGISTRY.gauge(
    "positronic_validators_active",
    "Number of active validators",
)
consensus_participation = REGISTRY.gauge(
    "positronic_consensus_participation_rate",
    "Consensus participation rate (0-1)",
)

# Mempool
mempool_size = REGISTRY.gauge(
    "positronic_mempool_size",
    "Number of pending transactions in mempool",
)
mempool_bytes = REGISTRY.gauge(
    "positronic_mempool_bytes",
    "Total size of mempool in bytes",
)

# AI
ai_total_scored = REGISTRY.counter(
    "positronic_ai_total_scored",
    "Total transactions scored by AI",
)
ai_accepted = REGISTRY.counter(
    "positronic_ai_accepted",
    "Transactions accepted by AI",
)
ai_quarantined = REGISTRY.counter(
    "positronic_ai_quarantined",
    "Transactions quarantined by AI",
)
ai_rejected = REGISTRY.counter(
    "positronic_ai_rejected",
    "Transactions rejected by AI",
)

# Transactions
tx_total = REGISTRY.counter(
    "positronic_transactions_total",
    "Total processed transactions",
)

# Health
health_status = REGISTRY.gauge(
    "positronic_health_status",
    "Overall health status (0=healthy, 4=down)",
)
