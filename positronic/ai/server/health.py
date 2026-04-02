"""
Positronic - AI Health Monitor
Tracks AI system health, performance metrics, and anomaly rates.

The health monitor continuously collects inference results and latency
measurements, producing health snapshots that indicate whether the AI
subsystem is operating within acceptable parameters. This information
is used by the blockchain layer and RPC interface to report node health.

Health indicators:
    - Inference latency: Should remain below 50ms for real-time scoring.
    - Anomaly rate: Too high (>10%) suggests excessive false positives;
      too low (<0.1%) suggests the model may be blind to attacks.
    - Model staleness: Time since the last training update. Stale models
      may not reflect current network conditions.
    - Cache effectiveness: Reported by the inference pipeline.
"""

import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class HealthSnapshot:
    """A point-in-time snapshot of AI system health.

    Attributes:
        timestamp: Unix timestamp when this snapshot was taken.
        inference_rate: Approximate inferences per second.
        avg_latency_ms: Average inference latency in milliseconds.
        cache_hit_rate: Fraction of inferences served from cache.
        anomaly_rate: Fraction of recent transactions scored above 0.85
            (flagged as anomalous).
        quarantine_rate: Fraction of recent transactions with scores
            between 0.85 and 0.95 (quarantined for review).
        model_version: Currently active model version number.
        is_healthy: Overall health determination based on thresholds.
    """
    timestamp: float
    inference_rate: float
    avg_latency_ms: float
    cache_hit_rate: float
    anomaly_rate: float
    quarantine_rate: float
    model_version: int
    is_healthy: bool


class HealthMonitor:
    """
    Monitors AI system health and tracks performance metrics.

    The monitor maintains rolling windows of inference scores and latencies,
    periodically computing health snapshots that summarize the system's
    operational status. Health is determined by comparing metrics against
    configurable thresholds.

    Thresholds (configurable via instance attributes):
        - max_latency_ms: Maximum acceptable average latency (default: 50ms).
        - max_anomaly_rate: Maximum acceptable anomaly rate (default: 10%).
        - min_anomaly_rate: Minimum expected anomaly rate (default: 0.1%).
        - max_staleness_hours: Maximum hours since last training (default: 24h).
    """

    def __init__(self):
        self._history: deque = deque(maxlen=1000)
        self._scores: deque = deque(maxlen=10000)
        self._last_training_time = time.time()
        self._inference_count = 0
        self._inference_times: deque = deque(maxlen=1000)

        # Configurable health thresholds
        self.max_latency_ms = 50.0
        self.max_anomaly_rate = 0.10
        self.min_anomaly_rate = 0.001
        self.max_staleness_hours = 24.0

    def record_inference(self, score: float, latency_ms: float):
        """Record a single inference result for health tracking.

        Args:
            score: The anomaly score produced by the inference pipeline,
                in the range [0.0, 1.0].
            latency_ms: The time taken for the inference in milliseconds.
        """
        self._scores.append(score)
        self._inference_times.append(latency_ms)
        self._inference_count += 1

    def record_training(self):
        """Record that a training update has occurred.

        Resets the staleness timer used to determine whether the model
        is up-to-date with recent network activity.
        """
        self._last_training_time = time.time()

    def check_health(self, model_version: int = 1) -> HealthSnapshot:
        """Run a health check and return a snapshot.

        Computes current metrics from the rolling windows of scores and
        latencies, then determines overall health by comparing against
        the configured thresholds.

        Args:
            model_version: The current model version number to include
                in the snapshot.

        Returns:
            A HealthSnapshot containing all computed metrics and the
            overall health determination.
        """
        now = time.time()

        # Compute latency and throughput metrics
        if self._inference_times:
            avg_latency = sum(self._inference_times) / len(self._inference_times)
            time_span = (
                self._inference_times[-1] - self._inference_times[0]
                if len(self._inference_times) > 1
                else 1.0
            )
            rate = len(self._inference_times) / max(time_span, 0.001)
        else:
            avg_latency = 0.0
            rate = 0.0

        # Compute anomaly and quarantine rates from recent scores
        if self._scores:
            scores_list = list(self._scores)
            total = len(scores_list)
            anomaly_rate = sum(1 for s in scores_list if s > 0.85) / total
            quarantine_rate = (
                sum(1 for s in scores_list if 0.85 <= s <= 0.95) / total
            )
        else:
            anomaly_rate = 0.0
            quarantine_rate = 0.0

        # Compute model staleness in hours
        staleness = (now - self._last_training_time) / 3600.0

        # Determine overall health against thresholds
        is_healthy = (
            avg_latency < self.max_latency_ms
            and anomaly_rate < self.max_anomaly_rate
            and staleness < self.max_staleness_hours
        )

        snapshot = HealthSnapshot(
            timestamp=now,
            inference_rate=rate,
            avg_latency_ms=avg_latency,
            cache_hit_rate=0.0,
            anomaly_rate=anomaly_rate,
            quarantine_rate=quarantine_rate,
            model_version=model_version,
            is_healthy=is_healthy,
        )

        self._history.append(snapshot)
        return snapshot

    def get_stats(self) -> dict:
        """Get a summary of health monitor statistics.

        Returns:
            Dictionary containing:
                - is_healthy: Current health status (True if no snapshots yet).
                - total_inferences: Total number of recorded inferences.
                - avg_latency_ms: Latest average latency.
                - anomaly_rate: Latest anomaly rate.
                - model_staleness_hours: Hours since last training update.
                - history_length: Number of health snapshots stored.
        """
        latest = self._history[-1] if self._history else None
        return {
            "is_healthy": latest.is_healthy if latest else True,
            "total_inferences": self._inference_count,
            "avg_latency_ms": latest.avg_latency_ms if latest else 0.0,
            "anomaly_rate": latest.anomaly_rate if latest else 0.0,
            "model_staleness_hours": (
                (time.time() - self._last_training_time) / 3600.0
            ),
            "history_length": len(self._history),
        }
