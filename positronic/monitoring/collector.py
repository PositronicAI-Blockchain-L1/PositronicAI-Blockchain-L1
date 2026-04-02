"""Periodic collector that updates Prometheus metrics from HealthMonitor."""

import asyncio
import logging
from typing import Optional

from positronic.monitoring.metrics import (
    chain_height,
    chain_block_time,
    peers_connected,
    peers_inbound,
    peers_outbound,
    peers_avg_latency,
    validators_active,
    consensus_participation,
    mempool_size,
    health_status,
)

logger = logging.getLogger("positronic.monitoring.collector")


class MetricsCollector:
    """Collects metrics from node components and updates Prometheus gauges."""

    def __init__(self, health_monitor=None, blockchain=None, mempool=None):
        self._health_monitor = health_monitor
        self._blockchain = blockchain
        self._mempool = mempool
        self._running = False

    async def start(self, interval: float = 5.0):
        """Start periodic collection."""
        self._running = True
        while self._running:
            try:
                self.collect()
            except Exception as e:
                logger.warning(f"Metrics collection failed: {e}")
            await asyncio.sleep(interval)

    def stop(self):
        self._running = False

    def collect(self):
        """Collect current metrics from all sources."""
        if self._health_monitor:
            report = self._health_monitor.get_health_report()

            # Peer health
            if hasattr(report, "peer_health"):
                peers_connected.set(report.peer_health.connected_peers)
                peers_inbound.set(report.peer_health.inbound_peers)
                peers_outbound.set(report.peer_health.outbound_peers)
                peers_avg_latency.set(report.peer_health.avg_latency_ms)

            # Consensus health
            if hasattr(report, "consensus_health"):
                validators_active.set(report.consensus_health.active_validators)
                consensus_participation.set(
                    report.consensus_health.participation_rate
                )

            # Block production
            if hasattr(report, "block_production"):
                chain_height.set(report.block_production.current_height)
                chain_block_time.set(report.block_production.avg_block_time)

            # Overall health status (HealthStatus is an IntEnum)
            if hasattr(report, "status"):
                health_status.set(int(report.status))

            # Mempool health
            if hasattr(report, "mempool_health"):
                mempool_size.set(report.mempool_health.pending_txs)

        # Direct blockchain access as fallback / additional source
        if self._blockchain is not None:
            try:
                chain_height.set(self._blockchain.height)
            except Exception as e:
                logger.debug("metrics_blockchain_fallback_failed: %s", e)

        # Direct mempool access as fallback / additional source
        if self._mempool is not None:
            try:
                mempool_size.set(self._mempool.size)
            except Exception as e:
                logger.debug("metrics_mempool_fallback_failed: %s", e)
