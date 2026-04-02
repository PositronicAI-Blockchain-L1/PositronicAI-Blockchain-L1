"""
Positronic - Network Health Monitor
Comprehensive real-time monitoring of network health, node performance,
and consensus liveness. Superior to Bitcoin's lack of built-in monitoring.

Monitors:
- Peer connectivity and latency
- Block production rate and finality
- Mempool health and congestion
- Consensus participation rate
- Network throughput (TPS)
- AI system health
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("positronic.network.health_monitor")


class HealthStatus(IntEnum):
    """Overall network health status."""
    HEALTHY = 0       # All systems nominal
    DEGRADED = 1      # Minor issues, still operational
    WARNING = 2       # Significant issues, needs attention
    CRITICAL = 3      # Severe issues, intervention needed
    DOWN = 4          # Network is non-functional


@dataclass
class PeerHealthMetrics:
    """Health metrics for peer connections."""
    total_peers: int = 0
    connected_peers: int = 0
    inbound_peers: int = 0
    outbound_peers: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    banned_peers: int = 0
    stale_peers: int = 0
    peer_diversity_score: float = 0.0  # 0-1, how diverse are our connections


@dataclass
class ConsensusHealthMetrics:
    """Health metrics for consensus."""
    active_validators: int = 0
    total_validators: int = 0
    participation_rate: float = 0.0  # 0-1
    missed_blocks_recent: int = 0
    blocks_in_last_minute: int = 0
    expected_blocks_per_minute: float = 20.0  # 60/3 = 20
    finality_lag_slots: int = 0
    last_finalized_slot: int = 0


@dataclass
class MempoolHealthMetrics:
    """Health metrics for the mempool."""
    pending_txs: int = 0
    mempool_size_bytes: int = 0
    avg_gas_price: float = 0.0
    congestion_level: float = 0.0  # 0-1
    txs_per_second: float = 0.0
    ai_rejection_rate: float = 0.0
    quarantine_count: int = 0


@dataclass
class BlockProductionMetrics:
    """Health metrics for block production."""
    current_height: int = 0
    blocks_per_minute: float = 0.0
    avg_block_time: float = 3.0
    avg_block_gas_used: int = 0
    avg_block_gas_utilization: float = 0.0  # 0-1
    avg_txs_per_block: float = 0.0
    tps_current: float = 0.0
    tps_peak: float = 0.0


@dataclass
class HealthReport:
    """Complete health report."""
    status: HealthStatus = HealthStatus.HEALTHY
    timestamp: float = 0.0
    uptime_seconds: float = 0.0
    peer_health: PeerHealthMetrics = field(default_factory=PeerHealthMetrics)
    consensus_health: ConsensusHealthMetrics = field(default_factory=ConsensusHealthMetrics)
    mempool_health: MempoolHealthMetrics = field(default_factory=MempoolHealthMetrics)
    block_production: BlockProductionMetrics = field(default_factory=BlockProductionMetrics)
    issues: List[str] = field(default_factory=list)
    score: float = 100.0  # 0-100

    def to_dict(self) -> dict:
        return {
            "status": self.status.name,
            "score": round(self.score, 1),
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "issues": self.issues,
            "peer_health": {
                "total_peers": self.peer_health.total_peers,
                "connected_peers": self.peer_health.connected_peers,
                "inbound_peers": self.peer_health.inbound_peers,
                "outbound_peers": self.peer_health.outbound_peers,
                "avg_latency_ms": round(self.peer_health.avg_latency_ms, 1),
                "max_latency_ms": round(self.peer_health.max_latency_ms, 1),
                "banned_peers": self.peer_health.banned_peers,
                "peer_diversity_score": round(self.peer_health.peer_diversity_score, 3),
            },
            "consensus_health": {
                "active_validators": self.consensus_health.active_validators,
                "total_validators": self.consensus_health.total_validators,
                "participation_rate": round(self.consensus_health.participation_rate, 3),
                "missed_blocks_recent": self.consensus_health.missed_blocks_recent,
                "blocks_in_last_minute": self.consensus_health.blocks_in_last_minute,
                "finality_lag_slots": self.consensus_health.finality_lag_slots,
            },
            "mempool_health": {
                "pending_txs": self.mempool_health.pending_txs,
                "congestion_level": round(self.mempool_health.congestion_level, 3),
                "txs_per_second": round(self.mempool_health.txs_per_second, 2),
                "ai_rejection_rate": round(self.mempool_health.ai_rejection_rate, 3),
                "quarantine_count": self.mempool_health.quarantine_count,
            },
            "block_production": {
                "current_height": self.block_production.current_height,
                "blocks_per_minute": round(self.block_production.blocks_per_minute, 2),
                "avg_block_time": round(self.block_production.avg_block_time, 2),
                "avg_gas_utilization": round(self.block_production.avg_block_gas_utilization, 3),
                "tps_current": round(self.block_production.tps_current, 2),
                "tps_peak": round(self.block_production.tps_peak, 2),
            },
        }


class NetworkHealthMonitor:
    """
    Real-time network health monitoring system.

    Collects metrics from all node subsystems and produces a
    comprehensive health score and report.

    Usage:
        monitor = NetworkHealthMonitor()
        monitor.record_block(height, tx_count, gas_used, timestamp)
        monitor.record_peer_state(peers_info)
        report = monitor.get_health_report()
    """

    # Health thresholds
    MIN_HEALTHY_PEERS = 4
    WARNING_LATENCY_MS = 500
    CRITICAL_LATENCY_MS = 2000
    MIN_BLOCK_RATE = 10  # blocks/minute (50% of expected)
    MAX_CONGESTION = 0.9
    MAX_AI_REJECTION_RATE = 0.3

    def __init__(self):
        self._start_time = time.time()
        self._block_timestamps: List[float] = []
        self._block_tx_counts: List[int] = []
        self._block_gas_used: List[int] = []
        self._tx_timestamps: List[float] = []
        self._tps_peak: float = 0.0

        # Current state (updated by record_* methods)
        self._peer_count: int = 0
        self._connected_peers: int = 0
        self._inbound_peers: int = 0
        self._outbound_peers: int = 0
        self._avg_latency: float = 0.0
        self._max_latency: float = 0.0
        self._banned_peers: int = 0

        self._active_validators: int = 0
        self._total_validators: int = 0
        self._missed_blocks: int = 0
        self._last_finalized: int = 0

        self._mempool_size: int = 0
        self._ai_rejections: int = 0
        self._ai_total: int = 0
        self._quarantine_count: int = 0
        self._current_height: int = 0

    def record_block(
        self,
        height: int,
        tx_count: int,
        gas_used: int,
        timestamp: float = 0.0,
    ):
        """Record a new block for metrics."""
        ts = timestamp or time.time()
        self._block_timestamps.append(ts)
        self._block_tx_counts.append(tx_count)
        self._block_gas_used.append(gas_used)
        self._current_height = height

        # Keep only last 200 blocks
        if len(self._block_timestamps) > 200:
            self._block_timestamps = self._block_timestamps[-200:]
            self._block_tx_counts = self._block_tx_counts[-200:]
            self._block_gas_used = self._block_gas_used[-200:]

    def record_transaction(self, timestamp: float = 0.0):
        """Record a transaction for TPS metrics."""
        ts = timestamp or time.time()
        self._tx_timestamps.append(ts)
        # Keep only last 1000
        if len(self._tx_timestamps) > 1000:
            self._tx_timestamps = self._tx_timestamps[-1000:]

    def record_peer_state(
        self,
        total: int,
        connected: int,
        inbound: int,
        outbound: int,
        avg_latency: float,
        max_latency: float,
        banned: int,
    ):
        """Record peer connection state."""
        self._peer_count = total
        self._connected_peers = connected
        self._inbound_peers = inbound
        self._outbound_peers = outbound
        self._avg_latency = avg_latency
        self._max_latency = max_latency
        self._banned_peers = banned

    def record_consensus_state(
        self,
        active_validators: int,
        total_validators: int,
        missed_blocks: int,
        last_finalized: int,
    ):
        """Record consensus state."""
        self._active_validators = active_validators
        self._total_validators = total_validators
        self._missed_blocks = missed_blocks
        self._last_finalized = last_finalized

    def record_mempool_state(
        self,
        pending: int,
        quarantine_count: int = 0,
    ):
        """Record mempool state."""
        self._mempool_size = pending
        self._quarantine_count = quarantine_count

    def record_ai_decision(self, rejected: bool):
        """Record an AI validation decision."""
        self._ai_total += 1
        if rejected:
            self._ai_rejections += 1

    def get_health_report(self) -> HealthReport:
        """Generate a comprehensive health report."""
        now = time.time()
        report = HealthReport(timestamp=now)
        report.uptime_seconds = now - self._start_time
        issues = []
        score = 100.0

        # === Peer Health ===
        report.peer_health = PeerHealthMetrics(
            total_peers=self._peer_count,
            connected_peers=self._connected_peers,
            inbound_peers=self._inbound_peers,
            outbound_peers=self._outbound_peers,
            avg_latency_ms=self._avg_latency,
            max_latency_ms=self._max_latency,
            banned_peers=self._banned_peers,
        )

        # Peer diversity (balance of inbound/outbound)
        if self._connected_peers > 0:
            ratio = min(self._inbound_peers, self._outbound_peers) / max(
                self._inbound_peers, self._outbound_peers, 1
            )
            report.peer_health.peer_diversity_score = ratio

        if self._connected_peers < self.MIN_HEALTHY_PEERS:
            score -= 20
            issues.append(f"Low peer count: {self._connected_peers}")
        if self._avg_latency > self.CRITICAL_LATENCY_MS:
            score -= 15
            issues.append(f"Critical latency: {self._avg_latency:.0f}ms")
        elif self._avg_latency > self.WARNING_LATENCY_MS:
            score -= 5
            issues.append(f"High latency: {self._avg_latency:.0f}ms")

        # === Block Production ===
        blocks_per_minute = self._calc_blocks_per_minute(now)
        avg_block_time = self._calc_avg_block_time()
        tps = self._calc_tps(now)
        self._tps_peak = max(self._tps_peak, tps)

        avg_gas = 0
        avg_txs = 0.0
        gas_util = 0.0
        if self._block_gas_used:
            avg_gas = sum(self._block_gas_used[-20:]) // max(len(self._block_gas_used[-20:]), 1)
            gas_util = avg_gas / 30_000_000  # Relative to default gas limit
        if self._block_tx_counts:
            avg_txs = sum(self._block_tx_counts[-20:]) / max(len(self._block_tx_counts[-20:]), 1)

        report.block_production = BlockProductionMetrics(
            current_height=self._current_height,
            blocks_per_minute=blocks_per_minute,
            avg_block_time=avg_block_time,
            avg_block_gas_used=avg_gas,
            avg_block_gas_utilization=gas_util,
            avg_txs_per_block=avg_txs,
            tps_current=tps,
            tps_peak=self._tps_peak,
        )

        if blocks_per_minute < self.MIN_BLOCK_RATE and self._current_height > 10:
            score -= 20
            issues.append(f"Low block rate: {blocks_per_minute:.1f}/min")

        # === Consensus Health ===
        participation = 0.0
        if self._total_validators > 0:
            participation = self._active_validators / self._total_validators

        finality_lag = 0
        if self._current_height > 0 and self._last_finalized > 0:
            finality_lag = self._current_height - self._last_finalized

        report.consensus_health = ConsensusHealthMetrics(
            active_validators=self._active_validators,
            total_validators=self._total_validators,
            participation_rate=participation,
            missed_blocks_recent=self._missed_blocks,
            blocks_in_last_minute=int(blocks_per_minute),
            finality_lag_slots=finality_lag,
            last_finalized_slot=self._last_finalized,
        )

        if participation < 0.67 and self._total_validators > 1:
            score -= 25
            issues.append(f"Low participation: {participation:.0%}")

        # === Mempool Health ===
        congestion = min(1.0, self._mempool_size / 10000) if self._mempool_size > 0 else 0.0
        ai_rej_rate = self._ai_rejections / max(self._ai_total, 1)

        report.mempool_health = MempoolHealthMetrics(
            pending_txs=self._mempool_size,
            congestion_level=congestion,
            txs_per_second=tps,
            ai_rejection_rate=ai_rej_rate,
            quarantine_count=self._quarantine_count,
        )

        if congestion > self.MAX_CONGESTION:
            score -= 10
            issues.append(f"High congestion: {congestion:.0%}")
        if ai_rej_rate > self.MAX_AI_REJECTION_RATE:
            score -= 10
            issues.append(f"High AI rejection rate: {ai_rej_rate:.0%}")

        # === Determine overall status ===
        score = max(0, score)
        report.score = score
        report.issues = issues

        if score >= 80:
            report.status = HealthStatus.HEALTHY
        elif score >= 60:
            report.status = HealthStatus.DEGRADED
        elif score >= 40:
            report.status = HealthStatus.WARNING
        elif score > 0:
            report.status = HealthStatus.CRITICAL
        else:
            report.status = HealthStatus.DOWN

        return report

    def _calc_blocks_per_minute(self, now: float) -> float:
        """Calculate blocks per minute from recent data."""
        recent = [t for t in self._block_timestamps if now - t < 60]
        return len(recent)

    def _calc_avg_block_time(self) -> float:
        """Calculate average block time from recent blocks."""
        if len(self._block_timestamps) < 2:
            return 3.0  # default
        recent = self._block_timestamps[-20:]
        if len(recent) < 2:
            return 3.0
        total_time = recent[-1] - recent[0]
        return total_time / (len(recent) - 1) if len(recent) > 1 else 3.0

    def _calc_tps(self, now: float) -> float:
        """Calculate current TPS from recent transactions."""
        recent = [t for t in self._tx_timestamps if now - t < 60]
        return len(recent) / 60.0 if recent else 0.0

    def get_stats(self) -> dict:
        """Quick stats summary."""
        report = self.get_health_report()
        return report.to_dict()
