"""
Positronic - Peer Management
Represents and manages connections to other nodes.
Tracks WebSocket connections, peer reputation, and connection lifecycle.
"""

import time
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import IntEnum

from positronic.constants import (
    PEER_SCORE_LATENCY_WEIGHT,
    PEER_SCORE_RELIABILITY_WEIGHT,
    PEER_SCORE_BANDWIDTH_WEIGHT,
    PEER_SCORE_CHAIN_WEIGHT,
    PEER_SCORE_BEHAVIOR_WEIGHT,
    PEER_SCORE_DECAY_RATE,
)

logger = logging.getLogger("positronic.network.peer")


# ===== Phase 17: Multi-dimensional peer scoring =====

@dataclass
class PeerScoreBreakdown:
    """Multi-dimensional peer scoring for intelligent peer selection.

    Each dimension ranges from 0 to 100 (50 = neutral). The composite
    total is a weighted sum of all dimensions.

    Phase 17 GOD CHAIN addition.
    """
    latency_score: float = 50.0      # Lower latency → higher score
    reliability_score: float = 50.0  # Successful message ratio
    bandwidth_score: float = 50.0    # Data transfer throughput
    chain_score: float = 50.0        # Is the peer up-to-date?
    behavior_score: float = 50.0     # Valid messages, no spam

    @property
    def total(self) -> float:
        """Weighted composite score (0-100)."""
        return (
            self.latency_score * PEER_SCORE_LATENCY_WEIGHT
            + self.reliability_score * PEER_SCORE_RELIABILITY_WEIGHT
            + self.bandwidth_score * PEER_SCORE_BANDWIDTH_WEIGHT
            + self.chain_score * PEER_SCORE_CHAIN_WEIGHT
            + self.behavior_score * PEER_SCORE_BEHAVIOR_WEIGHT
        )

    def decay(self):
        """Gradually decay scores toward neutral (50) over time."""
        rate = PEER_SCORE_DECAY_RATE
        self.latency_score = 50.0 + (self.latency_score - 50.0) * rate
        self.reliability_score = 50.0 + (self.reliability_score - 50.0) * rate
        self.bandwidth_score = 50.0 + (self.bandwidth_score - 50.0) * rate
        self.chain_score = 50.0 + (self.chain_score - 50.0) * rate
        self.behavior_score = 50.0 + (self.behavior_score - 50.0) * rate

    def update_latency(self, latency_ms: float):
        """Update latency score based on measured latency.

        < 50ms → 100, 50-200ms → 75, 200-500ms → 50, > 500ms → 25
        """
        if latency_ms < 50:
            target = 100.0
        elif latency_ms < 200:
            target = 75.0
        elif latency_ms < 500:
            target = 50.0
        else:
            target = 25.0
        self.latency_score = 0.7 * self.latency_score + 0.3 * target

    def update_reliability(self, success: bool):
        """Update reliability score based on message success/failure."""
        delta = 2.0 if success else -5.0
        self.reliability_score = max(0.0, min(100.0, self.reliability_score + delta))

    def update_chain_sync(self, our_height: int, peer_height: int):
        """Update chain score based on how up-to-date the peer is."""
        if peer_height >= our_height:
            self.chain_score = min(100.0, self.chain_score + 3.0)
        elif our_height - peer_height < 10:
            self.chain_score = max(30.0, min(80.0, self.chain_score))
        else:
            self.chain_score = max(0.0, self.chain_score - 5.0)

    def update_behavior(self, valid_message: bool):
        """Update behavior score based on message validity."""
        delta = 1.0 if valid_message else -10.0
        self.behavior_score = max(0.0, min(100.0, self.behavior_score + delta))

    def to_dict(self) -> dict:
        return {
            "latency": round(self.latency_score, 1),
            "reliability": round(self.reliability_score, 1),
            "bandwidth": round(self.bandwidth_score, 1),
            "chain": round(self.chain_score, 1),
            "behavior": round(self.behavior_score, 1),
            "total": round(self.total, 1),
        }


class PeerState(IntEnum):
    DISCONNECTED = 0
    CONNECTING = 1
    HANDSHAKING = 2
    CONNECTED = 3
    SYNCING = 4


class ConnectionDirection(IntEnum):
    """Whether we initiated the connection or they did."""
    INBOUND = 0   # They connected to us
    OUTBOUND = 1  # We connected to them


@dataclass
class Peer:
    """Represents a connected peer node."""
    peer_id: str
    host: str
    port: int
    state: PeerState = PeerState.DISCONNECTED
    chain_height: int = 0
    best_hash: str = ""
    protocol_version: int = 1
    client_name: str = ""
    connected_at: float = 0.0
    last_seen: float = 0.0
    last_height_update: float = 0.0
    latency_ms: float = 0.0
    is_validator: bool = False
    is_nvn: bool = False  # Neural Validator Node
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    score: float = 100.0  # Reputation score (legacy, use score_breakdown.total)

    # Phase 17: Multi-dimensional scoring
    score_breakdown: PeerScoreBreakdown = field(default_factory=PeerScoreBreakdown)

    # Connection management
    direction: ConnectionDirection = ConnectionDirection.OUTBOUND
    ws: Any = None          # WebSocket connection reference (aiohttp)
    session: Any = None     # aiohttp.ClientSession for outbound connections
    listen_port: int = 0    # Port the peer listens on for inbound connections
    tls_enabled: bool = False  # Whether peer connected via TLS

    # Ping/pong tracking
    last_ping_sent: float = 0.0
    last_pong_received: float = 0.0
    ping_failures: int = 0

    @property
    def url(self) -> str:
        proto = "wss" if self.tls_enabled else "ws"
        return f"{proto}://{self.host}:{self.port}/ws"

    @property
    def listen_url(self) -> str:
        """URL based on the peer's advertised listen port."""
        p = self.listen_port if self.listen_port > 0 else self.port
        proto = "wss" if self.tls_enabled else "ws"
        return f"{proto}://{self.host}:{p}/ws"

    @property
    def age(self) -> float:
        """Connection age in seconds."""
        if self.connected_at == 0:
            return 0
        return time.time() - self.connected_at

    @property
    def is_stale(self) -> bool:
        """Check if peer hasn't been seen recently (>60 seconds)."""
        if self.last_seen == 0:
            return True
        return time.time() - self.last_seen > 60

    @property
    def is_connected(self) -> bool:
        """Check if peer is in a connected state with a live websocket."""
        if self.state not in (PeerState.CONNECTED, PeerState.SYNCING):
            return False
        if self.ws is None:
            return False
        return not getattr(self.ws, 'closed', True)

    def update_seen(self):
        self.last_seen = time.time()

    def update_latency(self, ping_time: float, pong_time: float):
        """Update latency from a ping/pong round trip."""
        rtt = (pong_time - ping_time) * 1000  # Convert to ms
        if rtt > 0:
            # Exponential moving average
            if self.latency_ms == 0:
                self.latency_ms = rtt
            else:
                self.latency_ms = 0.7 * self.latency_ms + 0.3 * rtt
            # Also update the multi-dimensional score breakdown
            self.score_breakdown.update_latency(self.latency_ms)

    def update_latency_ms(self, ms: float):
        """
        Directly update latency from a measured value in milliseconds.
        Used by latency-based peer selection (TCP probe or PING/PONG).
        """
        if ms <= 0:
            return
        if self.latency_ms == 0:
            self.latency_ms = ms
        else:
            self.latency_ms = 0.7 * self.latency_ms + 0.3 * ms
        # Update multi-dimensional scoring
        self.score_breakdown.update_latency(self.latency_ms)

    def adjust_score(self, delta: float):
        """Adjust reputation score, clamped to [0, 100]."""
        self.score = max(0.0, min(100.0, self.score + delta))

    async def send(self, data: str) -> bool:
        """Send a string message over the WebSocket."""
        if self.ws is None:
            return False
        try:
            if getattr(self.ws, 'closed', True):
                return False
            await self.ws.send_str(data)
            self.messages_sent += 1
            self.bytes_sent += len(data)
            return True
        except Exception as e:
            logger.debug(f"Send failed to {self.peer_id[:8]}: {e}")
            return False

    async def close(self, reason: str = ""):
        """Close the WebSocket connection."""
        self.state = PeerState.DISCONNECTED
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                logger.debug("WebSocket close error: %s", e)
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug("Session close error: %s", e)
        self.ws = None
        self.session = None

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "host": self.host,
            "port": self.port,
            "state": self.state.name,
            "chain_height": self.chain_height,
            "best_hash": self.best_hash,
            "latency_ms": round(self.latency_ms, 1),
            "is_validator": self.is_validator,
            "is_nvn": self.is_nvn,
            "score": round(self.score, 1),
            "direction": "inbound" if self.direction == ConnectionDirection.INBOUND else "outbound",
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "age_seconds": round(self.age, 1),
        }


class PeerManager:
    """
    Manages the set of connected peers.
    Handles peer lifecycle, reputation, and eviction.
    """

    def __init__(self, max_peers: int = 25, target_peers: int = 12):
        self.peers: Dict[str, Peer] = {}
        self.max_peers = max_peers
        self.target_peers = target_peers
        self.banned: Dict[str, float] = {}  # peer_id -> ban_until
        self._lock = asyncio.Lock()

    async def add_peer(self, peer: Peer) -> bool:
        """Add a peer. Returns False if at capacity or banned."""
        async with self._lock:
            return self._add_peer_sync(peer)

    def _add_peer_sync(self, peer: Peer) -> bool:
        """Synchronous version of add_peer (for non-async contexts)."""
        if peer.peer_id in self.banned:
            if time.time() < self.banned[peer.peer_id]:
                return False
            del self.banned[peer.peer_id]

        # max_peers=0 means unlimited — peer_security handles capacity
        if self.max_peers > 0 and len(self.peers) >= self.max_peers:
            # Try to evict worst non-validator peer
            if not self._evict_worst():
                return False

        self.peers[peer.peer_id] = peer
        return True

    # Keep backwards compatibility with sync callers
    def add_peer_sync(self, peer: Peer) -> bool:
        """Non-async add_peer for backwards compatibility."""
        return self._add_peer_sync(peer)

    def remove_peer(self, peer_id: str) -> Optional[Peer]:
        return self.peers.pop(peer_id, None)

    def get_peer(self, peer_id: str) -> Optional[Peer]:
        return self.peers.get(peer_id)

    def get_connected_peers(self) -> List[Peer]:
        return [
            p for p in self.peers.values()
            if p.state in (PeerState.CONNECTED, PeerState.SYNCING)
        ]

    def get_best_peer(self) -> Optional[Peer]:
        """Get the peer with the highest chain height."""
        connected = self.get_connected_peers()
        if not connected:
            return None
        return max(connected, key=lambda p: p.chain_height)

    def needs_more_peers(self) -> bool:
        return len(self.get_connected_peers()) < self.target_peers

    def has_peer(self, peer_id: str) -> bool:
        """Check if we already have this peer."""
        return peer_id in self.peers

    def ban_peer(self, peer_id: str, duration: float = 3600):
        """Ban a peer for a duration (default 1 hour)."""
        self.banned[peer_id] = time.time() + duration
        self.remove_peer(peer_id)

    def _evict_worst(self) -> bool:
        """Evict the worst-scoring non-validator peer."""
        if not self.peers:
            return False
        # Validators are never evicted
        candidates = [p for p in self.peers.values() if not p.is_validator]
        if not candidates:
            return False
        worst = min(candidates, key=lambda p: p.score_breakdown.total)
        if worst.score_breakdown.total < 50 or worst.score < 50:
            self.remove_peer(worst.peer_id)
            return True
        return False

    def get_peer_urls(self) -> List[str]:
        """Get URLs of all connected peers (for sharing with new peers)."""
        return [p.listen_url for p in self.get_connected_peers()]

    def get_peers_by_state(self, state: PeerState) -> List[Peer]:
        """Get all peers in a given state."""
        return [p for p in self.peers.values() if p.state == state]

    def get_inbound_peers(self) -> List[Peer]:
        """Get all inbound peers."""
        return [
            p for p in self.peers.values()
            if p.direction == ConnectionDirection.INBOUND
        ]

    def get_outbound_peers(self) -> List[Peer]:
        """Get all outbound peers."""
        return [
            p for p in self.peers.values()
            if p.direction == ConnectionDirection.OUTBOUND
        ]

    async def disconnect_peer(self, peer_id: str, reason: str = "") -> bool:
        """Gracefully disconnect a peer."""
        peer = self.get_peer(peer_id)
        if not peer:
            return False
        await peer.close(reason)
        self.remove_peer(peer_id)
        logger.info(f"Disconnected peer {peer_id[:8]}: {reason}")
        return True

    async def disconnect_all(self):
        """Disconnect all peers (for shutdown)."""
        for peer_id in list(self.peers.keys()):
            await self.disconnect_peer(peer_id, "node shutdown")

    async def prune_stale_peers(self):
        """Remove peers that haven't been seen recently."""
        stale = [
            p.peer_id for p in self.peers.values()
            if p.is_stale and p.state == PeerState.CONNECTED
        ]
        for peer_id in stale:
            await self.disconnect_peer(peer_id, "stale connection")

    def clean_expired_bans(self):
        """Remove expired bans."""
        now = time.time()
        expired = [pid for pid, until in self.banned.items() if now >= until]
        for pid in expired:
            del self.banned[pid]

    @property
    def count(self) -> int:
        return len(self.peers)

    @property
    def connected_count(self) -> int:
        return len(self.get_connected_peers())

    def get_stats(self) -> dict:
        connected = self.get_connected_peers()
        inbound = sum(1 for p in connected if p.direction == ConnectionDirection.INBOUND)
        outbound = sum(1 for p in connected if p.direction == ConnectionDirection.OUTBOUND)
        avg_latency = (
            sum(p.latency_ms for p in connected) / len(connected)
            if connected else 0
        )
        return {
            "total_peers": self.count,
            "connected": self.connected_count,
            "inbound": inbound,
            "outbound": outbound,
            "banned": len(self.banned),
            "needs_more": self.needs_more_peers(),
            "avg_latency_ms": round(avg_latency, 1),
            "max_peers": self.max_peers,
            "target_peers": self.target_peers,
        }
