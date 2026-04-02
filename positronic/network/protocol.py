"""
Positronic - Wire Protocol
Handles message routing, protocol state machine, rate limiting,
and per-peer protocol enforcement.
"""

import asyncio
import time
import logging
from enum import IntEnum
from typing import Callable, Dict, Optional, List, Set
from dataclasses import dataclass, field

from positronic.network.messages import (
    MessageType, NetworkMessage, MessageIDCache,
    HANDSHAKE_MESSAGES, BLOCK_MESSAGES, TX_MESSAGES,
    SYNC_MESSAGES, HEALTH_MESSAGES,
)
from positronic.constants import PROTOCOL_VERSION, CHAIN_ID, NETWORK_MAGIC

logger = logging.getLogger("positronic.network.protocol")


class PeerProtocolState(IntEnum):
    """Protocol state for each peer connection."""
    NEW = 0               # Just connected, no handshake yet
    HELLO_SENT = 1        # We sent HELLO, waiting for HELLO_ACK
    HELLO_RECEIVED = 2    # We received HELLO, sent HELLO_ACK
    ACTIVE = 3            # Handshake complete, normal operation
    SYNCING = 4           # Currently syncing blocks from this peer
    DISCONNECTING = 5     # Graceful disconnect in progress


@dataclass
class PeerProtocolInfo:
    """Per-peer protocol tracking."""
    peer_id: str
    state: PeerProtocolState = PeerProtocolState.NEW
    protocol_version: int = 0
    chain_id: int = 0
    remote_height: int = 0
    remote_best_hash: str = ""
    client_name: str = ""
    listen_port: int = 0

    # Handshake timing
    handshake_started: float = 0.0
    handshake_completed: float = 0.0

    # Rate limiting
    message_counts: Dict[int, int] = field(default_factory=dict)  # msg_type -> count
    rate_window_start: float = 0.0

    # Scoring
    good_messages: int = 0
    bad_messages: int = 0

    @property
    def is_handshaked(self) -> bool:
        return self.state == PeerProtocolState.ACTIVE

    @property
    def behavior_score(self) -> float:
        """Score from 0 to 100 based on message behavior."""
        total = self.good_messages + self.bad_messages
        if total == 0:
            return 100.0
        return (self.good_messages / total) * 100


# Base rate limits: (max_count, window_seconds) - adjusted dynamically
BASE_RATE_LIMITS: Dict[MessageType, tuple] = {
    MessageType.PING: (10, 60),         # 10 pings per minute
    MessageType.GET_PEERS: (10, 60),    # 10 peer requests per minute
    MessageType.GET_BLOCKS: (50, 60),   # 50 block requests per minute
    MessageType.GET_HEADERS: (50, 60),  # 50 header requests per minute
    MessageType.NEW_TX: (500, 60),      # 500 transactions per minute (higher for throughput)
    MessageType.NEW_BLOCK: (60, 60),    # 60 blocks per minute
    MessageType.STATUS: (300, 60),      # 300 status per minute (syncing peers send rapid updates)
    MessageType.REQUEST_STATUS: (60, 60),  # 60 requests per minute (for re-sync)
    MessageType.SYNC_REQUEST: (200, 60), # 200 sync requests per minute (bulk sync needs many batches)
}

# Keep backward compat alias
RATE_LIMITS = BASE_RATE_LIMITS

# Handshake timeout in seconds
HANDSHAKE_TIMEOUT = 15.0


class DynamicRateLimiter:
    """
    Dynamic rate limiter that adjusts based on:
    1. Network congestion level (higher congestion = stricter limits)
    2. Peer reputation (higher rep = more lenient)
    3. Current sync state (syncing = more lenient for block messages)

    Superior to Bitcoin's static rate limiting.
    """

    # Congestion thresholds
    LOW_CONGESTION = 0.3
    HIGH_CONGESTION = 0.7

    def __init__(self):
        self._congestion_level: float = 0.0  # 0.0-1.0
        self._is_syncing: bool = False
        self._peer_reputation: Dict[str, float] = {}  # peer_id -> reputation (0-100)

    def set_congestion(self, level: float):
        """Update network congestion level (0.0 = idle, 1.0 = full)."""
        self._congestion_level = max(0.0, min(1.0, level))

    def set_syncing(self, syncing: bool):
        """Update sync state."""
        self._is_syncing = syncing

    def set_peer_reputation(self, peer_id: str, reputation: float):
        """Update peer reputation score."""
        self._peer_reputation[peer_id] = max(0.0, min(100.0, reputation))

    def get_limit(
        self,
        msg_type: MessageType,
        peer_id: str = "",
    ) -> tuple:
        """
        Get the dynamic rate limit for a message type and peer.
        Returns (max_count, window_seconds).
        """
        base = BASE_RATE_LIMITS.get(msg_type)
        if not base:
            return (100, 60)  # Default if no base limit

        base_count, window = base
        multiplier = 1.0

        # Adjust for congestion
        if self._congestion_level > self.HIGH_CONGESTION:
            # High congestion: reduce limits to 50%
            multiplier *= 0.5
        elif self._congestion_level < self.LOW_CONGESTION:
            # Low congestion: increase limits by 50%
            multiplier *= 1.5

        # Adjust for peer reputation
        rep = self._peer_reputation.get(peer_id, 50.0)
        if rep >= 80:
            multiplier *= 1.5  # Trusted peers get 50% more
        elif rep < 20:
            multiplier *= 0.3  # Untrusted peers get 70% less

        # Adjust for syncing state
        if self._is_syncing:
            if msg_type in (MessageType.GET_BLOCKS, MessageType.GET_HEADERS,
                          MessageType.SYNC_REQUEST, MessageType.STATUS,
                          MessageType.REQUEST_STATUS):
                multiplier *= 3.0  # 3x for sync-related during sync

        adjusted = max(1, int(base_count * multiplier))
        return (adjusted, window)

    def check_limit(
        self,
        peer_id: str,
        msg_type: MessageType,
        current_count: int,
    ) -> bool:
        """Check if a message is within the dynamic rate limit."""
        limit, _ = self.get_limit(msg_type, peer_id)
        return current_count < limit

    def remove_peer(self, peer_id: str):
        """Remove peer reputation data."""
        self._peer_reputation.pop(peer_id, None)

    def get_stats(self) -> dict:
        return {
            "congestion_level": round(self._congestion_level, 3),
            "is_syncing": self._is_syncing,
            "tracked_peers": len(self._peer_reputation),
        }


class ProtocolHandler:
    """
    Routes incoming messages to registered handlers.
    Manages per-peer protocol state, handshake flow,
    rate limiting, and message deduplication.
    """

    def __init__(self, node_id: str = ""):
        self.node_id = node_id
        self._handlers: Dict[MessageType, Callable] = {}
        self._peer_states: Dict[str, PeerProtocolInfo] = {}
        self._msg_cache = MessageIDCache(max_size=10_000)
        self.rate_limiter = DynamicRateLimiter()

        # Statistics
        self._total_received = 0
        self._total_dropped = 0
        self._total_duplicates = 0

    def register(self, msg_type: MessageType, handler: Callable):
        """Register a handler for a message type."""
        self._handlers[msg_type] = handler

    def register_many(self, handlers: Dict[MessageType, Callable]):
        """Register multiple handlers at once."""
        self._handlers.update(handlers)

    def get_peer_state(self, peer_id: str) -> PeerProtocolInfo:
        """Get or create protocol state for a peer."""
        if peer_id not in self._peer_states:
            self._peer_states[peer_id] = PeerProtocolInfo(
                peer_id=peer_id,
                rate_window_start=time.time(),
            )
        return self._peer_states[peer_id]

    def set_peer_active(self, peer_id: str, info: dict):
        """Mark a peer as having completed the handshake."""
        state = self.get_peer_state(peer_id)
        state.state = PeerProtocolState.ACTIVE
        state.protocol_version = info.get("protocol_version", 1)
        state.chain_id = info.get("chain_id", CHAIN_ID)
        state.remote_height = info.get("height", 0)
        state.remote_best_hash = info.get("best_hash", "")
        state.client_name = info.get("client", "")
        state.listen_port = info.get("listen_port", 0)
        state.handshake_completed = time.time()

    def remove_peer(self, peer_id: str):
        """Remove peer protocol state."""
        self._peer_states.pop(peer_id, None)

    def update_peer_height(self, peer_id: str, height: int, best_hash: str = ""):
        """Update a peer's known chain height."""
        state = self.get_peer_state(peer_id)
        state.remote_height = height
        if best_hash:
            state.remote_best_hash = best_hash

    async def handle_message(
        self, message: NetworkMessage
    ) -> Optional[NetworkMessage]:
        """
        Process an incoming message through the full protocol pipeline:
        1. Deduplication check
        2. Expiry check
        3. Protocol state validation
        4. Rate limiting
        5. Handler dispatch

        Returns an optional response message.
        """
        self._total_received += 1
        peer_id = message.sender_id

        # 1. Message deduplication
        if not self._msg_cache.add_and_check(message.msg_id):
            self._total_duplicates += 1
            return None

        # 2. Expiry check (skip for handshake messages)
        if message.msg_type not in HANDSHAKE_MESSAGES and message.is_expired:
            self._total_dropped += 1
            return None

        # 3. Protocol state validation
        peer_state = self.get_peer_state(peer_id)

        if message.msg_type not in HANDSHAKE_MESSAGES:
            # Non-handshake messages require completed handshake
            if not peer_state.is_handshaked:
                # Allow PING/PONG during handshake for keepalive
                if message.msg_type not in HEALTH_MESSAGES:
                    self._total_dropped += 1
                    peer_state.bad_messages += 1
                    logger.debug(
                        f"Dropped {message.msg_type.name} from "
                        f"{peer_id[:8]}: not handshaked"
                    )
                    return None

        # 4. Rate limiting
        if not self._check_rate_limit(peer_id, message.msg_type):
            self._total_dropped += 1
            peer_state.bad_messages += 1
            logger.debug(
                f"Rate limited {message.msg_type.name} from {peer_id[:8]}"
            )
            return None

        # 5. Dispatch to handler
        handler = self._handlers.get(message.msg_type)
        if handler:
            peer_state.good_messages += 1
            try:
                result = handler(message)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                peer_state.bad_messages += 1
                logger.error(
                    f"Handler error for {message.msg_type.name}: {e}"
                )
                return None

        logger.debug(f"No handler for {message.msg_type.name}")
        return None

    def _check_rate_limit(self, peer_id: str, msg_type: MessageType) -> bool:
        """
        Check if a peer is within rate limits for a message type.
        Uses dynamic rate limiting based on congestion and peer reputation.
        Returns True if the message is allowed, False if rate limited.
        """
        # Get dynamic limit for this peer and message type
        max_count, window_secs = self.rate_limiter.get_limit(msg_type, peer_id)

        state = self.get_peer_state(peer_id)
        now = time.time()

        # Reset window if expired
        if now - state.rate_window_start > window_secs:
            state.message_counts.clear()
            state.rate_window_start = now

        # Count messages of this type
        type_key = int(msg_type)
        count = state.message_counts.get(type_key, 0)
        if count >= max_count:
            return False

        state.message_counts[type_key] = count + 1
        return True

    def validate_hello(self, payload: dict) -> tuple:
        """
        Validate a HELLO message payload.
        Returns (is_valid, error_reason).
        """
        # Check protocol version
        version = payload.get("protocol_version", 0)
        if version < 1:
            return False, f"Unsupported protocol version: {version}"

        # Check chain ID
        chain_id = payload.get("chain_id", 0)
        if chain_id != CHAIN_ID:
            return False, f"Chain ID mismatch: expected {CHAIN_ID}, got {chain_id}"

        # Check node ID
        node_id = payload.get("node_id", "")
        if not node_id:
            return False, "Missing node_id"

        # Don't connect to ourselves
        if node_id == self.node_id:
            return False, "Self-connection detected"

        return True, ""

    def has_handler(self, msg_type: MessageType) -> bool:
        return msg_type in self._handlers

    def get_active_peers(self) -> List[str]:
        """Get peer IDs that have completed the handshake."""
        return [
            pid for pid, state in self._peer_states.items()
            if state.is_handshaked
        ]

    def get_best_sync_peer(self) -> Optional[str]:
        """Get the peer with the highest chain height."""
        best_id = None
        best_height = -1
        for pid, state in self._peer_states.items():
            if state.is_handshaked and state.remote_height > best_height:
                best_height = state.remote_height
                best_id = pid
        return best_id

    def get_stats(self) -> dict:
        return {
            "total_received": self._total_received,
            "total_dropped": self._total_dropped,
            "total_duplicates": self._total_duplicates,
            "tracked_peers": len(self._peer_states),
            "active_peers": len(self.get_active_peers()),
            "msg_cache_size": self._msg_cache.size,
            "handlers_registered": len(self._handlers),
        }

    def cleanup(self):
        """Periodic cleanup of stale data."""
        self._msg_cache.prune_expired()
        # Remove protocol state for peers that have been disconnected
        stale = [
            pid for pid, state in self._peer_states.items()
            if state.state == PeerProtocolState.DISCONNECTING
        ]
        for pid in stale:
            del self._peer_states[pid]
