"""
Positronic - Network Time Synchronization
Ensures accurate timestamps for block production and validation.

Strategies:
1. NTP-based: Query NTP servers for network time (primary)
2. Peer-based: Cross-validate with connected peers' timestamps
3. Drift detection: Alert if local clock drifts beyond threshold

Block timestamps must be within acceptable range of network time.
"""

import time
import socket
import struct
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("positronic.utils.time_sync")


# NTP constants
NTP_PORT = 123
NTP_PACKET_FORMAT = "!12I"
NTP_DELTA = 2208988800  # NTP epoch (1900) to Unix epoch (1970) offset

# Well-known NTP servers
DEFAULT_NTP_SERVERS = [
    "pool.ntp.org",
    "time.google.com",
    "time.cloudflare.com",
    "time.windows.com",
]


@dataclass
class TimeSyncState:
    """Current time synchronization state."""
    offset_seconds: float = 0.0       # Estimated offset from true time
    last_sync: float = 0.0            # When we last synced
    sync_count: int = 0               # Number of successful syncs
    ntp_available: bool = False       # Whether NTP is reachable
    peer_offsets: List[float] = field(default_factory=list)  # Offsets from peers
    max_drift: float = 0.0            # Maximum observed drift
    synced: bool = False              # Whether we've ever synced


class NetworkTimeSynchronizer:
    """
    Maintains accurate network time for block timestamp validation.

    Usage:
        sync = NetworkTimeSynchronizer()
        sync.sync_ntp()                    # Initial NTP sync
        current_time = sync.network_time() # Get adjusted network time
        sync.validate_timestamp(ts)        # Check if a timestamp is valid
    """

    # Maximum acceptable clock drift in seconds
    MAX_CLOCK_DRIFT = 15.0  # 15 seconds (5 block times)

    # Block timestamp tolerance
    BLOCK_FUTURE_LIMIT = 15.0   # Block can be at most 15s in the future
    BLOCK_PAST_LIMIT = 3600.0   # Block can be at most 1 hour in the past

    # Sync intervals
    NTP_SYNC_INTERVAL = 600     # Re-sync NTP every 10 minutes
    PEER_SYNC_INTERVAL = 60     # Check peer times every 60 seconds

    def __init__(self, ntp_servers: List[str] = None):
        self.ntp_servers = ntp_servers or DEFAULT_NTP_SERVERS
        self.state = TimeSyncState()

    def sync_ntp(self, timeout: float = 3.0) -> bool:
        """
        Synchronize with NTP servers.
        Returns True if sync was successful.
        Safe to call; never raises.
        """
        offsets = []
        for server in self.ntp_servers:
            try:
                offset = self._query_ntp(server, timeout)
                if offset is not None:
                    offsets.append(offset)
                    if len(offsets) >= 2:
                        break  # 2 successful queries is enough
            except Exception as e:
                logger.debug(f"NTP query to {server} failed: {e}")

        if not offsets:
            logger.warning("NTP sync failed: no servers reachable")
            self.state.ntp_available = False
            return False

        # Use median offset (resistant to outliers)
        offsets.sort()
        median_offset = offsets[len(offsets) // 2]

        self.state.offset_seconds = median_offset
        self.state.last_sync = time.time()
        self.state.sync_count += 1
        self.state.ntp_available = True
        self.state.synced = True
        self.state.max_drift = max(self.state.max_drift, abs(median_offset))

        if abs(median_offset) > self.MAX_CLOCK_DRIFT:
            logger.warning(
                f"Large clock drift detected: {median_offset:.3f}s. "
                f"Consider syncing your system clock."
            )
        else:
            logger.debug(f"NTP sync: offset={median_offset:.3f}s")

        return True

    def _query_ntp(self, server: str, timeout: float) -> Optional[float]:
        """
        Query a single NTP server and return the clock offset in seconds.
        Returns None on failure.
        """
        try:
            # Build NTP request packet
            # Mode 3 (client), Version 3
            packet = b'\x1b' + 47 * b'\0'

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)

            send_time = time.time()
            sock.sendto(packet, (server, NTP_PORT))
            data, _ = sock.recvfrom(1024)
            recv_time = time.time()
            sock.close()

            if len(data) < 48:
                return None

            # Parse NTP response
            unpacked = struct.unpack(NTP_PACKET_FORMAT, data[:48])

            # Transmit timestamp is at offset 40 (words 10-11)
            ntp_time = unpacked[10] + unpacked[11] / (2**32)
            ntp_unix = ntp_time - NTP_DELTA

            # Calculate offset
            # offset = server_time - local_time (adjusted for RTT)
            rtt = recv_time - send_time
            offset = ntp_unix - recv_time + (rtt / 2)

            return offset

        except (socket.error, socket.timeout, struct.error, OSError):
            return None

    def network_time(self) -> float:
        """
        Get current network time (local time adjusted by NTP offset).
        Falls back to local time if no sync has been performed.
        """
        return time.time() + self.state.offset_seconds

    def validate_block_timestamp(
        self,
        block_timestamp: float,
        parent_timestamp: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        Validate a block timestamp.

        Rules:
        1. Must be greater than parent timestamp
        2. Must not be too far in the future
        3. Must not be too far in the past

        Returns (is_valid, error_message).
        """
        now = self.network_time()

        # Rule 1: Must be after parent
        if parent_timestamp > 0 and block_timestamp <= parent_timestamp:
            return False, (
                f"Block timestamp {block_timestamp:.0f} is not after "
                f"parent {parent_timestamp:.0f}"
            )

        # Rule 2: Not too far in the future
        if block_timestamp > now + self.BLOCK_FUTURE_LIMIT:
            return False, (
                f"Block timestamp {block_timestamp:.0f} is "
                f"{block_timestamp - now:.1f}s in the future "
                f"(limit: {self.BLOCK_FUTURE_LIMIT}s)"
            )

        # Rule 3: Not too far in the past
        if block_timestamp < now - self.BLOCK_PAST_LIMIT:
            return False, (
                f"Block timestamp {block_timestamp:.0f} is "
                f"{now - block_timestamp:.0f}s in the past "
                f"(limit: {self.BLOCK_PAST_LIMIT}s)"
            )

        return True, ""

    def update_peer_offset(self, peer_timestamp: float):
        """
        Record a timestamp from a peer for cross-validation.
        Called when receiving STATUS or HELLO messages.
        """
        local_time = time.time()
        offset = peer_timestamp - local_time
        self.state.peer_offsets.append(offset)

        # Keep only last 20 peer offsets
        if len(self.state.peer_offsets) > 20:
            self.state.peer_offsets = self.state.peer_offsets[-20:]

    def should_resync(self) -> bool:
        """Check if we need to re-sync NTP."""
        if not self.state.synced:
            return True
        elapsed = time.time() - self.state.last_sync
        return elapsed > self.NTP_SYNC_INTERVAL

    def get_stats(self) -> dict:
        """Get time synchronization stats."""
        avg_peer_offset = 0.0
        if self.state.peer_offsets:
            avg_peer_offset = sum(self.state.peer_offsets) / len(self.state.peer_offsets)

        return {
            "synced": self.state.synced,
            "ntp_available": self.state.ntp_available,
            "offset_seconds": round(self.state.offset_seconds, 4),
            "max_drift": round(self.state.max_drift, 4),
            "sync_count": self.state.sync_count,
            "last_sync_ago": round(time.time() - self.state.last_sync, 1) if self.state.last_sync > 0 else None,
            "peer_offsets_count": len(self.state.peer_offsets),
            "avg_peer_offset": round(avg_peer_offset, 4),
        }
