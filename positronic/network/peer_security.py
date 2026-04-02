"""
Positronic - P2P Peer Security System

Multi-layer protection against Eclipse attacks, Sybil attacks,
flood attacks, and low-quality peers:

  Layer 1 — IP Rate Limiter  : 3 new connections/IP/hour, 10/subnet/hour
  Layer 2 — PoW Challenge    : SHA-256 first-byte < 0x40 (~4 attempts avg)
                               Validators are exempt from PoW
  Layer 3 — Quality Scoring  : auto-kick score<0, idle>5min, latency>8s,
                               flood>100msg/min
  Layer 4 — Validator Guard  : active staked validators score=1000,
                               are never kicked or evicted
  Layer 5 — Resource Watch   : evict worst non-validator when RAM>80%
                               or FD count exceeds limit
  Layer 6 — AI Sybil Check   : integrates SybilDetector if available

All peer capacity limits are handled here; MAX_PEERS=0 (unlimited)
in the peer manager so this layer decides who stays.
"""

import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Set

from positronic.constants import PEER_SEC_MAX_PER_IP, PEER_SEC_MAX_PER_SUBNET

logger = logging.getLogger("positronic.network.security")


# ---------------------------------------------------------------------------
# Layer 1: IP Rate Limiter
# ---------------------------------------------------------------------------

class IPRateLimiter:
    """Sliding-window rate limiter per IP and /24 subnet."""

    def __init__(
        self,
        max_per_ip: int = 3,
        max_per_subnet: int = 10,
        window: float = 3600,  # 1-hour window
    ):
        self._max_per_ip = max_per_ip
        self._max_per_subnet = max_per_subnet
        self._window = window
        self._ip_times: Dict[str, deque] = defaultdict(deque)
        self._subnet_times: Dict[str, deque] = defaultdict(deque)

    @staticmethod
    def _subnet(ip: str) -> str:
        """Return /24 prefix (e.g. '89.167.87' from '89.167.87.108')."""
        parts = ip.split(".")
        return ".".join(parts[:3]) if len(parts) >= 3 else ip

    def _prune(self, dq: deque, now: float):
        while dq and now - dq[0] > self._window:
            dq.popleft()

    def is_allowed(self, ip: str) -> bool:
        """Return True if this IP may attempt a new connection now."""
        now = time.time()
        ip_dq = self._ip_times[ip]
        self._prune(ip_dq, now)
        if len(ip_dq) >= self._max_per_ip:
            return False
        sub_dq = self._subnet_times[self._subnet(ip)]
        self._prune(sub_dq, now)
        if len(sub_dq) >= self._max_per_subnet:
            return False
        return True

    def record(self, ip: str):
        """Record a new connection attempt from this IP."""
        now = time.time()
        self._ip_times[ip].append(now)
        self._subnet_times[self._subnet(ip)].append(now)

    def cleanup(self):
        """Prune expired entries to free memory."""
        now = time.time()
        for dq in list(self._ip_times.values()):
            self._prune(dq, now)
        for dq in list(self._subnet_times.values()):
            self._prune(dq, now)
        self._ip_times = defaultdict(deque, {k: v for k, v in self._ip_times.items() if v})
        self._subnet_times = defaultdict(deque, {k: v for k, v in self._subnet_times.items() if v})


# ---------------------------------------------------------------------------
# Layer 2: PoW Challenge
# ---------------------------------------------------------------------------

class PoWChallenge:
    """
    Lightweight SHA-256 PoW gate.

    Challenge: server issues a random 16-byte nonce (hex).
    Solution:  client must find a 16-byte solution (hex) such that:
               SHA-256(nonce_bytes + solution_bytes)[0] < DIFFICULTY

    DIFFICULTY = 0x40  →  2 leading zero bits  →  ~4 attempts on average.
    Active validators are exempt (skip challenge entirely).
    Challenges expire after CHALLENGE_TTL seconds.
    Each challenge can only be used once (replay prevention).
    """

    DIFFICULTY = 0x40
    CHALLENGE_TTL = 60  # seconds

    def __init__(self):
        # nonce -> (issued_at, already_used)
        self._challenges: Dict[str, tuple] = {}

    def issue(self) -> str:
        """Issue and return a new challenge nonce (32 hex chars)."""
        nonce = os.urandom(16).hex()
        self._challenges[nonce] = (time.time(), False)
        return nonce

    def verify(self, nonce: str, solution: str) -> bool:
        """
        Verify a PoW solution. Returns True on first valid use.
        Marks the challenge as consumed (no replay).
        """
        entry = self._challenges.get(nonce)
        if entry is None:
            return False
        issued_at, used = entry
        if used:
            return False
        if time.time() - issued_at > self.CHALLENGE_TTL:
            del self._challenges[nonce]
            return False
        try:
            nonce_b = bytes.fromhex(nonce)
            sol_b = bytes.fromhex(solution)
        except (ValueError, TypeError):
            return False
        digest = hashlib.sha256(nonce_b + sol_b).digest()
        if digest[0] < self.DIFFICULTY:
            self._challenges[nonce] = (issued_at, True)
            return True
        return False

    def cleanup(self):
        """Remove expired challenges."""
        now = time.time()
        expired = [n for n, (t, _) in self._challenges.items()
                   if now - t > self.CHALLENGE_TTL * 2]
        for n in expired:
            del self._challenges[n]


# ---------------------------------------------------------------------------
# Layer 3 + 4: Per-Peer Security Record
# ---------------------------------------------------------------------------

class PeerSecurityRecord:
    """Security state for one connected peer."""

    VALIDATOR_SCORE = 1000  # Validators are immune to eviction/kick
    INITIAL_SCORE = 50      # Normal peer starting score

    def __init__(self, ip: str, is_validator: bool = False):
        self.ip = ip
        self.is_validator = is_validator
        self.score: float = self.VALIDATOR_SCORE if is_validator else self.INITIAL_SCORE
        self.connected_at = time.time()
        self.last_seen = time.time()
        # Rolling 60-second message window for flood detection
        self._msg_times: deque = deque()

    def touch(self):
        self.last_seen = time.time()

    def record_message(self):
        now = time.time()
        self._msg_times.append(now)
        # Prune messages older than 60 s
        while self._msg_times and now - self._msg_times[0] > 60:
            self._msg_times.popleft()

    @property
    def messages_per_minute(self) -> int:
        now = time.time()
        while self._msg_times and now - self._msg_times[0] > 60:
            self._msg_times.popleft()
        return len(self._msg_times)

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_seen

    def penalize(self, amount: float = 10.0):
        if not self.is_validator:
            self.score -= amount

    def reward(self, amount: float = 1.0):
        self.score = min(100.0, self.score + amount)

    def promote_validator(self):
        """Elevate peer to validator tier (called on stake detection)."""
        self.is_validator = True
        self.score = self.VALIDATOR_SCORE


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class PeerSecurityManager:
    """
    Coordinates all security layers for inbound and connected peers.

    Usage (in P2PServer):
        # On new inbound connection:
        if not security.check_ip(ip): reject()
        security.record_connection_attempt(ip)

        # After peer registered:
        security.register_peer(peer_id, ip)

        # On every received message:
        if not security.on_message(peer_id): kick(peer_id)

        # Periodically:
        for peer_id, reason in security.get_peers_to_kick(latencies):
            disconnect(peer_id, reason)

        # On disconnect:
        security.unregister_peer(peer_id)
    """

    # Kick thresholds
    SCORE_KICK_THRESHOLD = 0       # kick if score drops below 0
    LATENCY_KICK_MS = 8000         # kick if latency > 8 s
    IDLE_KICK_SECONDS = 300        # kick if idle > 5 min
    FLOOD_MSG_PER_MIN = 2000       # kick if > 2000 msg/min (high for bulk sync peers)
    BAN_DURATION = 3600            # ban IP for 1 hour after kick

    # Resource limits
    RAM_LIMIT_PERCENT = 80         # evict when RAM > 80 %
    FD_LIMIT = 50_000              # evict when open FDs > 50 000

    def __init__(self, validator_addresses: Optional[Set[str]] = None):
        self._rate_limiter = IPRateLimiter(
            max_per_ip=PEER_SEC_MAX_PER_IP,
            max_per_subnet=PEER_SEC_MAX_PER_SUBNET,
        )
        self._pow = PoWChallenge()
        self._records: Dict[str, PeerSecurityRecord] = {}   # peer_id -> record
        self._banned_ips: Dict[str, float] = {}             # ip -> ban_until
        self._validator_addresses: Set[str] = validator_addresses or set()
        self._maintenance_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start background maintenance loop (call after event loop is running)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._maintenance_task = loop.create_task(self._maintenance_loop())
        except RuntimeError:
            pass

    def stop(self):
        if self._maintenance_task:
            self._maintenance_task.cancel()

    # ------------------------------------------------------------------ #
    # Validator Set                                                        #
    # ------------------------------------------------------------------ #

    def update_validator_set(self, validator_addresses: Set[str]):
        """Update active validator set (called on every new block)."""
        self._validator_addresses = validator_addresses
        # Promote already-connected validators
        for peer_id, rec in self._records.items():
            if rec.ip in validator_addresses and not rec.is_validator:
                rec.promote_validator()
                logger.debug(
                    "Peer %s promoted to validator (score=1000)", peer_id[:8]
                )

    # ------------------------------------------------------------------ #
    # Layer 1: IP Gate                                                     #
    # ------------------------------------------------------------------ #

    def check_ip(self, ip: str) -> bool:
        """
        Return True if this IP is allowed to open a new connection.
        Checks ban list first, then rate limiter.
        """
        ban_until = self._banned_ips.get(ip, 0)
        if time.time() < ban_until:
            logger.debug("Connection from banned IP %s rejected", _mask(ip))
            return False
        if not self._rate_limiter.is_allowed(ip):
            logger.info(
                "Rate limit exceeded for %s — rejecting connection", _mask(ip)
            )
            return False
        return True

    def record_connection_attempt(self, ip: str):
        """Record that we accepted a new connection attempt from this IP."""
        self._rate_limiter.record(ip)

    # ------------------------------------------------------------------ #
    # Layer 2: PoW                                                         #
    # ------------------------------------------------------------------ #

    def issue_challenge(self) -> str:
        """Issue a PoW challenge nonce for a new inbound peer."""
        return self._pow.issue()

    def verify_challenge(self, nonce: str, solution: str) -> bool:
        """Verify a PoW challenge response."""
        return self._pow.verify(nonce, solution)

    # ------------------------------------------------------------------ #
    # Peer Registration                                                    #
    # ------------------------------------------------------------------ #

    def register_peer(self, peer_id: str, ip: str) -> PeerSecurityRecord:
        """Register a newly admitted peer."""
        is_validator = ip in self._validator_addresses
        rec = PeerSecurityRecord(ip=ip, is_validator=is_validator)
        self._records[peer_id] = rec
        if is_validator:
            logger.info(
                "Validator peer %s registered (score=1000, exempt from kick)",
                peer_id[:8],
            )
        return rec

    def unregister_peer(self, peer_id: str):
        """Remove peer tracking on disconnect."""
        self._records.pop(peer_id, None)

    # ------------------------------------------------------------------ #
    # Layer 3: Per-Message Flood Check                                    #
    # ------------------------------------------------------------------ #

    def on_message(self, peer_id: str) -> bool:
        """
        Called for every incoming message from a peer.
        Returns False if the peer is flooding and should be kicked.
        Validators always pass.
        """
        rec = self._records.get(peer_id)
        if not rec:
            return True
        rec.touch()
        rec.record_message()
        if rec.is_validator:
            return True
        if rec.messages_per_minute > self.FLOOD_MSG_PER_MIN:
            logger.warning(
                "Peer %s flooding (%d msg/min) — kicking and banning IP",
                peer_id[:8], rec.messages_per_minute,
            )
            self._ban_ip(rec.ip)
            rec.score = -1  # force-kick flag
            return False
        return True

    # ------------------------------------------------------------------ #
    # Score Adjustments                                                    #
    # ------------------------------------------------------------------ #

    def penalize(self, peer_id: str, amount: float = 10.0, reason: str = ""):
        rec = self._records.get(peer_id)
        if rec:
            rec.penalize(amount)
            if reason:
                logger.debug(
                    "Peer %s penalized %.0f (%s) → score=%.0f",
                    peer_id[:8], amount, reason, rec.score,
                )

    def reward(self, peer_id: str, amount: float = 1.0):
        rec = self._records.get(peer_id)
        if rec:
            rec.reward(amount)

    # ------------------------------------------------------------------ #
    # Layer 3: Kick Evaluation (called periodically)                      #
    # ------------------------------------------------------------------ #

    def get_peers_to_kick(
        self, peer_latencies: Optional[Dict[str, float]] = None
    ) -> list:
        """
        Return [(peer_id, reason), ...] for peers that should be disconnected.

        Criteria (validators are always exempt):
          - score < 0
          - latency > LATENCY_KICK_MS
          - idle > IDLE_KICK_SECONDS
        """
        peer_latencies = peer_latencies or {}
        to_kick = []

        for peer_id, rec in list(self._records.items()):
            if rec.is_validator:
                continue

            # Score-based kick
            if rec.score < self.SCORE_KICK_THRESHOLD:
                to_kick.append((peer_id, f"score={rec.score:.0f}"))
                continue

            # Latency-based kick (only penalise, kick only if score drops)
            latency = peer_latencies.get(peer_id, 0)
            if latency > self.LATENCY_KICK_MS:
                rec.penalize(20)
                if rec.score < self.SCORE_KICK_THRESHOLD:
                    to_kick.append((peer_id, f"latency={latency:.0f}ms"))
                continue

            # Idle kick
            if rec.idle_seconds > self.IDLE_KICK_SECONDS:
                to_kick.append((peer_id, f"idle={rec.idle_seconds:.0f}s"))

        return to_kick

    # ------------------------------------------------------------------ #
    # Layer 5: Resource Eviction                                          #
    # ------------------------------------------------------------------ #

    def get_lowest_score_peer(self) -> Optional[str]:
        """Return peer_id of the lowest-scoring non-validator (for resource eviction)."""
        candidates = [
            (pid, rec.score)
            for pid, rec in self._records.items()
            if not rec.is_validator
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    def needs_resource_eviction(self) -> bool:
        """True if RAM or FD count is above the configured limit."""
        try:
            if os.path.exists("/proc/self/fd"):
                fd_count = len(os.listdir("/proc/self/fd"))
                if fd_count > self.FD_LIMIT:
                    return True
        except (PermissionError, FileNotFoundError):
            pass
        try:
            import psutil
            if psutil.virtual_memory().percent > self.RAM_LIMIT_PERCENT:
                return True
        except ImportError:
            pass
        return False

    # ------------------------------------------------------------------ #
    # Layer 6: AI Sybil Check                                             #
    # ------------------------------------------------------------------ #

    def run_sybil_check(self, peer_id: str, peer_info: dict) -> bool:
        """
        Run AI Sybil detection if available.
        Returns True if peer passes (or AI unavailable).
        """
        try:
            from positronic.ai.anomaly import SybilDetector
            result = SybilDetector.on_peer_connection(peer_info)
            if result and result.get("is_sybil"):
                confidence = result.get("confidence", 0)
                logger.warning(
                    "AI flagged peer %s as Sybil (confidence=%.2f) — penalising",
                    peer_id[:8], confidence,
                )
                rec = self._records.get(peer_id)
                if rec:
                    rec.penalize(50.0 * confidence)
                    return rec.score >= self.SCORE_KICK_THRESHOLD
        except (ImportError, Exception):
            pass
        return True

    # ------------------------------------------------------------------ #
    # IP Ban Helpers                                                       #
    # ------------------------------------------------------------------ #

    def ban_ip(self, ip: str, duration: float = BAN_DURATION):
        self._ban_ip(ip, duration)

    def is_banned_ip(self, ip: str) -> bool:
        ban_until = self._banned_ips.get(ip, 0)
        if time.time() < ban_until:
            return True
        if ip in self._banned_ips:
            del self._banned_ips[ip]
        return False

    def _ban_ip(self, ip: str, duration: float = BAN_DURATION):
        self._banned_ips[ip] = time.time() + duration
        logger.info("IP %s banned for %ds", _mask(ip), int(duration))

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        validators = sum(1 for r in self._records.values() if r.is_validator)
        avg_score = (
            sum(r.score for r in self._records.values()) / len(self._records)
            if self._records else 0
        )
        return {
            "tracked_peers": len(self._records),
            "validators": validators,
            "banned_ips": len(self._banned_ips),
            "avg_score": round(avg_score, 1),
        }

    # ------------------------------------------------------------------ #
    # Background Maintenance                                               #
    # ------------------------------------------------------------------ #

    async def _maintenance_loop(self):
        """Runs every 5 minutes to clean up expired bans and PoW nonces."""
        while True:
            try:
                await asyncio.sleep(300)
                now = time.time()
                # Expire bans
                expired = [ip for ip, until in self._banned_ips.items() if now >= until]
                for ip in expired:
                    del self._banned_ips[ip]
                # Clean rate limiter and PoW nonces
                self._rate_limiter.cleanup()
                self._pow.cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Security maintenance error: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask(ip: str) -> str:
    """Mask last two octets of an IPv4 address for log safety."""
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.x.x"
    return ip
