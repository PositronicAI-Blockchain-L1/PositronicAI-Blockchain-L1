"""
Positronic - Peer Discovery
Bootstrap-based peer discovery with periodic peer exchange,
address reputation tracking, and connection management.
Supports optional SQLite persistence of the address book.
"""

import asyncio
import random
import socket
import sqlite3
import time
import logging
from typing import List, Set, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

from positronic.network.peer import Peer, PeerManager, PeerState
from positronic.constants import DEFAULT_P2P_PORT, MIN_PEERS, TARGET_PEERS

# Lazy import to avoid hard dependency on NAT libraries
# from positronic.network.nat import NATTraversal

logger = logging.getLogger("positronic.network.discovery")


@dataclass
class AddressInfo:
    """Tracking info for a discovered peer address."""
    url: str
    last_seen: float = 0.0
    last_attempt: float = 0.0
    failures: int = 0
    successes: int = 0
    source: str = "bootstrap"  # bootstrap, peer_exchange, manual
    latency_ms: float = 999.0  # Measured latency; 999 = unknown
    latency_measured_at: float = 0.0  # When latency was last measured

    @property
    def score(self) -> float:
        """Connection reliability score (0-1)."""
        total = self.failures + self.successes
        if total == 0:
            return 0.5  # Unknown
        return self.successes / total

    @property
    def backoff_seconds(self) -> float:
        """Exponential backoff based on failure count."""
        if self.failures == 0:
            return 0
        return min(300, 5 * (2 ** min(self.failures, 6)))

    @property
    def can_retry(self) -> bool:
        """Check if enough time has passed since last attempt."""
        if self.last_attempt == 0:
            return True
        return time.time() - self.last_attempt > self.backoff_seconds


class PeerDiscovery:
    """
    Discovers peers through bootstrap nodes and peer exchange.
    Manages address book with reputation and retry logic.

    Bootstrap nodes are the initial entry points for new nodes joining
    the network. They should be well-known, reliable nodes.
    """

    # Official Positronic seed nodes
    # These are long-lived infrastructure nodes that serve as entry points.
    # They can be DNS names (auto-resolve) or direct IP addresses.
    MAINNET_SEEDS = [
        # Official seed nodes — TLS enforced (wss://)
        f"wss://seed1.positronic-ai.network:{DEFAULT_P2P_PORT}/ws",
        f"wss://seed2.positronic-ai.network:{DEFAULT_P2P_PORT}/ws",
        f"wss://seed3.positronic-ai.network:{DEFAULT_P2P_PORT}/ws",
    ]

    TESTNET_SEEDS = [
        # Hetzner infrastructure (Helsinki / Nuremberg)
        # seed-1 requires TLS (wss://); val-1 and rpc-1 use plain ws://
        f"wss://89.167.87.108:{DEFAULT_P2P_PORT}/ws",  # seed node (TLS)
        f"ws://87.99.139.220:{DEFAULT_P2P_PORT}/ws",   # validator
        f"ws://89.167.96.119:{DEFAULT_P2P_PORT}/ws",   # rpc node
    ]

    # Fallback: localhost for single-node development
    LOCAL_SEEDS = [
        f"ws://localhost:{DEFAULT_P2P_PORT}/ws",
    ]

    def __init__(
        self,
        peer_manager: PeerManager,
        bootstrap_nodes: List[str] = None,
        network_type: str = "local",
        db_path: Optional[str] = None,
        security_manager=None,  # PeerSecurityManager (optional)
    ):
        self.peer_manager = peer_manager
        self.network_type = network_type
        self._security_manager = security_manager

        # Use provided bootstrap nodes, or select based on network type
        if bootstrap_nodes:
            self.bootstrap_nodes = bootstrap_nodes
        elif network_type == "mainnet":
            self.bootstrap_nodes = self.MAINNET_SEEDS
        elif network_type == "testnet":
            self.bootstrap_nodes = self.TESTNET_SEEDS
        else:
            self.bootstrap_nodes = self.LOCAL_SEEDS

        self.discovery_interval = 30  # seconds
        self.peer_exchange_interval = 60  # seconds

        # Optional SQLite persistence for address book
        self._db: Optional[sqlite3.Connection] = None
        self._db_path = db_path
        if db_path:
            self._init_db(db_path)

        # Address book with tracking info
        self._addresses: Dict[str, AddressInfo] = {}
        for addr in self.bootstrap_nodes:
            self._addresses[addr] = AddressInfo(
                url=addr, source="bootstrap"
            )

        # Load persisted peers from DB (if available)
        if self._db:
            self._load_from_db()

        # NAT traversal (set up via setup_nat())
        self.nat: Optional[object] = None

        # Our own listen URL (set when server starts)
        self._own_url: Optional[str] = None

        # All IPs that belong to this node (to prevent self-connection via NAT)
        self._own_ips: Set[str] = set()
        self._detect_local_ips()

        # Callback for initiating connections
        self._connect_callback: Optional[Callable] = None

        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._exchange_task: Optional[asyncio.Task] = None
        self._latency_task: Optional[asyncio.Task] = None
        self._running = False

        # Latency measurement settings
        self.latency_measurement_interval = 60  # Re-measure every 60s
        self.latency_measurement_timeout = 5.0  # 5s timeout per probe
        self.max_preferred_peers = 5  # Connect to top-N closest peers
        self.latency_cache_ttl = 120  # Cache measurements for 2 minutes

    # ------------------------------------------------------------------ #
    #  Latency-Based Peer Selection                                       #
    # ------------------------------------------------------------------ #

    async def measure_latency(self, peer_address: str) -> float:
        """
        Measure TCP-level round-trip latency to a peer address.
        Performs a TCP connect to the peer's host:port and measures the
        time taken. Returns latency in milliseconds, or 999.0 on failure.
        """
        try:
            host, port = self._parse_host_port(peer_address)
            if not host or not port:
                return 999.0

            start = time.monotonic()
            # Use asyncio to avoid blocking the event loop
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.latency_measurement_timeout,
            )
            elapsed = (time.monotonic() - start) * 1000  # ms
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return round(elapsed, 2)
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError, Exception) as e:
            logger.debug(f"Latency probe failed for {peer_address}: {e}")
            return 999.0

    def _parse_host_port(self, url: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract host and port from a ws:// or wss:// URL."""
        try:
            if "://" in url:
                hostport = url.split("://")[1].split("/")[0]
                parts = hostport.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else DEFAULT_P2P_PORT
                return host, port
        except (ValueError, IndexError):
            pass
        return None, None

    async def measure_latency_cached(self, peer_address: str) -> float:
        """
        Return cached latency if fresh, otherwise measure and cache.
        """
        info = self._addresses.get(peer_address)
        if info and info.latency_measured_at > 0:
            age = time.time() - info.latency_measured_at
            if age < self.latency_cache_ttl:
                return info.latency_ms

        latency = await self.measure_latency(peer_address)

        # Store in address book
        if peer_address in self._addresses:
            self._addresses[peer_address].latency_ms = latency
            self._addresses[peer_address].latency_measured_at = time.time()

        return latency

    async def sort_peers_by_latency(
        self, peer_list: List[str], max_concurrent: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Measure latency to all peers in the list and return them sorted
        by latency (lowest first). Measurements run concurrently.

        Returns list of (url, latency_ms) tuples.
        """
        if not peer_list:
            return []

        # Measure concurrently with a semaphore to limit parallelism
        sem = asyncio.Semaphore(max_concurrent)

        async def _measure(url: str) -> Tuple[str, float]:
            async with sem:
                latency = await self.measure_latency_cached(url)
                return (url, latency)

        results = await asyncio.gather(
            *[_measure(url) for url in peer_list],
            return_exceptions=True,
        )

        # Filter out exceptions and sort by latency
        measured = []
        for r in results:
            if isinstance(r, tuple):
                measured.append(r)
            # Exceptions are silently dropped (peer gets default 999ms)

        measured.sort(key=lambda x: x[1])
        return measured

    def _detect_local_ips(self):
        """Detect all IPs on local network interfaces to prevent self-connection."""
        # Collect interface IPs
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                self._own_ips.add(info[4][0])
        except Exception:
            pass
        # Primary outbound IP (LAN IP)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            self._own_ips.add(s.getsockname()[0])
            s.close()
        except Exception:
            pass
        self._own_ips.discard("0.0.0.0")
        if self._own_ips:
            logger.debug(f"Local IPs detected: {self._own_ips}")

    def add_own_ip(self, ip: str):
        """Add an externally-discovered IP to the self-filter list."""
        if ip and ip not in ("0.0.0.0", "127.0.0.1", "::1", "localhost"):
            self._own_ips.add(ip)
            logger.debug(f"Added own external IP: {ip}")

    async def detect_external_ip(self):
        """Best-effort detection of external/public IP via STUN or HTTP."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    "https://api.ipify.org",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        ip = (await resp.text()).strip()
                        self.add_own_ip(ip)
                        logger.info(f"External IP detected: {ip}")
                        return ip
        except Exception as e:
            logger.debug(f"External IP detection failed: {e}")
        return None

    def set_own_url(self, url: str):
        """Set our own listen URL so we don't try to connect to ourselves."""
        self._own_url = url
        # Extract hostname and add to own IPs set
        try:
            parsed = urlparse(url)
            if parsed.hostname and parsed.hostname not in ("0.0.0.0", "localhost", "::1"):
                self._own_ips.add(parsed.hostname)
        except Exception:
            pass

    def set_connect_callback(self, callback: Callable):
        """Set callback function for connecting to peers."""
        self._connect_callback = callback

    async def setup_nat(self, port: int):
        """Attempt NAT traversal for inbound connections."""
        from positronic.network.nat import NATTraversal
        self.nat = NATTraversal(port=port)
        result = await self.nat.setup()
        if result:
            self.set_own_url(f"ws://{result.ip}:{result.port}/ws")
        return result

    @property
    def known_addresses(self) -> Set[str]:
        """All known addresses."""
        return set(self._addresses.keys())

    def add_known_address(self, address: str, source: str = "peer_exchange"):
        """Add an address to the book."""
        if address == self._own_url:
            return
        # Filter out unroutable/self addresses
        _skip = ("0.0.0.0", "127.0.0.1", "localhost", "::1")
        for s in _skip:
            if s in address:
                return
        # Also filter own URL variants (ws vs wss)
        if self._own_url:
            own_stripped = self._own_url.replace("wss://", "").replace("ws://", "")
            addr_stripped = address.replace("wss://", "").replace("ws://", "")
            if own_stripped == addr_stripped:
                return
        # Check if hostname matches any known own IPs (prevents self-connection via NAT)
        try:
            parsed = urlparse(address)
            if parsed.hostname and parsed.hostname in self._own_ips:
                logger.debug(f"Skipping own-IP address: {address}")
                return
        except Exception:
            pass
        if address not in self._addresses:
            self._addresses[address] = AddressInfo(
                url=address, source=source, last_seen=time.time()
            )
        else:
            self._addresses[address].last_seen = time.time()

    def add_known_addresses(self, addresses: List[str], source: str = "peer_exchange"):
        """Add multiple addresses."""
        for addr in addresses:
            self.add_known_address(addr, source)

    def on_connect_success(self, url: str):
        """Record a successful connection to an address."""
        if url in self._addresses:
            info = self._addresses[url]
            info.successes += 1
            info.last_seen = time.time()
            info.failures = max(0, info.failures - 1)  # Forgive past failures

    def on_connect_failure(self, url: str):
        """Record a failed connection attempt."""
        if url in self._addresses:
            info = self._addresses[url]
            info.failures += 1
            info.last_attempt = time.time()

    def get_peers_to_connect(self, max_count: int = 0) -> List[str]:
        """
        Get addresses we should try to connect to.
        Prioritizes by: bootstrap > high score > least recent attempt.
        Respects backoff timing for failed addresses.
        """
        if not self.peer_manager.needs_more_peers():
            return []

        connected_peers = self.peer_manager.get_connected_peers()
        connected_urls = set(p.url for p in connected_peers)
        connected_ids = set(p.peer_id for p in connected_peers)
        # Also track connected hostnames to avoid duplicate connections
        # to the same server via different URL schemes (ws:// vs wss://)
        connected_hosts = set()
        for p in connected_peers:
            try:
                h = urlparse(p.url).hostname
                if h:
                    connected_hosts.add(h)
            except Exception:
                pass
            if hasattr(p, 'listen_url') and p.listen_url:
                try:
                    h = urlparse(p.listen_url).hostname
                    if h:
                        connected_hosts.add(h)
                except Exception:
                    pass

        candidates = []
        for url, info in self._addresses.items():
            # Skip if already connected (by URL or hostname)
            if url in connected_urls:
                continue
            try:
                addr_host = urlparse(url).hostname
                if addr_host and addr_host in connected_hosts:
                    continue
            except Exception:
                pass
            # Skip our own URL
            if url == self._own_url:
                continue
            # Skip if backoff not expired
            if not info.can_retry:
                continue
            # Skip addresses that always fail — but never skip bootstrap/seed nodes
            if info.failures >= 3 and info.successes == 0 and info.source != "bootstrap":
                continue
            # Skip IPs banned by the security manager
            if self._security_manager:
                try:
                    parsed = urlparse(url)
                    host = parsed.hostname or ""
                    if host and self._security_manager.is_banned_ip(host):
                        continue
                except Exception:
                    pass

            candidates.append((url, info))

        # Sort: bootstrap first, then by latency (lower is better),
        # then by score (descending), then by least recent attempt.
        # Latency-based selection: peers with measured latency < 999
        # are preferred over unmeasured peers.
        candidates.sort(key=lambda x: (
            -(x[1].source == "bootstrap"),  # Bootstrap first (always keep)
            x[1].latency_ms,                # Lower latency preferred
            -x[1].score,                     # Higher score first
            x[1].last_attempt,               # Least recent attempt first
        ))

        needed = self.peer_manager.target_peers - self.peer_manager.connected_count
        if max_count > 0:
            needed = min(needed, max_count)

        result = [url for url, _ in candidates[:needed]]
        # Mark attempt time
        for url in result:
            self._addresses[url].last_attempt = time.time()

        return result

    def get_shareable_peers(self, max_count: int = 10) -> List[str]:
        """Get peer addresses to share with other peers."""
        urls = self.peer_manager.get_peer_urls()
        if self._own_url:
            urls.append(self._own_url)
        random.shuffle(urls)
        return urls[:max_count]

    def resolve_dns_seeds(self):
        """
        Resolve DNS seed hostnames to IP addresses.
        This allows seed nodes to use DNS names that can be updated
        without changing the client code. Safe to call; failures are logged.
        """
        import socket
        resolved_count = 0
        for addr in list(self._addresses.keys()):
            info = self._addresses[addr]
            if info.source != "bootstrap":
                continue
            try:
                # Extract hostname from URL
                if "://" in addr:
                    hostport = addr.split("://")[1].split("/")[0]
                    host = hostport.split(":")[0]
                    # Only resolve if it's not already an IP
                    try:
                        socket.inet_aton(host)
                        continue  # Already an IP
                    except socket.error:
                        pass
                    # Resolve DNS
                    ips = socket.getaddrinfo(host, None, socket.AF_INET)
                    if ips:
                        resolved_count += 1
                        logger.debug(f"Resolved seed {host} -> {ips[0][4][0]}")
            except (socket.gaierror, IndexError, ValueError):
                logger.debug(f"DNS resolution failed for seed: {addr}")
        if resolved_count > 0:
            logger.info(f"Resolved {resolved_count} DNS seed nodes")

    async def start(self):
        """Start background discovery tasks."""
        self._running = True

        # Try to resolve DNS seeds (non-blocking, best-effort)
        try:
            self.resolve_dns_seeds()
        except Exception as e:
            logger.debug(f"DNS seed resolution failed: {e}")

        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._exchange_task = asyncio.create_task(self._peer_exchange_loop())
        self._latency_task = asyncio.create_task(self._latency_measurement_loop())
        logger.info(
            f"Peer discovery started ({self.network_type}) "
            f"with {len(self._addresses)} known addresses, "
            f"latency-based selection enabled"
        )

    async def stop(self):
        """Stop background discovery tasks and persist address book."""
        self._running = False
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        if self._exchange_task:
            self._exchange_task.cancel()
            try:
                await self._exchange_task
            except asyncio.CancelledError:
                pass
        if self._latency_task:
            self._latency_task.cancel()
            try:
                await self._latency_task
            except asyncio.CancelledError:
                pass
        # Persist address book on shutdown
        self.close_db()

    async def _discovery_loop(self):
        """
        Periodically attempt to connect to new peers.
        Runs more frequently when we have fewer peers than MIN_PEERS.
        """
        # Initial discovery: try to connect immediately
        await asyncio.sleep(2)
        await self._try_connect_peers()

        while self._running:
            try:
                # Adapt interval based on peer count and candidate availability
                connected = self.peer_manager.connected_count
                candidates = self.get_peers_to_connect()
                if connected < MIN_PEERS:
                    interval = 5  # Aggressive discovery when below minimum
                elif not candidates:
                    interval = 120  # No candidates available — slow down significantly
                elif connected < TARGET_PEERS:
                    interval = self.discovery_interval
                else:
                    interval = self.discovery_interval * 2  # Relaxed when healthy

                await asyncio.sleep(interval)
                await self._try_connect_peers()

                # Periodic maintenance
                self.peer_manager.clean_expired_bans()
                self._prune_dead_addresses()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(10)

    async def _peer_exchange_loop(self):
        """Periodically request peers from connected nodes."""
        while self._running:
            try:
                await asyncio.sleep(self.peer_exchange_interval)

                # This is handled by the Node/Server via GET_PEERS messages
                # We just update our address book when we receive PEERS responses
                pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Peer exchange error: {e}")

    async def _latency_measurement_loop(self):
        """
        Periodically re-measure latency to known addresses.
        This enables dynamic peer selection: if a closer peer appears
        (e.g., a neighbor on the same LAN), we'll discover it.
        """
        # Initial delay to let the node establish first connections
        await asyncio.sleep(15)

        while self._running:
            try:
                # Re-measure latency for connected peers (via their addresses)
                connected_urls = [
                    p.url for p in self.peer_manager.get_connected_peers()
                ]
                # Also measure a sample of unconnected addresses
                unconnected = [
                    url for url in self._addresses
                    if url not in connected_urls
                    and url != self._own_url
                ]
                # Sample up to 10 unconnected addresses to probe
                sample = random.sample(
                    unconnected, min(10, len(unconnected))
                ) if unconnected else []

                all_to_measure = connected_urls + sample
                if all_to_measure:
                    results = await self.sort_peers_by_latency(all_to_measure)
                    # Update connected peers' latency_ms field
                    for url, lat in results:
                        for peer in self.peer_manager.get_connected_peers():
                            if peer.url == url or peer.listen_url == url:
                                if lat < 999:
                                    peer.latency_ms = lat
                                    peer.score_breakdown.update_latency(lat)
                                break

                    measured_count = sum(1 for _, l in results if l < 999)
                    if measured_count > 0:
                        best = min(results, key=lambda x: x[1])
                        logger.debug(
                            f"Latency sweep: {measured_count}/{len(results)} "
                            f"reachable, best={best[1]:.0f}ms"
                        )

                await asyncio.sleep(self.latency_measurement_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Latency measurement loop error: {e}")
                await asyncio.sleep(30)

    async def _try_connect_peers(self):
        """
        Attempt to connect to peers we need.
        Measures latency to candidate peers first and prefers the closest.
        Ensures at least 1 bootstrap/seed node connection for chain reliability.
        """
        if not self._connect_callback:
            return

        addresses = self.get_peers_to_connect()
        if not addresses:
            return

        # Measure latency to candidates before connecting.
        # Filter out addresses that are known-unreachable (failed latency probe)
        # to avoid noisy connection attempts.
        sorted_peers = await self.sort_peers_by_latency(addresses)
        if not sorted_peers:
            return

        # Ensure at least 1 bootstrap node is included for chain sync reliability
        bootstrap_urls = set(self.bootstrap_nodes)
        has_bootstrap_connected = any(
            p.url in bootstrap_urls or p.listen_url in bootstrap_urls
            for p in self.peer_manager.get_connected_peers()
        )
        sorted_urls = [url for url, _ in sorted_peers]

        if not has_bootstrap_connected:
            # Find first bootstrap in candidates and move to front
            for i, (url, lat) in enumerate(sorted_peers):
                if url in bootstrap_urls:
                    if i > 0:
                        sorted_urls.remove(url)
                        sorted_urls.insert(0, url)
                    break

        logger.debug(
            f"Attempting to connect to {len(sorted_urls)} peers "
            f"(currently {self.peer_manager.connected_count}, "
            f"closest: {sorted_peers[0][1]:.0f}ms)"
        )

        # Connect to top peers concurrently (prefer closest)
        tasks = []
        for addr in sorted_urls[:self.max_preferred_peers]:
            tasks.append(self._connect_one(addr))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _connect_one(self, url: str):
        """Try to connect to a single peer."""
        try:
            success = await self._connect_callback(url)
            if success:
                self.on_connect_success(url)
                logger.info(f"Connected to peer at {url}")
            else:
                self.on_connect_failure(url)
                info = self._addresses.get(url)
                failures = info.failures if info else 0
                backoff = info.backoff_seconds if info else 0
                logger.debug(f"Could not connect to {url} (attempt {failures}, backoff {backoff:.0f}s)")
        except Exception as e:
            self.on_connect_failure(url)
            logger.debug(f"Connection attempt to {url} failed: {e}")

    def _prune_dead_addresses(self):
        """Remove addresses that have consistently failed."""
        to_remove = []
        for url, info in self._addresses.items():
            # Keep bootstrap nodes forever
            if info.source == "bootstrap":
                continue
            # Remove if 3+ consecutive failures and no recent success
            if info.failures >= 3 and info.successes == 0:
                to_remove.append(url)
            # Remove stale addresses (not seen in 24 hours)
            if info.last_seen > 0 and time.time() - info.last_seen > 86400:
                to_remove.append(url)

        for url in to_remove:
            del self._addresses[url]

    # ------------------------------------------------------------------ #
    #  SQLite Persistence for Address Book                                 #
    # ------------------------------------------------------------------ #

    def _init_db(self, db_path: str):
        """Initialize SQLite database for peer address persistence."""
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS peer_addresses (
                url TEXT PRIMARY KEY,
                last_seen REAL DEFAULT 0,
                last_attempt REAL DEFAULT 0,
                failures INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                source TEXT DEFAULT 'peer_exchange'
            )
        """)
        self._db.commit()
        logger.info(f"Peer address book DB initialized: {db_path}")

    def _load_from_db(self):
        """Load persisted peer addresses from database."""
        if self._db is None:
            return
        rows = self._db.execute("SELECT * FROM peer_addresses").fetchall()
        for url, last_seen, last_attempt, failures, successes, source in rows:
            if url not in self._addresses:
                self._addresses[url] = AddressInfo(
                    url=url,
                    last_seen=last_seen,
                    last_attempt=last_attempt,
                    failures=failures,
                    successes=successes,
                    source=source,
                )
        logger.info(f"Loaded {len(rows)} peer addresses from database")

    def save_to_db(self):
        """Persist current address book to database."""
        if self._db is None:
            return
        for url, info in self._addresses.items():
            self._db.execute("""
                INSERT OR REPLACE INTO peer_addresses
                (url, last_seen, last_attempt, failures, successes, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (url, info.last_seen, info.last_attempt,
                  info.failures, info.successes, info.source))
        self._db.commit()

    def close_db(self):
        """Close the peer address database."""
        if self._db:
            self.save_to_db()
            self._db.close()
            self._db = None

    def get_stats(self) -> dict:
        measured = [
            i for i in self._addresses.values()
            if i.latency_ms < 999
        ]
        avg_latency = (
            sum(i.latency_ms for i in measured) / len(measured)
            if measured else 0
        )
        return {
            "known_addresses": len(self._addresses),
            "bootstrap_count": sum(
                1 for i in self._addresses.values() if i.source == "bootstrap"
            ),
            "exchange_count": sum(
                1 for i in self._addresses.values() if i.source == "peer_exchange"
            ),
            "latency_measured": len(measured),
            "avg_latency_ms": round(avg_latency, 1),
            "running": self._running,
            "persisted": self._db is not None,
        }
