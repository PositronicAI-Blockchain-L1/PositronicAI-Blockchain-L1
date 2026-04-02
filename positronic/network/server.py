"""
Positronic - P2P Network Server
aiohttp-based WebSocket server for P2P communication and JSON-RPC API.
Handles inbound/outbound connection lifecycle, message routing,
peer exchange, and graceful shutdown.
"""

import asyncio
import hashlib
import json
import os
import ssl
import time
import logging
from typing import Dict, Optional, Callable, List

import aiohttp
from aiohttp import web

from positronic.network.messages import (
    MessageType, NetworkMessage, MessageIDCache,
    make_hello, make_hello_ack, make_disconnect,
    make_pong, make_peers, make_ping,
    make_blocks, make_get_blocks, make_get_peers,
    make_new_block, make_new_tx,
    make_status, make_request_status, make_sync_request, make_sync_response,
    make_get_headers, make_headers,
    make_attestation,
    make_pow_challenge, make_pow_solution,
)
from positronic.network.peer import (
    Peer, PeerManager, PeerState, ConnectionDirection,
)
from positronic.network.protocol import ProtocolHandler, PeerProtocolState
from positronic.constants import CHAIN_ID, PROTOCOL_VERSION, DEFAULT_P2P_PORT

logger = logging.getLogger("positronic.network.server")


class P2PServer:
    """
    WebSocket-based P2P server with JSON-RPC HTTP endpoint.

    Manages both inbound connections (peers connecting to us) and
    outbound connections (us connecting to peers). Each connection
    has a dedicated listener task for receiving messages.
    """

    def __init__(
        self,
        host: str,
        port: int,
        node_id: str,
        peer_manager: PeerManager,
        ssl_context: Optional[ssl.SSLContext] = None,
        client_ssl_context: Optional[ssl.SSLContext] = None,
        security_manager=None,  # PeerSecurityManager (optional)
    ):
        self.host = host
        self.port = port
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.protocol = ProtocolHandler(node_id=node_id)
        self.app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._running = False

        # TLS support
        self._ssl_context = ssl_context              # Server-side TLS
        self._client_ssl_context = client_ssl_context  # Client-side TLS for outbound
        self._tls_enabled = ssl_context is not None

        # Peer security manager (IP gate, flood check, kick evaluation)
        self._security_manager = security_manager

        # Track active listener tasks for outbound connections
        self._listener_tasks: Dict[str, asyncio.Task] = {}

        # Message deduplication cache (separate from protocol-level)
        self._relay_cache = MessageIDCache(max_size=20_000)

        # Track send failures per peer; disconnect after threshold (fixes zombie connection)
        self._send_failures: Dict[str, int] = {}
        self._send_failure_threshold = 5

        # Periodic latency measurement via PING/PONG
        self._latency_ping_task: Optional[asyncio.Task] = None
        self._latency_ping_interval = 60  # seconds

        # Callbacks for the Node layer
        self.on_new_block: Optional[Callable] = None
        self.on_new_tx: Optional[Callable] = None
        self.on_attestation: Optional[Callable] = None
        self.on_rpc_request: Optional[Callable] = None
        self.on_sync_request: Optional[Callable] = None
        self.on_blocks_received: Optional[Callable] = None
        self.on_headers_received: Optional[Callable] = None
        self.on_peers_received: Optional[Callable] = None
        self.on_status_received: Optional[Callable] = None
        self.on_peer_connected: Optional[Callable] = None
        self.on_peer_disconnected: Optional[Callable] = None

        # Chain state (set by Node)
        self._chain_height: int = 0
        self._best_hash: str = ""

        self._setup_routes()
        self._setup_protocol()

    def set_chain_state(self, height: int, best_hash: str):
        """Update local chain state for handshakes and status messages."""
        self._chain_height = height
        self._best_hash = best_hash

    def _setup_routes(self):
        self.app.router.add_get("/ws", self._handle_ws_inbound)
        self.app.router.add_post("/rpc", self._handle_rpc)
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/peers", self._handle_peers_api)

    def _setup_protocol(self):
        """Register all message type handlers."""
        self.protocol.register_many({
            MessageType.HELLO: self._on_hello,
            MessageType.HELLO_ACK: self._on_hello_ack,
            MessageType.DISCONNECT: self._on_disconnect,
            MessageType.PING: self._on_ping,
            MessageType.PONG: self._on_pong,
            MessageType.GET_PEERS: self._on_get_peers,
            MessageType.PEERS: self._on_peers,
            MessageType.NEW_BLOCK: self._on_new_block,
            MessageType.NEW_TX: self._on_new_tx,
            MessageType.STATUS: self._on_status,
            MessageType.REQUEST_STATUS: self._on_request_status,
            MessageType.GET_BLOCKS: self._on_get_blocks,
            MessageType.BLOCKS: self._on_blocks,
            MessageType.GET_HEADERS: self._on_get_headers,
            MessageType.HEADERS: self._on_headers,
            MessageType.SYNC_REQUEST: self._on_sync_request,
            MessageType.SYNC_RESPONSE: self._on_sync_response,
            MessageType.ATTESTATION: self._on_attestation,
            MessageType.SYSTEM_TX: self._on_system_tx,
        })
        self.on_system_tx = None  # Callback set by Node

    # === Server Lifecycle ===

    async def start(self):
        """Start the P2P server (with optional TLS)."""
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(
            self._runner,
            self.host,
            self.port,
            ssl_context=self._ssl_context,
            backlog=128,
        )
        await site.start()
        self._running = True
        # Start periodic latency measurement via PING to all connected peers
        self._latency_ping_task = asyncio.create_task(self._latency_ping_loop())
        proto = "wss" if self._tls_enabled else "ws"
        logger.info(f"P2P server listening on {proto}://{self.host}:{self.port}")

    async def stop(self):
        """Gracefully stop the server and all connections."""
        self._running = False

        # Cancel latency ping task
        if self._latency_ping_task:
            self._latency_ping_task.cancel()
            try:
                await self._latency_ping_task
            except asyncio.CancelledError:
                pass

        # Cancel all listener tasks
        for peer_id, task in list(self._listener_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._listener_tasks.clear()

        # Disconnect all peers
        await self.peer_manager.disconnect_all()

        # Stop the HTTP server
        if self._runner:
            await self._runner.cleanup()

        logger.info("P2P server stopped")

    # === Inbound Connection Handling ===

    async def _handle_ws_inbound(self, request) -> web.WebSocketResponse:
        """
        Handle incoming WebSocket P2P connections.
        This is the server-side handler for peers connecting to us.
        """
        ws = web.WebSocketResponse(
            heartbeat=30,  # Built-in WebSocket ping/pong
            max_msg_size=4 * 1024 * 1024,  # 4MB max message
        )
        await ws.prepare(request)

        peer_id = None
        peer_host = request.remote or "unknown"
        # Strip port if present (e.g. "1.2.3.4:12345" → "1.2.3.4")
        peer_ip = peer_host.split(":")[0] if ":" in peer_host else peer_host

        # Layer 1: IP rate-limit / ban gate
        if self._security_manager:
            if not self._security_manager.check_ip(peer_ip):
                logger.info("Rejected inbound from %s (rate-limited or banned)", peer_ip)
                await ws.close()
                return ws
            self._security_manager.record_connection_attempt(peer_ip)

        # Layer 2 — PoW anti-Sybil challenge (exempts validators)
        if self._security_manager:
            try:
                nonce = self._security_manager.issue_challenge()
                await ws.send_str(
                    make_pow_challenge(self.node_id, nonce).serialize()
                )
                # Wait for solution (10s timeout)
                sol_msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                if sol_msg.type != aiohttp.WSMsgType.TEXT:
                    logger.info(
                        "PoW challenge: %s sent non-text response, closing",
                        peer_ip,
                    )
                    await ws.close()
                    return ws
                try:
                    sol = NetworkMessage.deserialize(sol_msg.data)
                    if sol.msg_type != MessageType.POW_SOLUTION:
                        logger.info(
                            "PoW challenge: %s sent wrong msg type %s, closing",
                            peer_ip, sol.msg_type,
                        )
                        await ws.close()
                        return ws
                    sol_nonce    = sol.payload.get("nonce", "")
                    sol_solution = sol.payload.get("solution", "")
                    if not self._security_manager.verify_challenge(sol_nonce, sol_solution):
                        logger.info(
                            "PoW challenge failed from %s, closing", peer_ip
                        )
                        await ws.close()
                        return ws
                except Exception as e:
                    logger.info("PoW parse error from %s: %s", peer_ip, e)
                    await ws.close()
                    return ws
            except asyncio.TimeoutError:
                logger.info("PoW challenge timeout from %s", peer_ip)
                await ws.close()
                return ws
            except Exception as e:
                logger.warning("PoW challenge error for %s: %s", peer_ip, e)
                # Non-fatal: allow connection if PoW machinery fails

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        network_msg = NetworkMessage.deserialize(msg.data)
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            f"Invalid message from {peer_host}: {e}"
                        )
                        continue

                    sender_id = network_msg.sender_id

                    # On first message, track the peer
                    if peer_id is None and sender_id:
                        peer_id = sender_id
                        # Create an inbound peer entry if HELLO
                        if network_msg.msg_type == MessageType.HELLO:
                            peer = Peer(
                                peer_id=peer_id,
                                host=peer_ip,
                                port=network_msg.payload.get("listen_port", 0),
                                state=PeerState.HANDSHAKING,
                                direction=ConnectionDirection.INBOUND,
                                ws=ws,
                                connected_at=time.time(),
                                last_seen=time.time(),
                            )
                            self.peer_manager._add_peer_sync(peer)
                            # Layer 3+4: register with security manager
                            if self._security_manager:
                                self._security_manager.register_peer(peer_id, peer_ip)

                    # Layer 3: per-message flood check
                    if peer_id and self._security_manager:
                        if not self._security_manager.on_message(peer_id):
                            logger.info(
                                "Inbound peer %s flood-kicked", peer_id[:8]
                            )
                            break

                    # Update peer tracking
                    if peer_id:
                        peer = self.peer_manager.get_peer(peer_id)
                        if peer:
                            peer.update_seen()
                            peer.messages_received += 1
                            peer.bytes_received += len(msg.data)

                    # Handle message through protocol
                    response = await self.protocol.handle_message(network_msg)
                    if response:
                        await ws.send_str(response.serialize())

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.debug(
                        f"WS error from {peer_host}: {ws.exception()}"
                    )
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    break

        except Exception as e:
            logger.error(f"Error handling inbound connection from {peer_host}: {e}")
        finally:
            if peer_id:
                self._send_failures.pop(peer_id, None)
                self.protocol.remove_peer(peer_id)
                if self._security_manager:
                    self._security_manager.unregister_peer(peer_id)
                removed = self.peer_manager.remove_peer(peer_id)
                if removed:
                    logger.info(f"Inbound peer {peer_id[:8]} disconnected")
                    if self.on_peer_disconnected:
                        try:
                            self.on_peer_disconnected(peer_id)
                        except Exception as e:
                            logger.warning(f"on_peer_disconnected callback error: {e}")

        return ws

    # === Outbound Connection Handling ===

    async def connect_to_peer(self, url: str) -> bool:
        """
        Connect to a peer node (outbound).
        Sends HELLO handshake and starts a listener task.
        Returns True on successful connection.
        """
        if not self._running:
            return False

        try:
            # Use TLS for wss:// URLs — always use CERT_NONE for P2P outbound
            # (seed nodes use self-signed certs; identity verified via node_id, not TLS)
            connector = None
            peer_ssl_ctx = None
            if url.startswith("wss://"):
                peer_ssl_ctx = ssl.create_default_context()
                peer_ssl_ctx.check_hostname = False
                peer_ssl_ctx.verify_mode = ssl.CERT_NONE
                connector = aiohttp.TCPConnector(ssl=peer_ssl_ctx)
            session = aiohttp.ClientSession(connector=connector)
            ws = await asyncio.wait_for(
                session.ws_connect(
                    url,
                    heartbeat=30,
                    max_msg_size=4 * 1024 * 1024,
                    ssl=peer_ssl_ctx,
                ),
                timeout=10,
            )

            # Layer 2 — solve PoW challenge if server sends one
            # Read first message; it may be a POW_CHALLENGE or immediately HELLO_ACK
            # (some peers may not enforce PoW yet)
            try:
                first_msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                if first_msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        parsed = NetworkMessage.deserialize(first_msg.data)
                        if parsed.msg_type == MessageType.POW_CHALLENGE:
                            nonce_hex = parsed.payload.get("nonce", "")
                            # Solve: find solution where SHA-256(nonce+sol)[0] < 0x40
                            nonce_bytes = bytes.fromhex(nonce_hex)
                            solution = ""
                            for _ in range(10000):
                                candidate = os.urandom(16)
                                digest = hashlib.sha256(nonce_bytes + candidate).digest()
                                if digest[0] < 0x40:
                                    solution = candidate.hex()
                                    break
                            if solution:
                                await ws.send_str(
                                    make_pow_solution(self.node_id, nonce_hex, solution).serialize()
                                )
                                logger.debug("PoW challenge solved for %s", url)
                            else:
                                logger.warning("PoW challenge unsolvable for %s", url)
                                await session.close()
                                return False
                        # else: server sent something else first (unexpected) — continue
                    except Exception:
                        pass  # Not a valid message — server may not enforce PoW
                elif first_msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.info("Peer %s closed connection during PoW phase (rate limited)", url)
                    await session.close()
                    return False
            except asyncio.TimeoutError:
                pass  # Server didn't send a challenge — proceed without PoW

            # Send HELLO handshake
            hello = make_hello(
                node_id=self.node_id,
                chain_id=CHAIN_ID,
                height=self._chain_height,
                best_hash=self._best_hash,
                protocol_version=PROTOCOL_VERSION,
                listen_port=self.port,
            )
            await ws.send_str(hello.serialize())

            # Wait for HELLO_ACK
            resp = await asyncio.wait_for(ws.receive(), timeout=15)
            if resp.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                logger.info(f"Peer {url} closed connection during handshake (rate limited or rejected)")
                await session.close()
                return False
            elif resp.type != aiohttp.WSMsgType.TEXT:
                logger.warning(f"Handshake failed for {url}: expected TEXT, got {resp.type}")
                await session.close()
                return False

            try:
                ack = NetworkMessage.deserialize(resp.data)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Handshake failed for {url}: bad HELLO_ACK payload: {e}")
                await session.close()
                return False

            if ack.msg_type != MessageType.HELLO_ACK:
                logger.warning(f"Handshake failed for {url}: expected HELLO_ACK, got {ack.msg_type}")
                await session.close()
                return False

            # Extract peer info
            remote_id = ack.payload.get("node_id", ack.sender_id)
            if not remote_id or remote_id == self.node_id:
                logger.warning(f"Handshake rejected for {url}: invalid or self node_id")
                await session.close()
                return False

            # Check if already connected
            if self.peer_manager.has_peer(remote_id):
                await session.close()
                return False

            # Parse host/port from URL
            host = "localhost"
            port = DEFAULT_P2P_PORT
            try:
                if "://" in url:
                    hostport = url.split("://")[1].split("/")[0]
                    parts = hostport.split(":")
                    host = parts[0]
                    if len(parts) > 1:
                        port = int(parts[1])
            except (ValueError, IndexError):
                pass

            # Create peer
            peer = Peer(
                peer_id=remote_id,
                host=host,
                port=port,
                state=PeerState.CONNECTED,
                chain_height=ack.payload.get("height", 0),
                best_hash=ack.payload.get("best_hash", ""),
                direction=ConnectionDirection.OUTBOUND,
                ws=ws,
                session=session,
                connected_at=time.time(),
                last_seen=time.time(),
                tls_enabled=url.startswith("wss://"),
            )
            self.peer_manager._add_peer_sync(peer)
            # Register outbound peer with security manager
            if self._security_manager:
                self._security_manager.register_peer(remote_id, host)

            # Mark peer as active in protocol
            self.protocol.set_peer_active(remote_id, ack.payload)

            # Start listener for this outbound connection
            listener_ready = asyncio.Event()
            task = asyncio.create_task(
                self._outbound_listener(remote_id, ws, session, ready_event=listener_ready)
            )
            self._listener_tasks[remote_id] = task

            # Wait for listener to be ready before notifying Node layer
            try:
                await asyncio.wait_for(listener_ready.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Listener for {remote_id[:8]} not ready in 5s, proceeding")

            logger.info(
                f"Connected to peer {remote_id[:8]} at {url} "
                f"(height={ack.payload.get('height', 0)})"
            )

            # Notify Node layer
            if self.on_peer_connected:
                try:
                    self.on_peer_connected(remote_id)
                except Exception as e:
                    logger.warning(f"on_peer_connected callback error: {e}")

            return True

        except asyncio.TimeoutError:
            logger.info(f"Connection timeout to {url}")
            try:
                await session.close()
            except Exception as e:
                logger.debug(f"Session close error after timeout: {e}")
            return False
        except Exception as e:
            logger.debug(f"Failed to connect to {url}: {e}")
            try:
                await session.close()
            except Exception as e:
                logger.debug(f"Session close error after connect failure: {e}")
            return False

    async def _outbound_listener(
        self,
        peer_id: str,
        ws: aiohttp.ClientWebSocketResponse,
        session: aiohttp.ClientSession,
        ready_event: asyncio.Event = None,
    ):
        """
        Listen for messages on an outbound WebSocket connection.
        Runs until the connection is closed or the node shuts down.
        """
        try:
            if ready_event:
                ready_event.set()
            async for msg in ws:
                if not self._running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        network_msg = NetworkMessage.deserialize(msg.data)
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.warning(
                            f"Invalid message from {peer_id[:8]}: {e}"
                        )
                        continue

                    # Layer 3: per-message flood check (outbound listener)
                    if self._security_manager:
                        if not self._security_manager.on_message(peer_id):
                            logger.info(
                                "Outbound peer %s flood-kicked", peer_id[:8]
                            )
                            break

                    # Update peer tracking
                    peer = self.peer_manager.get_peer(peer_id)
                    if peer:
                        peer.update_seen()
                        peer.messages_received += 1
                        peer.bytes_received += len(msg.data)

                    # Handle through protocol
                    response = await self.protocol.handle_message(network_msg)
                    if response:
                        try:
                            await ws.send_str(response.serialize())
                        except Exception as e:
                            logger.debug(f"Failed to send response to {peer_id[:8]}: {e}")
                            break

                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Outbound listener error for {peer_id[:8]}: {e}")
        finally:
            # Cleanup
            self._listener_tasks.pop(peer_id, None)
            self._send_failures.pop(peer_id, None)
            self.protocol.remove_peer(peer_id)
            if self._security_manager:
                self._security_manager.unregister_peer(peer_id)
            removed = self.peer_manager.remove_peer(peer_id)
            if removed:
                logger.info(f"Outbound peer {peer_id[:8]} disconnected")
                if self.on_peer_disconnected:
                    try:
                        self.on_peer_disconnected(peer_id)
                    except Exception as e:
                        logger.warning(f"on_peer_disconnected callback error: {e}")
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"WebSocket close error for {peer_id[:8]}: {e}")
            try:
                await session.close()
            except Exception as e:
                logger.debug(f"Session close error for {peer_id[:8]}: {e}")

    # === Message Sending ===

    async def broadcast(
        self,
        message: NetworkMessage,
        exclude: str = None,
    ):
        """
        Send message to all connected peers except excluded.
        Uses relay cache to prevent re-broadcasting already-seen messages.
        """
        # Check relay dedup
        if not self._relay_cache.add_and_check(message.msg_id):
            return  # Already relayed

        data = message.serialize()
        peers = self.peer_manager.get_connected_peers()
        failed = []

        target_count = sum(1 for p in peers if p.peer_id != exclude and p.peer_id != message.sender_id)
        logger.debug("Broadcasting %s to %d peers", message.msg_type.name if hasattr(message, 'msg_type') else 'unknown', target_count)

        for peer in peers:
            if peer.peer_id == exclude:
                continue
            if peer.peer_id == message.sender_id:
                continue  # Don't send back to originator

            success = await peer.send(data)
            if success:
                self._send_failures.pop(peer.peer_id, None)
            else:
                failed.append(peer.peer_id)

        if failed:
            logger.debug("Broadcast failed for %d peers: %s", len(failed), [p[:8] for p in failed])

        # On repeated send failure: disconnect zombie so follower can reconnect
        for peer_id in failed:
            peer = self.peer_manager.get_peer(peer_id)
            if peer:
                peer.adjust_score(-5)
            count = self._send_failures.get(peer_id, 0) + 1
            self._send_failures[peer_id] = count
            if count >= self._send_failure_threshold:
                logger.warning(
                    "Peer %s: %d send failures in a row, disconnecting (zombie connection)",
                    peer_id[:8], count,
                )
                self._send_failures.pop(peer_id, None)
                asyncio.create_task(
                    self.peer_manager.disconnect_peer(peer_id, "repeated send failure")
                )

    async def send_to_peer(
        self, peer_id: str, message: NetworkMessage
    ) -> bool:
        """Send a message to a specific peer."""
        peer = self.peer_manager.get_peer(peer_id)
        if not peer:
            return False
        return await peer.send(message.serialize())

    async def request_peers(self, peer_id: str):
        """Send a GET_PEERS request to a specific peer."""
        msg = make_get_peers(self.node_id)
        await self.send_to_peer(peer_id, msg)

    async def send_status(self, peer_id: str):
        """Send our current chain status to a peer."""
        msg = make_status(self.node_id, self._chain_height, self._best_hash)
        await self.send_to_peer(peer_id, msg)

    async def request_status_from_peer(self, peer_id: str) -> bool:
        """Ask peer to send STATUS so we get fresh chain height (for periodic re-sync)."""
        msg = make_request_status(self.node_id)
        return await self.send_to_peer(peer_id, msg)

    async def ping_peer(self, peer_id: str):
        """Send a PING to a specific peer."""
        peer = self.peer_manager.get_peer(peer_id)
        if peer:
            msg = make_ping(self.node_id)
            peer.last_ping_sent = time.time()
            await self.send_to_peer(peer_id, msg)

    async def request_blocks(
        self, peer_id: str, start_height: int, count: int
    ):
        """Request blocks from a specific peer."""
        msg = make_get_blocks(start_height, count, self.node_id)
        await self.send_to_peer(peer_id, msg)

    # === Protocol Handlers ===

    def _on_hello(self, msg: NetworkMessage) -> Optional[NetworkMessage]:
        """Handle incoming HELLO handshake from inbound peer."""
        payload = msg.payload
        peer_id = payload.get("node_id", msg.sender_id)

        # Validate handshake
        is_valid, error = self.protocol.validate_hello(payload)
        if not is_valid:
            if "Self-connection" in error:
                logger.debug(f"HELLO rejected from {peer_id[:8]}: {error}")
            else:
                logger.warning(f"HELLO rejected from {peer_id[:8]}: {error}")
            return make_disconnect(self.node_id, error)

        # Update peer state
        peer = self.peer_manager.get_peer(peer_id)
        if peer:
            peer.state = PeerState.CONNECTED
            peer.chain_height = payload.get("height", 0)
            peer.best_hash = payload.get("best_hash", "")
            peer.protocol_version = payload.get("protocol_version", 1)
            peer.client_name = payload.get("client", "")
            peer.listen_port = payload.get("listen_port", 0)

        # Mark as active in protocol
        self.protocol.set_peer_active(peer_id, payload)

        logger.info(
            f"Handshake with inbound peer {peer_id[:8]} "
            f"(height={payload.get('height', 0)}, "
            f"client={payload.get('client', 'unknown')})"
        )

        # Notify Node layer
        if self.on_peer_connected:
            try:
                self.on_peer_connected(peer_id)
            except Exception as e:
                logger.warning(f"on_peer_connected callback error: {e}")

        return make_hello_ack(
            self.node_id, self._chain_height, self._best_hash
        )

    def _on_hello_ack(self, msg: NetworkMessage) -> None:
        """Handle HELLO_ACK (response to our outbound HELLO)."""
        # This is normally handled inline in connect_to_peer,
        # but can arrive if timing is unusual
        peer_id = msg.payload.get("node_id", msg.sender_id)
        self.protocol.set_peer_active(peer_id, msg.payload)
        return None

    def _on_disconnect(self, msg: NetworkMessage) -> None:
        """Handle graceful DISCONNECT from peer."""
        peer_id = msg.sender_id
        reason = msg.payload.get("reason", "")
        logger.info(f"Peer {peer_id[:8]} disconnecting: {reason}")
        # Peer will be cleaned up when the WS connection closes
        return None

    def _on_ping(self, msg: NetworkMessage) -> NetworkMessage:
        """Handle PING, respond with PONG."""
        peer = self.peer_manager.get_peer(msg.sender_id)
        if peer:
            peer.update_seen()
        return make_pong(self.node_id, msg.payload.get("time", 0))

    def _on_pong(self, msg: NetworkMessage) -> None:
        """Handle PONG, update peer latency and multi-dimensional score."""
        peer = self.peer_manager.get_peer(msg.sender_id)
        if peer:
            ping_time = msg.payload.get("ping_time", 0)
            pong_time = msg.payload.get("pong_time", time.time())
            peer.update_latency(ping_time, pong_time)
            peer.last_pong_received = time.time()
            peer.ping_failures = 0
            peer.update_seen()
            if peer.latency_ms > 0:
                logger.debug(
                    f"Peer {msg.sender_id[:8]} latency: {peer.latency_ms:.1f}ms"
                )
        return None

    def _on_get_peers(self, msg: NetworkMessage) -> NetworkMessage:
        """Handle GET_PEERS request, respond with known peers."""
        urls = self.peer_manager.get_peer_urls()
        # Include our own listen URL (use wss:// if TLS enabled)
        proto = "wss" if self._tls_enabled else "ws"
        own_url = f"{proto}://{self.host}:{self.port}/ws"
        if own_url not in urls:
            urls.append(own_url)
        return make_peers(urls[:20], self.node_id)  # Max 20 peers

    def _on_peers(self, msg: NetworkMessage) -> None:
        """Handle PEERS response."""
        peers = msg.payload.get("peers", [])
        if self.on_peers_received:
            try:
                self.on_peers_received(peers, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling peers: {e}")
        return None

    def _on_new_block(self, msg: NetworkMessage) -> None:
        """Handle a new block announcement."""
        block_dict = msg.payload.get("block")
        if not block_dict:
            return None

        # Update peer's known height
        block_height = block_dict.get("header", {}).get("height", 0)
        peer = self.peer_manager.get_peer(msg.sender_id)
        if peer and block_height > peer.chain_height:
            peer.chain_height = block_height
            self.protocol.update_peer_height(
                msg.sender_id, block_height
            )

        if self.on_new_block:
            try:
                self.on_new_block(block_dict, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling new block: {e}")

        return None

    def _on_new_tx(self, msg: NetworkMessage) -> None:
        """Handle a new transaction announcement."""
        tx_dict = msg.payload.get("transaction")
        if not tx_dict:
            return None
        if self.on_new_tx:
            try:
                self.on_new_tx(tx_dict, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling new tx: {e}")
        return None

    def _on_attestation(self, msg: NetworkMessage) -> None:
        """Handle ATTESTATION message for BFT finality."""
        if self.on_attestation:
            try:
                self.on_attestation(msg.payload, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling attestation: {e}")
        return None

    def _on_system_tx(self, msg: NetworkMessage) -> None:
        """Handle SYSTEM_TX: stake/unstake/claim propagated from another node."""
        if self.on_system_tx:
            try:
                self.on_system_tx(msg.payload, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling system TX: {e}")
        return None

    def _on_status(self, msg: NetworkMessage) -> None:
        """Handle STATUS message (chain state advertisement)."""
        height = msg.payload.get("height", 0)
        best_hash = msg.payload.get("best_hash", "")

        peer = self.peer_manager.get_peer(msg.sender_id)
        if peer:
            peer.chain_height = height
            peer.best_hash = best_hash

        self.protocol.update_peer_height(msg.sender_id, height, best_hash)

        if self.on_status_received:
            try:
                self.on_status_received(msg.sender_id, height, best_hash)
            except Exception as e:
                logger.error(f"Error handling status: {e}")

        return None

    def _on_request_status(self, msg: NetworkMessage) -> Optional[NetworkMessage]:
        """Reply to REQUEST_STATUS with our current STATUS (for re-sync / fresh height)."""
        return make_status(self.node_id, self._chain_height, self._best_hash)

    def _on_get_blocks(self, msg: NetworkMessage) -> None:
        """Handle GET_BLOCKS request."""
        if self.on_sync_request:
            start = msg.payload.get("start_height", 0)
            count = msg.payload.get("count", 50)
            try:
                self.on_sync_request(
                    msg.sender_id, "get_blocks", start, count
                )
            except Exception as e:
                logger.error(f"Error handling get_blocks: {e}")
        return None

    def _on_blocks(self, msg: NetworkMessage) -> None:
        """Handle BLOCKS response."""
        blocks = msg.payload.get("blocks", [])
        if self.on_blocks_received:
            try:
                self.on_blocks_received(blocks, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling blocks: {e}")
        return None

    def _on_get_headers(self, msg: NetworkMessage) -> None:
        """Handle GET_HEADERS request."""
        if self.on_sync_request:
            start = msg.payload.get("start_height", 0)
            count = msg.payload.get("count", 100)
            try:
                self.on_sync_request(
                    msg.sender_id, "get_headers", start, count
                )
            except Exception as e:
                logger.error(f"Error handling get_headers: {e}")
        return None

    def _on_headers(self, msg: NetworkMessage) -> None:
        """Handle HEADERS response."""
        headers = msg.payload.get("headers", [])
        if self.on_headers_received:
            try:
                self.on_headers_received(headers, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling headers: {e}")
        return None

    def _on_sync_request(self, msg: NetworkMessage) -> None:
        """Handle SYNC_REQUEST."""
        if self.on_sync_request:
            start = msg.payload.get("start_height", 0)
            end = msg.payload.get("end_height", 0)
            try:
                self.on_sync_request(
                    msg.sender_id, "sync", start, end - start
                )
            except Exception as e:
                logger.error(f"Error handling sync request: {e}")
        return None

    def _on_sync_response(self, msg: NetworkMessage) -> None:
        """Handle SYNC_RESPONSE."""
        blocks = msg.payload.get("blocks", [])
        has_more = msg.payload.get("has_more", False)
        if self.on_blocks_received:
            try:
                self.on_blocks_received(blocks, msg.sender_id)
            except Exception as e:
                logger.error(f"Error handling sync response: {e}")
        return None

    # === JSON-RPC ===

    async def _handle_rpc(self, request) -> web.Response:
        """Handle JSON-RPC requests (MetaMask compatible)."""
        try:
            body = await request.json()
            method = body.get("method", "")
            params = body.get("params", [])
            req_id = body.get("id", 1)

            if self.on_rpc_request:
                result = self.on_rpc_request(method, params)
                if asyncio.iscoroutine(result):
                    result = await result
            else:
                result = {"error": "RPC not configured"}

            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
            return web.json_response(response)
        except Exception as e:
            logger.warning(f"RPC request error: {e}")
            return web.json_response({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32603, "message": str(e)},
            })

    # === HTTP API ===

    async def _handle_health(self, request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "ok",
            "node_id": self.node_id,
            "peers": self.peer_manager.connected_count,
            "height": self._chain_height,
            "running": self._running,
        })

    async def _handle_peers_api(self, request) -> web.Response:
        """Peers listing endpoint."""
        peers = [p.to_dict() for p in self.peer_manager.get_connected_peers()]
        return web.json_response({
            "connected": len(peers),
            "peers": peers,
            "stats": self.peer_manager.get_stats(),
        })

    # === Periodic Latency Measurement ===

    async def _latency_ping_loop(self):
        """
        Periodically PING all connected peers to measure latency.
        This feeds into the multi-dimensional peer scoring system
        and the discovery layer's latency-based selection.
        Runs every 60 seconds.
        """
        # Initial delay to let connections establish
        await asyncio.sleep(10)

        while self._running:
            try:
                peers = self.peer_manager.get_connected_peers()
                for peer in peers:
                    try:
                        await self.ping_peer(peer.peer_id)
                    except Exception as e:
                        logger.debug(
                            f"Latency ping failed for {peer.peer_id[:8]}: {e}"
                        )
                        peer.ping_failures += 1

                # Log summary
                if peers:
                    latencies = [
                        p.latency_ms for p in peers if p.latency_ms > 0
                    ]
                    if latencies:
                        avg = sum(latencies) / len(latencies)
                        best = min(latencies)
                        logger.debug(
                            f"Latency sweep: {len(latencies)} peers measured, "
                            f"avg={avg:.0f}ms, best={best:.0f}ms"
                        )

                await asyncio.sleep(self._latency_ping_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Latency ping loop error: {e}")
                await asyncio.sleep(30)

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "node_id": self.node_id,
            "listener_tasks": len(self._listener_tasks),
            "relay_cache_size": self._relay_cache.size,
            "protocol": self.protocol.get_stats(),
            "peers": self.peer_manager.get_stats(),
        }
