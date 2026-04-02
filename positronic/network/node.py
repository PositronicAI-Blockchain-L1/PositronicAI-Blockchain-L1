"""
Positronic - Main Node Class
Orchestrates all node components: blockchain, network, consensus, AI, mempool.

Manages the complete lifecycle of a Positronic node including:
- P2P networking (server, peer management, discovery)
- Block synchronization with peers
- Block production (for validators)
- Transaction broadcasting and mempool management
- Periodic maintenance (ping, peer exchange, stale cleanup)
"""

import asyncio
import logging
import time
from typing import Optional, List

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512
from positronic.crypto.address import address_from_pubkey
from positronic.chain.blockchain import Blockchain
from positronic.core.block import Block
from positronic.core.transaction import Transaction
from positronic.network.server import P2PServer
from positronic.network.peer import PeerManager
from positronic.network.mempool import Mempool
from positronic.network.discovery import PeerDiscovery
from positronic.network.peer_security import PeerSecurityManager
from positronic.network.sync import BlockSync
from positronic.network.messages import (
    make_new_block, make_new_tx, make_blocks,
    make_get_peers, make_status, make_attestation,
)
from positronic.consensus.finality import FinalityTracker
from positronic.utils.config import NodeConfig
from positronic.utils.time_sync import NetworkTimeSynchronizer
from positronic.network.tls import create_server_ssl_context, create_client_ssl_context
from positronic.rpc.server import RPCServer
from positronic.storage.database import derive_storage_password
from positronic.constants import BLOCK_TIME, DEFAULT_P2P_PORT
from positronic.emergency.controller import EmergencyController
from positronic.monitoring.collector import MetricsCollector

logger = logging.getLogger("positronic.network.node")


class Node:
    """
    A Positronic network node.

    Supports three modes:
    - Full Node: Validates and stores all blocks
    - Validator Node: Full node + proposes blocks
    - NVN (Neural Validator Node): Full node + runs AI scoring

    Lifecycle:
    1. __init__  - Create components
    2. start()   - Start server, discovery, sync, background tasks
    3. stop()    - Gracefully shut down everything
    """

    def __init__(self, config: NodeConfig = None):
        self.config = config or NodeConfig()
        self.config.ensure_dirs()

        # Generate or load node identity.
        # Every node gets a keypair for P2P identity and block signing.
        # Priority: 1) Founder keypair  2) Persisted keypair  3) New random
        if (getattr(self.config, '_founder_mode', False)
                and self.config.validator.enabled):
            from positronic.core.genesis import get_genesis_founder_keypair
            self.keypair = get_genesis_founder_keypair()
            logger.info("Using genesis founder keypair for block production")
        else:
            # Try to load persisted node keypair (so address stays the same)
            import os
            from positronic.crypto.data_encryption import DataEncryptor
            kp_path = os.path.join(self.config.storage.data_dir, "node_keypair.bin")
            encryptor = DataEncryptor(self.config.storage.data_dir)
            loaded = False
            if os.path.exists(kp_path):
                try:
                    seed = encryptor.decrypt_file(kp_path)
                    if len(seed) == 32:
                        self.keypair = KeyPair.from_bytes(seed)
                        loaded = True
                        logger.info("Loaded persisted node keypair: %s", self.keypair.address_hex)
                        # Migrate: if file was plaintext, encrypt it now
                        with open(kp_path, "rb") as f:
                            raw = f.read()
                        if not raw[:17].startswith(b"POSITRONIC_ENC"):
                            encryptor.encrypt_and_write(kp_path, seed)
                            logger.info("Migrated node keypair to encrypted format")
                except Exception as e:
                    logger.warning("Failed to load node keypair: %s", e)
            if not loaded:
                self.keypair = KeyPair()
                # Persist encrypted for next restart
                try:
                    os.makedirs(os.path.dirname(kp_path), exist_ok=True)
                    encryptor.encrypt_and_write(kp_path, self.keypair.private_key_bytes)
                    logger.info("Persisted new node keypair (encrypted): %s", self.keypair.address_hex)
                except Exception as e:
                    logger.warning("Failed to persist node keypair: %s", e)
        self.node_id = sha512(self.keypair.public_key_bytes).hex()[:16]

        # Derive machine-specific encryption password for all databases
        db_password = derive_storage_password(self.config.storage.data_dir)

        # Core components — pass full config so Blockchain can wire AI thresholds
        self.blockchain = Blockchain(
            db_path=self.config.storage.db_path,
            config=self.config,
            encryption_password=db_password,
        )
        self.mempool = Mempool()
        self.peer_manager = PeerManager(
            max_peers=self.config.network.max_peers,
        )
        # Peer security: IP rate-limiter, flood detection, quality scoring,
        # validator protection, and resource-based eviction
        self.security_manager = PeerSecurityManager()
        self.discovery = PeerDiscovery(
            self.peer_manager,
            self.config.network.bootstrap_nodes or None,
            network_type=self.config.network.network_type,
            security_manager=self.security_manager,
        )
        self.sync = BlockSync(self.peer_manager)

        # Time synchronization
        self.time_sync = NetworkTimeSynchronizer()

        # TLS support
        server_ssl = None
        client_ssl = None
        if self.config.network.tls.enabled:
            try:
                server_ssl = create_server_ssl_context(
                    cert_path=self.config.network.tls.cert_path,
                    key_path=self.config.network.tls.key_path,
                    ca_path=self.config.network.tls.ca_path,
                    verify_peers=self.config.network.tls.verify_peers,
                    data_dir=self.config.storage.data_dir,
                )
                client_ssl = create_client_ssl_context(
                    ca_path=self.config.network.tls.ca_path,
                    verify_peers=self.config.network.tls.verify_peers,
                )
                logger.info("TLS enabled for P2P connections")
            except Exception as e:
                logger.warning(f"TLS setup failed, falling back to plain: {e}")

        # Network server
        self.server = P2PServer(
            host=self.config.network.p2p_host,
            port=self.config.network.p2p_port,
            node_id=self.node_id,
            peer_manager=self.peer_manager,
            ssl_context=server_ssl,
            client_ssl_context=client_ssl,
            security_manager=self.security_manager,
        )

        # RPC server (JSON-RPC 2.0 API for wallets & dApps)
        # Shares TLS context with P2P server for HTTPS support
        self.rpc_server = RPCServer(
            host=self.config.network.rpc_host,
            port=self.config.network.rpc_port,
            blockchain=self.blockchain,
            mempool=self.mempool,
            network_type=self.config.network.network_type,
            ssl_context=server_ssl,
        )

        # Wire up callbacks
        self._setup_callbacks()

        # Give RPC access to peer info so positronic_nodeInfo reports peers
        self.rpc_server.rpc.set_peer_manager(
            self.peer_manager,
            network_type=self.config.network.network_type,
        )
        # Give RPC access to sync state so positronic_nodeInfo reports real sync progress
        self.rpc_server.rpc.set_sync(self.sync)
        # Give RPC access to P2P server for SYSTEM_TX broadcast (stake/unstake sync)
        self.rpc_server.rpc._p2p_server = self.server
        self.rpc_server.rpc._node_event_loop = None  # Set in start()
        # RPC TX broadcast is handled via _pending_broadcasts queue
        # (polled in _block_production_loop)

        # Connect AI to mempool for early screening
        self._wire_ai_to_mempool()

        # State
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        self._p2p_pending_blocks: list = []  # blocks buffered while syncing

        # Emergency control system
        self.emergency = EmergencyController(node=self)

        # Tendermint BFT consensus (for validators)
        # NOTE: Initialized LAZILY in _init_bft() after blockchain.start()
        # because consensus registry is only available after _load_chain()
        self.bft = None

        # Light Validator mode (desktop stakers: TAD-only scoring + attestation)
        self._light_validator = None

        # Prometheus metrics collector
        self.metrics_collector = MetricsCollector(
            health_monitor=getattr(self.blockchain, 'health_monitor', None),
            blockchain=self.blockchain,
            mempool=self.mempool,
        )

    def _setup_callbacks(self):
        """Connect all component callbacks."""
        # Server -> Node callbacks
        self.server.on_new_block = self._handle_new_block
        self.server.on_new_tx = self._handle_new_tx
        self.server.on_attestation = self._handle_attestation
        self.server.on_system_tx = self._handle_system_tx
        self.server.on_blocks_received = self._handle_blocks_received
        self.server.on_peers_received = self._handle_peers_received
        self.server.on_status_received = self._handle_status_received
        self.server.on_sync_request = self._handle_sync_request
        self.server.on_peer_connected = self._handle_peer_connected
        self.server.on_peer_disconnected = self._handle_peer_disconnected

        # Wire P2P server's /rpc endpoint to RPC handler
        self.server.on_rpc_request = self._handle_rpc_request

        # Sync callbacks
        self.sync.set_callbacks(
            add_block=self._sync_add_block,
            get_block=self._sync_get_block,
            send_request=self._sync_send_request,
            on_sync_complete=self._on_sync_complete,
            on_sync_stalled=self._on_sync_stalled,
            get_chain_height=lambda: self.blockchain.height,
        )

        # Discovery callback
        self.discovery.set_connect_callback(self.server.connect_to_peer)

    def _wire_ai_to_mempool(self):
        """Connect AI validation gate and immune system to mempool."""
        try:
            # AI gate is available from blockchain
            if hasattr(self.blockchain, 'ai_gate'):
                self.mempool.set_ai_gate(self.blockchain.ai_gate)
                logger.info("AI validation gate connected to mempool")
            # Connect state manager for nonce validation (anti-double-spend)
            if hasattr(self.blockchain, 'state'):
                self.mempool._state_manager = self.blockchain.state
                logger.info("State manager connected to mempool (nonce validation)")
        except Exception as e:
            logger.debug(f"AI gate not available for mempool: {e}")

        try:
            # Quarantine pool
            if hasattr(self.blockchain, 'quarantine_pool'):
                self.mempool.set_quarantine_pool(self.blockchain.quarantine_pool)
            elif hasattr(self.blockchain, 'ai_gate'):
                # Create a quarantine pool if blockchain doesn't have one
                from positronic.ai.quarantine import QuarantinePool
                self._quarantine_pool = QuarantinePool()
                self.mempool.set_quarantine_pool(self._quarantine_pool)
        except Exception as e:
            logger.debug(f"Quarantine pool not available: {e}")

        try:
            # Immune system
            if hasattr(self.blockchain, 'immune_system'):
                self.mempool.set_immune_system(self.blockchain.immune_system)
        except Exception as e:
            logger.debug(f"Immune system not available: {e}")

    # === Node Lifecycle ===

    async def start(self, founder_mode: bool = False):
        """
        Start the node.

        Args:
            founder_mode: If True, create genesis block. Only for chain initialization.
                Uses the canonical genesis keypair (deterministic) so all nodes
                produce the same genesis block. The node's own keypair is used
                for block production after genesis.
        """
        # Store event loop reference for P2P broadcast from RPC thread
        import asyncio as _asyncio
        self.rpc_server.rpc._node_event_loop = _asyncio.get_running_loop()

        # Initialize blockchain:
        # - Founder mode OR existing DB: create/load genesis + allocations
        # - Non-founder with empty DB: skip genesis creation, let sync
        #   deliver the genesis block from the network. This avoids
        #   hash mismatches when the codebase evolves (new header fields
        #   change the genesis hash).
        from positronic.core.genesis import get_genesis_founder_keypair
        genesis_kp = get_genesis_founder_keypair()

        # Always initialize — creates deterministic genesis + allocations
        # if DB is empty, or loads existing chain from DB.
        self.blockchain.initialize(genesis_kp)
        existing_height = self.blockchain.chain_db.get_chain_height()

        # Initialize BFT AFTER blockchain.initialize() so consensus registry exists
        if self.config.validator.enabled and self.bft is None:
            try:
                from positronic.consensus.tendermint import TendermintBFT
                self.bft = TendermintBFT(
                    registry=self.blockchain.consensus.registry,
                    election=self.blockchain.consensus.election,
                    clock=self.blockchain.consensus.clock,
                    my_address=self.keypair.address,
                    my_keypair=self.keypair,
                )
                logger.info("Tendermint BFT consensus initialized (post-chain-load)")
            except Exception as e:
                logger.warning("BFT initialization failed: %s", e)

        # Auto-detect validator eligibility on startup:
        # If node has staked >= MIN_STAKE, enable validator mode automatically
        # (only for server nodes — desktop apps use Light Validator mode instead)
        if not self.config.validator.enabled and self.blockchain.consensus:
            from positronic.constants import MIN_STAKE
            acc = self.blockchain.state.get_account(self.keypair.address)
            if acc.staked_amount >= MIN_STAKE:
                # Light Validator mode: desktop apps with stake
                # They attest blocks and run light AI scoring but don't produce blocks
                from positronic.ai.light_validator import LightValidator
                gate = getattr(self.blockchain, 'ai_gate', None)
                self._light_validator = LightValidator(gate)
                logger.info(
                    "Light Validator mode: attestation + light AI scoring "
                    "(staked %d ASF)", acc.staked_amount // (10**18)
                )

        # Register this node as active validator in consensus registry
        # Only nodes running the software can produce blocks (Ethereum-style)
        if self.config.validator.enabled and self.blockchain.consensus:
            self._register_node_as_validator()

        if founder_mode:
            logger.info("Genesis block created (founder mode). Node is ready.")
        else:
            logger.info("Chain initialized (height %d). Ready to sync.",
                       existing_height)

        logger.info(f"Genesis founder: {genesis_kp.address_hex}")
        logger.info(f"Node address: {self.keypair.address_hex}")

        # Bootstrap AI anomaly detector with synthetic data if not yet trained.
        # This ensures the TAD autoencoder has baseline reconstruction error
        # statistics from genesis, so it can detect anomalous transactions
        # even before any real transaction history is available.
        self._bootstrap_ai_if_needed()

        # Update server with current chain state
        self._update_server_chain_state()

        # Set up discovery
        proto = "wss" if self.config.network.tls.enabled else "ws"
        own_url = f"{proto}://{self.config.network.p2p_host}:{self.config.network.p2p_port}/ws"
        self.discovery.set_own_url(own_url)

        # Detect external IP to prevent self-connection via NAT
        asyncio.ensure_future(self.discovery.detect_external_ip())

        # NTP time synchronization (best-effort, non-blocking)
        try:
            if self.time_sync.sync_ntp(timeout=2.0):
                logger.info(
                    f"NTP synced: offset={self.time_sync.state.offset_seconds:.3f}s"
                )
            else:
                logger.info("NTP sync failed - using local clock")
        except Exception as e:
            logger.debug(f"NTP sync error: {e}")

        # Start P2P server
        await self.server.start()
        logger.info(
            f"P2P server started on "
            f"{self.config.network.p2p_host}:{self.config.network.p2p_port}"
        )
        logger.info(f"Node ID: {self.node_id}")

        # Start RPC server (JSON-RPC API)
        try:
            await self.rpc_server.start()
            logger.info(
                f"RPC server started on "
                f"{self.config.network.rpc_host}:{self.config.network.rpc_port}"
            )
        except Exception as e:
            logger.warning(f"RPC server failed to start: {e}")

        self._running = True

        # Start peer security manager (background cleanup loop)
        self.security_manager.start()

        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._maintenance_loop()),
            asyncio.create_task(self._sync_check_loop()),
            asyncio.create_task(self._peer_exchange_loop()),
            asyncio.create_task(self._time_sync_loop()),
            asyncio.create_task(self.metrics_collector.start(interval=5.0)),
            asyncio.create_task(self._security_kick_loop()),
        ]

        # Start discovery
        await self.discovery.start()

        # Start block production only when validator mode is explicitly enabled.
        # The --founder flag creates the deterministic genesis but does NOT
        # imply block production — a node can be a founder sync node without
        # producing blocks (useful for RPC/seed nodes).
        if self.config.validator.enabled:
            self._background_tasks.append(
                asyncio.create_task(self._block_production_loop())
            )
            logger.info("Block production enabled")

        # Start Light Validator loop if in light validator mode
        if self._light_validator is not None:
            self._background_tasks.append(
                asyncio.create_task(self._light_validation_loop())
            )
            logger.info("Light Validator loop started")

        logger.info("Node started successfully")

    async def stop(self):
        """Gracefully stop the node and all background tasks."""
        logger.info("Stopping node...")
        self._running = False

        # Deregister from consensus election (Ethereum-style: offline = no slots)
        if self.config.validator.enabled and self.blockchain.consensus:
            try:
                registry = self.blockchain.consensus.registry
                if self.keypair.address in registry._validators:
                    registry.deactivate(self.keypair.address)
                    logger.info("Validator deregistered from election")
            except Exception as e:
                logger.debug("Deregister failed: %s", e)

        # Stop RPC server
        try:
            await self.rpc_server.stop()
        except Exception as e:
            logger.debug(f"RPC server stop error: {e}")

        # Stop peer security manager
        self.security_manager.stop()

        # Stop discovery
        await self.discovery.stop()

        # Stop sync
        await self.sync.stop()

        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()

        # Stop server (disconnects all peers)
        await self.server.stop()

        logger.info("Node stopped")

    # === Transaction Handling ===

    def submit_transaction(self, tx: Transaction) -> bool:
        """
        Submit a transaction to the mempool.
        If accepted, broadcasts to all connected peers.
        """
        if not self.blockchain.validate_transaction(tx):
            return False
        if not self.mempool.add(tx):
            return False

        # Broadcast to peers
        asyncio.create_task(
            self.server.broadcast(
                make_new_tx(tx.to_dict(), self.node_id)
            )
        )
        return True

    # === Network Message Handlers ===

    def _handle_new_block(self, block_dict: dict, from_peer: str):
        """Handle a new block received from a peer."""
        try:
            # Buffer broadcast blocks while syncing — process after sync completes
            if self.sync.state.syncing:
                block_height = block_dict.get("header", {}).get("height", 0)
                if block_height > self.blockchain.height:
                    self._p2p_pending_blocks.append((block_dict, from_peer))
                    logger.debug(
                        f"Block #{block_height} buffered (syncing, from {from_peer[:8]})"
                    )
                return

            block = Block.from_dict(block_dict)

            # Skip if we already have this block
            if block.height <= self.blockchain.height:
                existing = self.blockchain.get_block(block.height)
                if existing and existing.hash == block.hash:
                    return

            # Skip state root verification for P2P-received blocks.
            # State root may differ between nodes due to non-deterministic
            # trie rebuild ordering. The producing node already verified
            # state consistency; syncing nodes trust the consensus.
            if self.blockchain.add_block(block, skip_state_root=True):
                self.mempool.on_block_added(block.transactions)
                self._update_server_chain_state()

                logger.info(
                    f"Block #{block.height} received from "
                    f"{from_peer[:8]} ({block.tx_count} txs)"
                )

                # Relay to other peers (exclude sender)
                asyncio.create_task(
                    self.server.broadcast(
                        make_new_block(block_dict, self.node_id),
                        exclude=from_peer,
                    )
                )

                # Broadcast updated STATUS to all peers so they discover
                # our new height and can trigger sync if they're behind.
                # This enables indirect chain propagation (gossip):
                # A -> C -> D  (D learns about A's chain through C's status)
                asyncio.create_task(self._broadcast_chain_status())

                # Broadcast attestation for BFT finality (validators + light validators)
                if self.blockchain.consensus and self.config.validator.enabled:
                    self._broadcast_attestation(block)
                elif self._light_validator is not None and self.blockchain.consensus:
                    # Light Validators attest blocks and score their transactions
                    self._broadcast_attestation(block)
                    self._light_validator.on_block_attested()
                    asyncio.create_task(
                        self._light_score_block(block)
                    )

                # Register received block with BFT engine
                if self.bft and hasattr(block, 'height'):
                    try:
                        self.bft.on_proposal(block, block.header.validator_pubkey[:20], 0, -1, b"")
                    except Exception:
                        pass

            else:
                # Block rejected -- could be a fork from a network partition.
                # If the sender has a longer chain, trigger sync to catch up
                # (chain reorganization).
                peer = self.peer_manager.get_peer(from_peer)
                peer_height = peer.chain_height if peer else 0
                if peer_height > self.blockchain.height:
                    if not self.sync.state.syncing:
                        logger.info(
                            f"Fork detected: peer {from_peer[:8]} has "
                            f"height {peer_height} > ours "
                            f"{self.blockchain.height}. Starting sync."
                        )
                        asyncio.create_task(
                            self.sync.start_sync(self.blockchain.height)
                        )
                else:
                    logger.debug(
                        f"Block #{block.height} from {from_peer[:8]} "
                        f"rejected"
                    )

        except Exception as e:
            logger.error(f"Error handling block from {from_peer[:8]}: {e}")

    async def _broadcast_chain_status(self):
        """Broadcast our current chain status to all connected peers."""
        try:
            for peer in self.peer_manager.get_connected_peers():
                try:
                    await self.server.send_status(peer.peer_id)
                except Exception as e:
                    logger.debug(f"Failed to send status to peer {peer.peer_id[:8]}: {e}")
        except Exception as e:
            logger.debug(f"Error broadcasting status: {e}")

    def _handle_new_tx(self, tx_dict: dict, from_peer: str):
        """Handle a new transaction received from a peer."""
        try:
            # Drop transactions while syncing — account state is stale
            if self.sync.state.syncing:
                return

            tx = Transaction.from_dict(tx_dict)
            if self.mempool.contains(tx.tx_hash):
                return  # Already have it

            if self.blockchain.validate_transaction(tx):
                if self.mempool.add(tx):
                    # Relay to other peers
                    asyncio.create_task(
                        self.server.broadcast(
                            make_new_tx(tx_dict, self.node_id),
                            exclude=from_peer,
                        )
                    )
        except Exception as e:
            logger.debug(f"Error handling tx from {from_peer[:8]}: {e}")

    def _handle_system_tx(self, payload: dict, from_peer: str):
        """Handle a SYSTEM_TX (stake/unstake/claim) received from a peer.
        Execute it locally via RPC handler with _forwarded flag to prevent re-broadcast."""
        method = payload.get("method", "")
        params = payload.get("params", [])
        if not method or not params:
            return
        logger.info("Received SYSTEM_TX from peer %s: %s", from_peer[:8], method)
        # Add _forwarded flag so local handler doesn't re-broadcast
        params_with_flag = list(params) + [{"_forwarded": True}]
        try:
            self.rpc_server.rpc.handle(method, params_with_flag)
        except Exception as e:
            logger.debug("System TX execution failed: %s %s", method, e)

    def _broadcast_attestation(self, block: Block):
        """
        Create, sign, and broadcast an attestation for a block.
        Also submits the attestation locally for finality tracking.
        """
        try:
            slot = block.header.slot
            block_hash = block.header.hash

            # Compute the canonical attestation message and sign it
            msg_bytes = FinalityTracker.compute_attestation_message(
                slot, block_hash
            )
            signature = self.keypair.sign(msg_bytes)
            validator_address = self.keypair.address

            # Submit attestation locally
            self.blockchain.submit_attestation(
                slot=slot,
                block_hash=block_hash,
                validator_address=validator_address,
                signature=signature,
            )

            # Broadcast to peers
            attestation_msg = make_attestation(
                block_hash=block_hash.hex(),
                height=slot,
                validator_id=validator_address.hex(),
                signature=signature.hex(),
                node_id=self.node_id,
            )
            asyncio.create_task(self.server.broadcast(attestation_msg))

            logger.debug(
                f"Attestation broadcast for slot {slot} "
                f"(block {block.hash_hex[:18]})"
            )
        except Exception as e:
            logger.error(f"Error broadcasting attestation: {e}")

    async def _light_validation_loop(self):
        """
        Background loop for Light Validators.

        Periodically scores mempool transactions using TAD-only scoring
        and broadcasts anomaly reports to peers.
        """
        from positronic.ai.light_validator import LightValidator
        logger.info("Light Validator loop running")

        while self._running:
            try:
                await asyncio.sleep(LightValidator.SCORING_INTERVAL)

                if self._light_validator is None or not self._light_validator.active:
                    continue

                # Score pending mempool transactions with TAD-only
                pending_txs = list(self.mempool.get_pending())[:50]  # batch limit
                for tx in pending_txs:
                    sender_account = None
                    if hasattr(tx, 'sender') and hasattr(self.blockchain, 'state'):
                        try:
                            sender_account = self.blockchain.state.get_account(tx.sender)
                        except Exception:
                            pass

                    score, is_anomaly = self._light_validator.score_transaction(
                        tx, sender_account,
                    )

                    if is_anomaly:
                        tx_hash = tx.hash_hex if hasattr(tx, 'hash_hex') else tx.hash.hex()
                        self._light_validator.report_anomaly(
                            tx_hash=tx_hash,
                            score=score,
                            reason="mempool-tad-anomaly",
                            reporter_address=self.keypair.address_hex,
                        )

                # Broadcast pending anomaly reports to peers
                reports = self._light_validator.get_pending_reports()
                if reports:
                    for report in reports:
                        alert_msg = {
                            "type": "anomaly_alert",
                            "tx_hash": report.tx_hash,
                            "score": report.score,
                            "reason": report.reason,
                            "reporter": report.reporter_address,
                            "timestamp": report.timestamp,
                            "node_id": self.node_id,
                        }
                        try:
                            await self.server.broadcast(alert_msg)
                        except Exception as e:
                            logger.debug("Anomaly broadcast failed: %s", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Light validation loop error: %s", e)

    async def _light_score_block(self, block: Block):
        """Score a newly received block's transactions using light (TAD-only) validation."""
        if self._light_validator is None:
            return

        try:
            state = getattr(self.blockchain, 'state', None)
            scored, anomalies, reports = self._light_validator.score_block_transactions(
                block, state,
            )

            if anomalies > 0:
                logger.info(
                    "Light Validator: block #%d scored %d txs, %d anomalies",
                    block.height, scored, anomalies,
                )

                # Broadcast anomaly reports
                for report in reports:
                    alert_msg = {
                        "type": "anomaly_alert",
                        "tx_hash": report.tx_hash,
                        "score": report.score,
                        "reason": report.reason,
                        "reporter": report.reporter_address or self.keypair.address_hex,
                        "timestamp": report.timestamp,
                        "node_id": self.node_id,
                    }
                    try:
                        await self.server.broadcast(alert_msg)
                    except Exception as e:
                        logger.debug("Anomaly broadcast failed: %s", e)

        except Exception as e:
            logger.debug("Light block scoring error: %s", e)

    def _handle_attestation(self, payload: dict, from_peer: str):
        """Handle an incoming attestation from a peer."""
        try:
            slot = payload.get("height", 0)
            block_hash_hex = payload.get("block_hash", "")
            validator_hex = payload.get("validator_id", "")
            signature_hex = payload.get("signature", "")

            if not all([slot, block_hash_hex, validator_hex, signature_hex]):
                return

            block_hash = bytes.fromhex(block_hash_hex)
            validator_address = bytes.fromhex(validator_hex)
            signature = bytes.fromhex(signature_hex)

            self.blockchain.submit_attestation(
                slot=slot,
                block_hash=block_hash,
                validator_address=validator_address,
                signature=signature,
            )
        except Exception as e:
            logger.debug(
                f"Error handling attestation from {from_peer[:8]}: {e}"
            )

    def _handle_blocks_received(self, blocks: List[dict], from_peer: str):
        """Handle blocks received (sync response)."""
        self.sync.on_blocks_received(blocks, from_peer)

    def _handle_peers_received(self, peers: List[str], from_peer: str):
        """Handle peers received from peer exchange."""
        self.discovery.add_known_addresses(peers)
        logger.debug(
            f"Received {len(peers)} peer addresses from {from_peer[:8]}"
        )

    def _handle_status_received(
        self, peer_id: str, height: int, best_hash: str
    ):
        """Handle status message from a peer."""
        self.sync.on_status_received(peer_id, height, best_hash)

        # Check if we need to sync
        if height > self.blockchain.height and not self.sync.state.syncing:
            asyncio.create_task(
                self.sync.start_sync(self.blockchain.height)
            )

    def _handle_sync_request(
        self, peer_id: str, request_type: str, start: int, count: int
    ):
        """Handle a sync/block request from a peer."""
        blocks = self.sync.handle_sync_request(
            peer_id, request_type, start, count
        )
        if blocks:
            msg = make_blocks(blocks, self.node_id)
            asyncio.create_task(
                self.server.send_to_peer(peer_id, msg)
            )

    def _handle_peer_connected(self, peer_id: str):
        """Called when a new peer completes handshake."""
        logger.info(f"Peer {peer_id[:8]} connected")

        # Send our status
        asyncio.create_task(
            self.server.send_status(peer_id)
        )

        # Request their peers
        asyncio.create_task(
            self.server.request_peers(peer_id)
        )

        # Eager sync: trigger sync shortly after connect so we don't rely only on
        # the 15s _sync_check_loop. Fixes intermittent stuck node (DEFERRED_ITEMS_REPORT).
        asyncio.create_task(self._maybe_sync_after_peer_connect(delay=1.5))
        asyncio.create_task(self._maybe_sync_after_peer_connect(delay=5.0))  # retry for race when 2 nodes connect at once

    async def _maybe_sync_after_peer_connect(self, delay: float = 1.5):
        """
        Run shortly after a peer connects: wait for any in-flight STATUS to update
        peer heights, then trigger sync if we're behind. Reduces reliance on the
        15s _sync_check_loop and fixes intermittent P2P sync (e.g. Node3 stuck at 179).
        Called twice (e.g. 1.5s and 5s) to handle race when two nodes connect at once.
        """
        try:
            await asyncio.sleep(delay)
            if not self._running:
                return
            if self.sync._is_sync_active():
                return
            if self.sync.needs_sync(self.blockchain.height):
                logger.info(
                    "Eager sync: peer has higher height, starting sync after connect"
                )
                await self.sync.start_sync(self.blockchain.height)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Eager sync after connect: {e}")

    def _handle_peer_disconnected(self, peer_id: str):
        """Called when a peer disconnects."""
        logger.info(f"Peer {peer_id[:8]} disconnected")

    def _handle_rpc_request(self, method: str, params: list):
        """Handle an RPC request from the P2P server's /rpc endpoint."""
        return self.rpc_server.rpc.handle(method, params)

    # === Sync Callbacks ===

    def _sync_add_block(self, block_dict: dict) -> bool:
        """Add a block during sync — TRUSTED path.

        During sync, blocks come from peers that have already validated
        them. We skip full validation (parent hash, state root, proposer)
        and directly persist the block. This avoids issues with genesis
        hash evolution when header format changes.

        Security: P2P broadcast blocks (new blocks) ARE fully validated
        in _handle_new_block(). Only historical sync is trusted.
        """
        try:
            block = Block.from_dict(block_dict)

            # If blockchain already has this block, skip
            if block.height <= self.blockchain.height:
                existing = self.blockchain.get_block(block.height)
                if existing:
                    return True

            # Genesis reconciliation: if this is block #1 and parent
            # hash doesn't match local genesis, adopt network's hash
            if (block.height == 1 and self.blockchain.height == 0
                    and self.blockchain.chain_head is not None):
                expected_parent = block.header.previous_hash
                local_hash = self.blockchain.chain_head.hash
                if local_hash != expected_parent:
                    logger.info(
                        "Adopting network genesis hash for chain continuity"
                    )
                    self.blockchain.chain_head._cached_hash = expected_parent
                    try:
                        self.blockchain.chain_db.update_block_hash(
                            0, expected_parent.hex()
                        )
                    except Exception:
                        pass

            # Direct persist — skip validation for sync'd blocks
            try:
                snapshot_id = self.blockchain.state.snapshot()

                # Derive validator address from block header (matches add_block)
                validator_addr = (
                    address_from_pubkey(block.header.validator_pubkey)
                    if block.header.validator_pubkey else b""
                )

                # Collect NVN addresses for fee distribution (matches add_block)
                nvn_addrs = []
                if self.blockchain.consensus:
                    try:
                        nvn_addrs = [
                            v.address
                            for v in self.blockchain.consensus.registry.nvn_validators
                        ]
                    except Exception:
                        nvn_addrs = []

                # Execute transactions to build state
                for tx in block.transactions:
                    try:
                        self.blockchain.executor.execute(
                            tx,
                            block_height=block.height,
                            block_hash=block.hash,
                            validator_address=validator_addr,
                            nvn_addresses=nvn_addrs,
                            block_timestamp=block.header.timestamp,
                        )
                    except Exception as e:
                        logger.debug("Sync block #%d tx exec failed: %s",
                                     block.height, e)

                # Sync validator registry from on-chain state so newly
                # staked validators are recognized during sync
                if self.blockchain.consensus and block.height > 0:
                    self.blockchain._sync_registry_from_state()

                # Distribute attestation rewards (attesters, NVN, treasury)
                if self.blockchain.consensus and block.height > 0:
                    self.blockchain._distribute_attestation_rewards(block)

                # Persist block and state
                self.blockchain.chain_db.put_block(block)
                self.blockchain.state.save_to_db(self.blockchain.state_db)
                self.blockchain.chain_head = block
                self.blockchain.height = block.height
                self.blockchain.state.commit_snapshot(snapshot_id)
            except Exception as e:
                logger.debug("Sync block #%d exec error (non-fatal): %s",
                           block.height, e)
                # Even if execution fails, persist the block for chain continuity
                try:
                    self.blockchain.chain_db.put_block(block)
                    self.blockchain.chain_head = block
                    self.blockchain.height = block.height
                except Exception:
                    return False
            success = True
            if success:
                self.mempool.on_block_added(block.transactions)
                self._update_server_chain_state()
                # Broadcast updated chain status to all peers so indirect
                # nodes (e.g. D connected only through C) discover the new
                # height and can trigger their own sync.
                asyncio.create_task(self._broadcast_chain_status())
            return success
        except Exception as e:
            logger.error(f"Sync add_block error: %s", e)
            return False

    def _sync_get_block(self, height: int) -> Optional[dict]:
        """Get a block by height (for serving sync requests)."""
        block = self.blockchain.get_block(height)
        if block:
            return block.to_dict()
        return None

    async def _sync_send_request(
        self, peer_id: str, start: int, count: int
    ):
        """Send a block request to a peer during sync."""
        await self.server.request_blocks(peer_id, start, count)

    def _on_sync_complete(self):
        """Called by the sync module after sync finishes.

        Broadcasts our updated chain status to ALL connected peers so
        that downstream nodes (not directly connected to the sync source)
        discover the new chain height and trigger their own sync.
        This enables indirect chain propagation: A -> C -> D.
        """
        asyncio.create_task(self._broadcast_chain_status())

        # Process P2P broadcast blocks that arrived during sync
        if self._p2p_pending_blocks:
            pending = self._p2p_pending_blocks[:]
            self._p2p_pending_blocks.clear()
            logger.info(f"Processing {len(pending)} blocks buffered during sync")
            for block_dict, peer_id in pending:
                self._handle_new_block(block_dict, peer_id)

    async def _on_sync_stalled(self, peer_id: str):
        """Called when sync stalls after MAX_RETRIES (no blocks received from peer).

        Disconnect this peer so discovery can reconnect; the new connection
        may fix one-way or broken links (fixes Node2 stuck at same height).
        """
        await self.peer_manager.disconnect_peer(peer_id, "sync stalled")

    # === Background Tasks ===

    async def _block_production_loop(self):
        """Produce blocks at regular intervals (for validators)."""
        while self._running:
            try:
                # Only produce if this node is a REGISTERED validator in consensus
                if self.blockchain.consensus:
                    my_addr = self.keypair.address
                    if my_addr not in self.blockchain.consensus.registry._validators:
                        await asyncio.sleep(BLOCK_TIME)
                        continue

                # Emergency pause check
                if not self.emergency.should_produce_blocks():
                    await asyncio.sleep(1)
                    continue

                await asyncio.sleep(BLOCK_TIME)

                # Don't produce blocks while syncing
                if self.sync.state.syncing:
                    continue

                # Broadcast any TXs queued by RPC (eth_sendRawTransaction)
                if hasattr(self.rpc_server, 'rpc') and self.rpc_server.rpc._pending_broadcasts:
                    from positronic.network.messages import make_new_tx
                    while self.rpc_server.rpc._pending_broadcasts:
                        tx_dict = self.rpc_server.rpc._pending_broadcasts.pop(0)
                        try:
                            await self.server.broadcast(
                                make_new_tx(tx_dict, self.node_id))
                            logger.info("Broadcast TX from RPC to %d peers",
                                        self.peer_manager.connected_count)
                        except Exception as e:
                            logger.debug("TX broadcast error: %s", e)

                # Get pending transactions
                pending = self.mempool.get_pending_transactions()
                if pending:
                    logger.info("Block production: %d pending txs from mempool", len(pending))
                else:
                    logger.debug("Mempool empty: size=%d pending_keys=%d",
                                 self.mempool.size, len(self.mempool.pending))
                if not pending and self.blockchain.height > 0:
                    # Skip empty blocks in mainnet to save resources.
                    # In testnet/local: produce empty blocks for chain liveness.
                    if self.config.network.network_type not in ("local", "testnet"):
                        continue

                # Create and add block — proposer check with fallback
                slot = self.blockchain.height + 1
                epoch = slot // 32
                FALLBACK_DELAY = 6  # seconds to wait before fallback production (half block time)

                # Check if this node is the assigned proposer for this slot
                is_my_slot = True  # default: produce if no consensus
                if self.blockchain.consensus:
                    self.blockchain._ensure_epoch_election(slot)
                    expected = self.blockchain.consensus.get_proposer(slot)
                    if expected and expected != self.keypair.address:
                        is_my_slot = False

                if not is_my_slot:
                    # Not my slot — wait FALLBACK_DELAY, then produce if no block arrived
                    height_before = self.blockchain.height
                    await asyncio.sleep(FALLBACK_DELAY)
                    if self.blockchain.height > height_before:
                        continue  # Primary proposer produced — skip
                    # Primary proposer is offline — fallback produce
                    slot = self.blockchain.height + 1
                    epoch = slot // 32
                    logger.info("Fallback: primary proposer offline, producing block #%d", slot)

                block = self.blockchain.create_block(
                    pending, self.keypair, slot=slot, epoch=epoch
                )

                # BFT proposal (single-validator shortcut)
                if self.bft:
                    self.bft.start_round(block.height, 0)
                    self.bft.on_proposal(block, self.keypair.address, 0, -1, b"")
                    logger.debug("BFT round started for block #%d", block.height)

                if self.blockchain.add_block(block):
                    self.mempool.on_block_added(block.transactions)
                    self._update_server_chain_state()

                    logger.info(
                        f"Block #{block.height} produced "
                        f"({block.tx_count} txs)"
                    )

                    # Broadcast to all peers
                    await self.server.broadcast(
                        make_new_block(block.to_dict(), self.node_id)
                    )

                    # Broadcast attestation for BFT finality
                    if self.blockchain.consensus:
                        self._broadcast_attestation(block)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Block production error: %s", e, exc_info=True)

    async def _sync_check_loop(self):
        """
        Periodically check if we need to sync (re-sync périodique).
        Request fresh STATUS from peers first so stuck nodes get updated heights.
        """
        while self._running:
            try:
                await asyncio.sleep(10)

                if self.sync._is_sync_active():
                    continue

                # Request STATUS from all peers so we see fresh heights (fixes stuck follower)
                peers = self.peer_manager.get_connected_peers()
                for peer in peers:
                    await self.server.request_status_from_peer(peer.peer_id)
                if peers:
                    await asyncio.sleep(1.5)  # let STATUS replies arrive

                if self.sync.needs_sync(self.blockchain.height):
                    logger.info(
                        "Periodic sync check: behind peers, starting sync (height=%s)",
                        self.blockchain.height,
                    )
                    await self.sync.start_sync(self.blockchain.height)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Sync check error: {e}")

    async def _peer_exchange_loop(self):
        """Periodically request peers and broadcast STATUS so peers get fresh heights."""
        # In local/test, broadcast STATUS more often so followers' peer height stays fresh
        interval = 20 if getattr(
            self.config.network, "network_type", None
        ) == "local" else 60
        while self._running:
            try:
                # Reconnect faster when isolated (no connected peers)
                sleep_time = 10 if len(self.peer_manager.get_connected_peers()) == 0 else interval
                await asyncio.sleep(sleep_time)

                peers = self.peer_manager.get_connected_peers()
                if not peers:
                    # Connection watchdog: no peers at all — try to reconnect to bootstrap
                    bootstrap = getattr(self.config.network, "bootstrap_nodes", [])
                    if bootstrap:
                        logger.warning(
                            "Connection watchdog: 0 connected peers, "
                            "reconnecting to %d bootstrap peer(s)", len(bootstrap)
                        )
                        for url in bootstrap:
                            try:
                                await self.server.connect_to_peer(url)
                            except Exception as e:
                                logger.debug(f"Reconnect to {url}: {e}")
                    continue

                # Request peers from a random connected peer
                import random
                target = random.choice(peers)
                await self.server.request_peers(target.peer_id)

                # Broadcast our status to all peers
                for peer in peers:
                    await self.server.send_status(peer.peer_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Peer exchange error: {e}")

    async def _maintenance_loop(self):
        """
        Periodic maintenance: ping peers, prune stale connections,
        cleanup caches, and log stats.
        """
        while self._running:
            try:
                await asyncio.sleep(30)

                # Ping all connected peers
                for peer in self.peer_manager.get_connected_peers():
                    if peer.is_stale:
                        peer.ping_failures += 1
                        if peer.ping_failures > 3:
                            await self.peer_manager.disconnect_peer(
                                peer.peer_id, "ping timeout"
                            )
                            continue
                    await self.server.ping_peer(peer.peer_id)

                # Prune stale peers
                await self.peer_manager.prune_stale_peers()

                # Cleanup protocol state
                self.server.protocol.cleanup()

                # Clean expired bans
                self.peer_manager.clean_expired_bans()

                # Auto-detect staking: if user staked while app is running,
                # enable Light Validator mode (desktop) or full validator (server)
                if (self.blockchain.consensus
                        and not self.config.validator.enabled
                        and self._light_validator is None):
                    from positronic.constants import MIN_STAKE
                    acc = self.blockchain.state.get_account(self.keypair.address)
                    if acc.staked_amount >= MIN_STAKE:
                        # Desktop apps: activate Light Validator (no block production)
                        from positronic.ai.light_validator import LightValidator
                        gate = getattr(self.blockchain, 'ai_gate', None)
                        self._light_validator = LightValidator(gate)
                        self._background_tasks.append(
                            asyncio.create_task(self._light_validation_loop())
                        )
                        logger.info(
                            "Light Validator auto-activated: staked %d ASF",
                            acc.staked_amount // (10**18),
                        )

                # Complete matured unstaking for all accounts
                for addr, acc in list(self.blockchain.state.accounts.items()):
                    if acc.unstaking_amount > 0:
                        released = self.blockchain.state.complete_unstaking(addr)
                        if released > 0:
                            logger.info("Released %d unstaked ASF for 0x%s",
                                        released // (10**18), addr.hex()[:16])

                # Log periodic stats
                connected = self.peer_manager.connected_count
                if connected > 0 or self.blockchain.height > 0:
                    logger.debug(
                        f"Stats: height={self.blockchain.height}, "
                        f"peers={connected}, "
                        f"mempool={self.mempool.size}, "
                        f"syncing={self.sync.state.syncing}"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Maintenance error: {e}")

    async def _security_kick_loop(self):
        """
        Periodic peer quality enforcement loop (every 60 seconds).

        - Updates the security manager's validator set from the live registry
        - Kicks peers that violate score, idle, or latency thresholds
        - Evicts lowest-scoring peer when node is under resource pressure
        Validators (is_validator=True, score=1000) are always exempt.
        """
        while self._running:
            try:
                await asyncio.sleep(60)

                # Update validator set so staked peers are never kicked
                try:
                    if self.blockchain.consensus:
                        validators = self.blockchain.consensus.registry.active_validators
                        val_addresses = {
                            v.address.hex() if isinstance(v.address, bytes) else str(v.address)
                            for v in validators
                        }
                        self.security_manager.update_validator_set(val_addresses)
                except Exception as e:
                    logger.debug("Security: validator set update failed: %s", e)

                # Skip peer quality enforcement during active sync
                # to avoid kicking peers that are serving block batches
                if getattr(self, 'sync', None) and getattr(self.sync.state, 'syncing', False):
                    continue

                # Build latency map from peer manager
                peer_latencies = {
                    p.peer_id: p.latency_ms
                    for p in self.peer_manager.get_connected_peers()
                }

                # Evaluate and kick bad peers
                to_kick = self.security_manager.get_peers_to_kick(peer_latencies)
                for peer_id, reason in to_kick:
                    logger.info(
                        "Security kick: peer %s (%s)", peer_id[:8], reason
                    )
                    await self.peer_manager.disconnect_peer(peer_id, f"security: {reason}")

                # Resource protection: evict lowest-scoring peer when overloaded
                if self.security_manager.needs_resource_eviction():
                    evict_id = self.security_manager.get_lowest_score_peer()
                    if evict_id:
                        logger.info(
                            "Resource eviction: kicking lowest-score peer %s",
                            evict_id[:8],
                        )
                        await self.peer_manager.disconnect_peer(
                            evict_id, "resource pressure"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Security kick loop error: %s", e)

    async def _time_sync_loop(self):
        """Periodically re-sync NTP time."""
        while self._running:
            try:
                await asyncio.sleep(self.time_sync.NTP_SYNC_INTERVAL)
                if self.time_sync.should_resync():
                    self.time_sync.sync_ntp(timeout=3.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Time sync error: {e}")

    def _register_node_as_validator(self):
        """Register THIS node's keypair in the consensus validator registry.

        Only this running node is registered — ensuring only nodes with
        software open can produce blocks (Ethereum-style).
        """
        try:
            registry = self.blockchain.consensus.registry
            node_addr = self.keypair.address
            node_pubkey = self.keypair.public_key_bytes
            acc = self.blockchain.state.get_account(node_addr)

            from positronic.constants import MIN_STAKE
            if acc.staked_amount < MIN_STAKE:
                if acc.effective_balance >= MIN_STAKE:
                    self.blockchain.state.stake(node_addr, MIN_STAKE)
                    acc = self.blockchain.state.get_account(node_addr)
                    logger.info("Auto-staked %d ASF for validator: %s",
                                MIN_STAKE // (10**18), self.keypair.address_hex)
                else:
                    logger.info("Insufficient balance for validator (%d ASF available)",
                                acc.effective_balance // (10**18))
                    return

            # Store real pubkey on account
            acc.is_validator = True
            acc.validator_pubkey = node_pubkey
            self.blockchain.state._sync_account_to_trie(node_addr)

            # Register in consensus registry with real keypair
            if node_addr in registry._validators:
                v = registry._validators[node_addr]
                v.stake = acc.staked_amount
                v.pubkey = node_pubkey
                registry.activate(node_addr)
            else:
                registry.register(pubkey=node_pubkey, stake=acc.staked_amount)

            # Re-run election
            current_slot = self.blockchain.height + 1
            from positronic.constants import SLOTS_PER_EPOCH
            current_epoch = current_slot // SLOTS_PER_EPOCH
            last_hash = (self.blockchain.chain_head.hash
                         if self.blockchain.chain_head else b"\x00" * 64)
            epoch_seed = self.blockchain.consensus.election.derive_epoch_seed(
                last_hash, current_epoch)
            self.blockchain.consensus.election._history.pop(current_epoch, None)
            self.blockchain.consensus.election.run_election(
                epoch=current_epoch, epoch_seed=epoch_seed)

            logger.info("Validator registered: %s (stake: %d ASF, active: %d)",
                        self.keypair.address_hex,
                        acc.staked_amount // (10**18),
                        registry.active_count)
        except Exception as e:
            logger.warning("Validator registration failed: %s", e)

    # === Helpers ===

    def _bootstrap_ai_if_needed(self):
        """Bootstrap the AI anomaly detector if it hasn't been trained yet.

        Uses synthetic normal transaction data so the TAD autoencoder can
        distinguish anomalous patterns from genesis — before any real
        transaction history is available.
        """
        try:
            gate = self.blockchain.ai_gate
            if gate and hasattr(gate, 'anomaly_detector'):
                detector = gate.anomaly_detector
                if not getattr(detector, 'trained', False):
                    from positronic.ai.training.bootstrap import bootstrap_gate
                    bootstrap_gate(gate)
                    logger.info(
                        "AI anomaly detector bootstrapped with synthetic data "
                        "(trained=%s, samples=%s)",
                        detector.trained,
                        getattr(detector, 'training_samples', 'N/A'),
                    )
                else:
                    logger.debug(
                        "AI anomaly detector already trained (%s samples)",
                        getattr(detector, 'training_samples', 'N/A'),
                    )
        except Exception as e:
            logger.warning("AI bootstrap failed (non-fatal): %s", e)

    def _update_server_chain_state(self):
        """Update the server with current chain state."""
        best_hash = ""
        if self.blockchain.chain_head:
            best_hash = self.blockchain.chain_head.hash_hex
        self.server.set_chain_state(self.blockchain.height, best_hash)

    # === Public API ===

    def get_info(self) -> dict:
        """Get comprehensive node information."""
        return {
            "node_id": self.node_id,
            "address": self.keypair.address_hex,
            "chain_height": self.blockchain.height,
            "peers": self.peer_manager.connected_count,
            "mempool": self.mempool.size,
            "validator": self.config.validator.enabled,
            "light_validator": self._light_validator is not None,
            "ai_enabled": self.blockchain.ai_gate.ai_enabled,
            "syncing": self.sync.state.syncing,
            "sync_progress": f"{self.sync.state.progress:.1%}",
            "running": self._running,
            "time_synced": self.time_sync.state.synced,
            "rpc_port": self.config.network.rpc_port,
        }

    def get_detailed_stats(self) -> dict:
        """Get detailed statistics for all subsystems."""
        return {
            "node": self.get_info(),
            "blockchain": self.blockchain.get_stats(),
            "mempool": self.mempool.get_stats(),
            "network": self.server.get_stats(),
            "peers": self.peer_manager.get_stats(),
            "sync": self.sync.get_stats(),
            "discovery": self.discovery.get_stats(),
            "time_sync": self.time_sync.get_stats(),
        }
