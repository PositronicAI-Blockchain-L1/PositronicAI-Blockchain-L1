"""
NodeRunner — runs the Positronic Node in a background daemon thread.

The background thread owns its own asyncio event loop, completely isolated
from the tkinter main thread. Communication happens via thread-safe primitives
(threading.Event) and the localhost JSON-RPC endpoint.
"""

import asyncio
import logging
import sys
import threading
from enum import Enum
from typing import Optional

logger = logging.getLogger("positronic.app.node_runner")


class NodeState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    SYNCING = "syncing"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    LIGHT_VALIDATING = "light_validating"  # Running + Light Validator mode


class NodeRunner:
    """Wraps the existing Node class in a background thread with its own event loop."""

    MAX_AUTO_RESTARTS = 3
    AUTO_RESTART_DELAY = 10  # seconds

    def __init__(self, data_dir: str, *, network_type: str = "testnet",
                 rpc_host: str = "127.0.0.1"):
        self.data_dir = data_dir
        self._network_type = network_type
        self._rpc_host = rpc_host
        self._state = NodeState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._node = None
        self._stop_event = threading.Event()
        self._error: Optional[str] = None
        self._crash_count = 0
        self._founder_mode = False
        self._validator = False  # Desktop app: sync-only by default, not validator

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state in (NodeState.RUNNING, NodeState.LIGHT_VALIDATING)

    @property
    def is_syncing(self) -> bool:
        return self._state == NodeState.SYNCING

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def is_light_validator(self) -> bool:
        """True if node is running in Light Validator mode (staked, TAD-only, no block production)."""
        if self._node and hasattr(self._node, '_light_validator'):
            return self._node._light_validator is not None
        return False

    @property
    def node_mode_label(self) -> str:
        """Human-readable label for the current node mode (for UI display)."""
        if not self.is_running:
            return self._state.value
        if self._node and getattr(self._node.config.validator, 'enabled', False):
            return "Full Validator"
        if self.is_light_validator:
            return "Light Validator \u2014 protecting the network"
        return "Sync Node"

    def start(self, founder_mode: bool = False, validator: bool = False):
        """Start the node in a background daemon thread."""
        if self._state in (NodeState.RUNNING, NodeState.STARTING):
            logger.warning("Node already running or starting")
            return

        self._state = NodeState.STARTING
        self._error = None
        self._stop_event.clear()
        self._founder_mode = founder_mode
        self._validator = validator

        self._thread = threading.Thread(
            target=self._run_loop,
            args=(founder_mode, validator),
            daemon=True,
            name="positronic-node",
        )
        self._thread.start()
        logger.info("Node thread started (founder=%s, validator=%s)",
                     founder_mode, validator)

    def stop(self):
        """Request graceful shutdown of the node."""
        if self._state not in (NodeState.RUNNING, NodeState.STARTING, NodeState.LIGHT_VALIDATING):
            return

        logger.info("Requesting node shutdown...")
        self._state = NodeState.STOPPING
        self._stop_event.set()

        # Schedule stop() coroutine on the node's event loop
        if self._loop and self._node and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self._node.stop(), self._loop)
            except Exception as exc:
                logger.debug("Failed to schedule node.stop(): %s", exc)

        # Wait for thread to finish (short timeout to avoid UI freeze)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                logger.warning("Node thread didn't stop in 3s — forcing")
                # Force close the event loop to unblock async operations
                if self._loop and not self._loop.is_closed():
                    try:
                        self._loop.call_soon_threadsafe(self._loop.stop)
                    except Exception:
                        pass
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    # Last resort: mark thread as daemon so it dies with process
                    logger.warning("Node thread still alive — abandoning")
                    self._thread.daemon = True

        self._state = NodeState.STOPPED
        self._thread = None
        self._loop = None
        self._node = None
        logger.info("Node stopped")

    # ------------------------------------------------------------------

    def _encrypt_sensitive_files(self):
        """Encrypt all sensitive files in the data directory on startup.

        Handles backward compatibility: plaintext files from older versions
        are transparently migrated to encrypted format.  Files that are
        already encrypted are skipped.
        """
        import os
        try:
            from positronic.crypto.data_encryption import DataEncryptor, _restrict_permissions
            enc = DataEncryptor(self.data_dir)

            # Files to encrypt: keypair, settings, admin key
            sensitive_files = [
                os.path.join(self.data_dir, "node_keypair.bin"),
                os.path.join(self.data_dir, "admin.key"),
            ]

            # Settings may be in a different location (AppData/Positronic)
            from positronic.app.ui.tab_settings import _get_settings_path
            try:
                settings_path = _get_settings_path()
                if os.path.isfile(settings_path):
                    settings_enc = DataEncryptor(os.path.dirname(settings_path))
                    settings_enc.encrypt_file(settings_path)
            except Exception as e:
                logger.debug("Settings encryption skip: %s", e)

            for fpath in sensitive_files:
                if os.path.isfile(fpath):
                    enc.encrypt_file(fpath)

            # Restrict permissions on all data files (databases, salt, etc.)
            all_data_files = [
                os.path.join(self.data_dir, f)
                for f in os.listdir(self.data_dir)
                if os.path.isfile(os.path.join(self.data_dir, f))
            ]
            for fpath in all_data_files:
                _restrict_permissions(fpath)

            logger.info("Sensitive data files encrypted and permissions restricted")
        except Exception as exc:
            logger.warning("Data encryption on startup failed: %s", exc)

    def _run_loop(self, founder_mode: bool, validator: bool):
        """Thread target: create event loop, run node, handle shutdown."""
        # On Windows, aiohttp needs SelectorEventLoop (not Proactor)
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()
            )

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(
                self._run_node(founder_mode, validator)
            )
        except Exception as exc:
            logger.exception("Node crashed: %s", exc)
            self._error = str(exc)
            self._state = NodeState.ERROR
            self._crash_count += 1
        finally:
            try:
                self._loop.run_until_complete(
                    self._loop.shutdown_asyncgens()
                )
            except Exception as e:
                logger.debug("Async generator shutdown error: %s", e)
            self._loop.close()
            self._loop = None
            if self._state != NodeState.ERROR:
                self._state = NodeState.STOPPED
            # Auto-restart on crash (up to MAX_AUTO_RESTARTS)
            elif (self._crash_count <= self.MAX_AUTO_RESTARTS
                  and not self._stop_event.is_set()):
                import time as _time
                delay = self.AUTO_RESTART_DELAY * self._crash_count
                logger.info("Auto-restarting node in %ds (crash #%d/%d)",
                            delay, self._crash_count, self.MAX_AUTO_RESTARTS)
                _time.sleep(delay)
                if not self._stop_event.is_set():
                    self._state = NodeState.STOPPED
                    self.start(self._founder_mode, self._validator)

    async def _run_node(self, founder_mode: bool, validator: bool):
        """Async node lifecycle inside the background thread."""
        from positronic.network.node import Node
        from positronic.utils.config import NodeConfig

        config = NodeConfig()
        config.storage.data_dir = self.data_dir
        config.validator.enabled = validator

        # RPC host: localhost for desktop, 0.0.0.0 for VPS/testnet
        config.network.rpc_host = self._rpc_host
        # P2P must be open so peers can connect
        config.network.p2p_host = "0.0.0.0"
        # Network type from constructor (overridable via env var)
        import os
        config.network.network_type = os.environ.get(
            "POSITRONIC_NETWORK", self._network_type
        )

        # Populate bootstrap nodes from discovery module's testnet seeds
        from positronic.network.discovery import PeerDiscovery
        config.network.bootstrap_nodes = PeerDiscovery.TESTNET_SEEDS

        self._node = Node(config)

        # Encrypt sensitive data files on startup (migration for existing installs)
        self._encrypt_sensitive_files()

        self._state = NodeState.SYNCING
        logger.info("Node syncing with network...")
        await self._node.start(founder_mode=founder_mode)

        # Wait for initial sync to complete before declaring RUNNING
        # Check sync module first; fallback to height stability check
        sync_wait = 0
        last_height = 0
        stable_count = 0  # consecutive polls where height didn't change fast
        while not self._stop_event.is_set():
            if hasattr(self._node, 'sync') and hasattr(self._node.sync, 'state'):
                if not self._node.sync.state.syncing:
                    break
            elif hasattr(self._node, 'blockchain') and self._node.blockchain:
                current_height = self._node.blockchain.height
                # Consider synced if height > 0 and stable for 3 consecutive checks
                if current_height > 0:
                    if current_height == last_height:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_height = current_height
                    if stable_count >= 3:
                        break
            sync_wait += 1
            if sync_wait % 10 == 0:
                h = self._node.blockchain.height if hasattr(self._node, 'blockchain') and self._node.blockchain else 0
                logger.info("Syncing... height: %d", h)
            await asyncio.sleep(1)
            # Safety: don't wait forever — transition after 60 seconds
            if sync_wait > 60:
                logger.warning("Sync taking too long — transitioning to RUNNING anyway")
                break

        if hasattr(self._node, 'blockchain') and self._node.blockchain:
            local_height = self._node.blockchain.height
            logger.info("Sync complete — local height: %d", local_height)

        # Detect Light Validator mode for UI state
        if (hasattr(self._node, '_light_validator')
                and self._node._light_validator is not None):
            self._state = NodeState.LIGHT_VALIDATING
            logger.info("Node is fully running (Light Validator mode)")
        else:
            self._state = NodeState.RUNNING
            logger.info("Node is fully running")

        # Initialize AI models if not present
        try:
            import os as _os
            ai_models_dir = _os.path.join(self.data_dir, "ai_models")
            if _os.path.isdir(ai_models_dir) and not _os.listdir(ai_models_dir):
                logger.info("ai_models directory empty — triggering AI pre-training...")
                import threading as _threading
                def _pretrain():
                    try:
                        from positronic.ai.attack_training_data import pretrain_ai_models
                        from positronic.ai.anomaly_detector import AnomalyDetector
                        detector = AnomalyDetector(model_dir=ai_models_dir)
                        pretrain_ai_models(detector)
                        logger.info("AI pre-training complete")
                    except Exception as e:
                        logger.debug("AI pre-train skipped: %s", e)
                _threading.Thread(target=_pretrain, daemon=True).start()
        except Exception as e:
            logger.debug("AI init check: %s", e)

        # Wait until stop is requested
        while not self._stop_event.is_set():
            # Check if Light Validator was activated while running
            if (self._state == NodeState.RUNNING
                    and hasattr(self._node, '_light_validator')
                    and self._node._light_validator is not None):
                self._state = NodeState.LIGHT_VALIDATING
                logger.info("Light Validator mode activated")
            await asyncio.sleep(0.5)

        # Graceful shutdown
        await self._node.stop()
        self._node = None
