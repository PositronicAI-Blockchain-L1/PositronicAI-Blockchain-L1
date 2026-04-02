"""
Positronic Headless Node — Entry point for Electron-managed child process.

Runs the validator node + RPC server without any GUI.
Electron spawns this as a child process and communicates via JSON-RPC on localhost.

Exit codes:
  0 = clean shutdown
  1 = fatal error
  2 = port already in use
"""

import asyncio
import logging
import multiprocessing
import os
import re
import signal
import sys


def get_data_dir() -> str:
    """Platform-specific data directory (can be overridden via env)."""
    env_dir = os.environ.get("POSITRONIC_DATA_DIR")
    if env_dir:
        return env_dir

    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "Positronic")
    elif sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~"),
            "Library", "Application Support", "Positronic",
        )
    else:
        return os.path.join(os.path.expanduser("~"), ".positronic")


class _IPMaskFormatter(logging.Formatter):
    """Masks IPv4 addresses in log output — keeps first two octets, hides last two."""
    _IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b')

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return self._IP_RE.sub(r'\1.x.x', msg)


def setup_logging(data_dir: str):
    """Configure logging to stdout + file with IP masking."""
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "positronic.log")

    fmt = _IPMaskFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()  # Remove any pre-existing handlers
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)


async def run_node(data_dir: str, founder_mode: bool = False):
    """Run the node until shutdown signal."""
    from positronic.network.node import Node
    from positronic.utils.config import NodeConfig

    logger = logging.getLogger("positronic.node")

    config = NodeConfig()
    config.storage.data_dir = data_dir

    # RPC: localhost only (Electron connects locally)
    config.network.rpc_host = os.environ.get("POSITRONIC_RPC_HOST", "127.0.0.1")
    config.network.rpc_port = int(os.environ.get("POSITRONIC_RPC_PORT", "8545"))

    # P2P: open to peers
    config.network.p2p_host = os.environ.get("POSITRONIC_P2P_HOST", "0.0.0.0")
    config.network.p2p_port = int(os.environ.get("POSITRONIC_P2P_PORT", "9000"))

    # Network type
    config.network.network_type = os.environ.get("POSITRONIC_NETWORK", "testnet")

    # Validator mode: controlled by --validator / --no-validator flag from Electron
    validator_flag = "--no-validator" not in sys.argv
    config.validator.enabled = validator_flag

    # Settings from Electron desktop app (passed via environment variables)
    max_peers = os.environ.get("POSITRONIC_MAX_PEERS")
    if max_peers and max_peers.isdigit() and int(max_peers) > 0:
        config.network.max_peers = int(max_peers)

    log_level = os.environ.get("POSITRONIC_LOG_LEVEL")
    if log_level and log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
        config.log_level = log_level.upper()
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))

    ai_enabled = os.environ.get("POSITRONIC_AI_ENABLED")
    if ai_enabled is not None:
        config.ai.enabled = ai_enabled.lower() in ("1", "true", "yes")

    logger.info("Starting Positronic Node (headless)")
    logger.info("  Data: %s", data_dir)
    logger.info("  RPC:  %s:%d", config.network.rpc_host, config.network.rpc_port)
    logger.info("  P2P:  %s:%d", config.network.p2p_host, config.network.p2p_port)
    logger.info("  Network: %s", config.network.network_type)

    node = Node(config)
    shutdown_event = asyncio.Event()

    # Handle shutdown signals
    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
    else:
        # Windows: handle via thread
        signal.signal(signal.SIGTERM, lambda s, f: _signal_handler())
        signal.signal(signal.SIGINT, lambda s, f: _signal_handler())

    try:
        await node.start(founder_mode=founder_mode)
        logger.info("Node started — waiting for shutdown signal")

        # Wait for shutdown
        await shutdown_event.wait()

    except OSError as e:
        if "10048" in str(e) or "Address already in use" in str(e):
            logger.error("Port already in use: %s", e)
            sys.exit(2)
        raise
    finally:
        logger.info("Shutting down node...")
        try:
            await node.stop()
        except Exception as e:
            logger.error("Error during shutdown: %s", e)
        logger.info("Node stopped cleanly")


def main():
    multiprocessing.freeze_support()

    # Verify cryptography
    try:
        import cryptography  # noqa: F401
    except ImportError:
        print("[FATAL] 'cryptography' package required")
        sys.exit(1)

    # Security checks (compiled binary only)
    if getattr(sys, 'frozen', False) or '__compiled__' in globals():
        try:
            from positronic.security.integrity import IntegrityChecker
            from positronic.security.anti_debug import AntiDebug

            if not IntegrityChecker.verify():
                print("[SECURITY] Binary integrity check failed")
                sys.exit(1)

            AntiDebug.start_monitoring(interval=30)
        except ImportError:
            pass

    founder_mode = "--founder" in sys.argv

    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    os.environ["POSITRONIC_DATA_DIR"] = data_dir

    setup_logging(data_dir)

    logger = logging.getLogger("positronic.node")
    logger.info("Positronic Headless Node starting")

    # Use SelectorEventLoop on Windows for aiohttp compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run_node(data_dir, founder_mode))
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except SystemExit:
        raise
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
