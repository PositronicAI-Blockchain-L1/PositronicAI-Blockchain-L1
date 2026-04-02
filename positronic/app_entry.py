"""
Positronic Desktop Application — PyInstaller entry point.

Handles frozen-app bootstrap:
1. Determine platform-specific writable data directory
2. Set up logging
3. Launch the GUI application
"""

import logging
import multiprocessing
import os
import re
import sys


def get_data_dir() -> str:
    """
    Determine the writable data directory for the current platform.

    Windows:  %APPDATA%/Positronic/
    macOS:    ~/Library/Application Support/Positronic/
    Linux:    ~/.positronic/
    """
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
    """Configure logging to both console and log file with IP masking."""
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "positronic.log")

    fmt = _IPMaskFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
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


def main():
    # Required for Windows frozen multiprocessing (PyInstaller)
    multiprocessing.freeze_support()

    # Verify cryptography library is available (hard requirement)
    try:
        import cryptography  # noqa: F401
    except ImportError:
        print(
            "\n[FATAL] The 'cryptography' package is required.\n"
            "Install it with: pip install cryptography>=42.0.0\n"
            "Validator data cannot be secured without AES-256-GCM encryption.\n"
        )
        sys.exit(1)

    # Runtime security checks (only in compiled binary)
    if getattr(sys, 'frozen', False) or '__compiled__' in globals():
        try:
            from positronic.security.integrity import IntegrityChecker
            from positronic.security.anti_debug import AntiDebug

            if not IntegrityChecker.verify():
                logger = logging.getLogger("positronic.security")
                logger.critical(
                    "Binary integrity check FAILED — application may be tampered. "
                    "Refusing to start."
                )
                print(
                    "\n[SECURITY] Binary integrity check failed.\n"
                    "The application binary has been modified and cannot be trusted.\n"
                    "Please download a fresh copy from https://positronic-ai.network/download/\n"
                )
                sys.exit(1)

            AntiDebug.start_monitoring(interval=30)
        except ImportError:
            pass  # Security modules not available (development mode)

    # Parse optional --founder flag
    founder_mode = "--founder" in sys.argv

    # Determine data directory
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)

    # Set environment so all positronic modules use this data dir
    os.environ["POSITRONIC_DATA_DIR"] = data_dir

    # Set up logging
    setup_logging(data_dir)

    logger = logging.getLogger("positronic.app")
    logger.info("Positronic Desktop Application starting")
    logger.info("Data directory: %s", data_dir)
    logger.info("Platform: %s", sys.platform)
    logger.info("Founder mode: %s", founder_mode)

    # Launch the GUI
    from positronic.app.app import PositronicApp
    app = PositronicApp(data_dir=data_dir, founder_mode=founder_mode)
    app.run()


if __name__ == "__main__":
    main()
