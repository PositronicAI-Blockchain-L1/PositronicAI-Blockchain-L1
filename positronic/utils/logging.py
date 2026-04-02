"""
Positronic - Centralized Logging Configuration

Structured logging with consistent formatting across all modules.
Every module should use: logger = get_logger(__name__)
"""

import logging
import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class PositronicFormatter(logging.Formatter):
    """Compact, colourless structured formatter for production logs.

    Format: ``TIMESTAMP LEVEL [module] message  key=value ...``
    IPv4 addresses are masked (last two octets hidden) for security.
    """

    FMT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"
    _IP_RE = re.compile(r'\b(\d{1,3}\.\d{1,3})\.\d{1,3}\.\d{1,3}\b')

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return self._IP_RE.sub(r'\1.x.x', msg)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_configured: bool = False


def _ensure_root_handler() -> None:
    """Attach a single stderr handler to the *positronic* root logger."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("positronic")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(PositronicFormatter())
        root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``positronic`` namespace.

    Usage::

        from positronic.utils.logging import get_logger
        logger = get_logger(__name__)
        logger.info("started", extra={"port": 8545})
    """
    _ensure_root_handler()
    # Strip the 'positronic.' prefix if the caller already passes __name__
    if name.startswith("positronic."):
        return logging.getLogger(name)
    return logging.getLogger(f"positronic.{name}")


def set_level(level: int) -> None:
    """Change the global positronic log level at runtime."""
    logging.getLogger("positronic").setLevel(level)
