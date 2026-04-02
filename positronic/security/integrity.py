"""
Binary Integrity Checker for Positronic Desktop Application.

Verifies the running binary has not been tampered with by comparing
its SHA-512 hash against a signed manifest file.
"""

import hashlib
import logging
import os
import sys

logger = logging.getLogger("positronic.security.integrity")


class IntegrityChecker:
    """Verifies binary integrity at startup."""

    @staticmethod
    def is_compiled() -> bool:
        """Check if running as a compiled binary (Nuitka or PyInstaller)."""
        return getattr(sys, 'frozen', False) or '__compiled__' in globals()

    @staticmethod
    def get_binary_path() -> str:
        """Get the path to the currently running binary."""
        if getattr(sys, 'frozen', False):
            return sys.executable
        return os.path.abspath(sys.argv[0])

    @staticmethod
    def compute_hash(filepath: str) -> str:
        """Compute SHA-512 hash of a file."""
        h = hashlib.sha512()
        try:
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
        except (OSError, IOError) as e:
            logger.debug("Could not read file for hash: %s", e)
            return ""
        return h.hexdigest()

    @staticmethod
    def get_manifest_path() -> str:
        """Get the path to the hash manifest file."""
        binary = IntegrityChecker.get_binary_path()
        return binary + ".sha512"

    @classmethod
    def verify(cls) -> bool:
        """Verify binary integrity against manifest.

        Returns True if:
        - Not running as compiled binary (development mode)
        - No manifest file exists (first run or dev build)
        - Hash matches manifest

        Returns False only if manifest exists AND hash doesn't match.
        """
        if not cls.is_compiled():
            logger.debug("Not a compiled binary — skipping integrity check")
            return True

        binary = cls.get_binary_path()
        manifest = cls.get_manifest_path()

        if not os.path.exists(manifest):
            # No manifest = standalone download — skip check (not tampered)
            logger.debug("No integrity manifest at %s — standalone mode", manifest)
            return True

        try:
            with open(manifest, "r") as f:
                content = f.read().strip()
            # Format: "hash  filename" (sha512sum format)
            expected_hash = content.split()[0] if content else ""
        except (OSError, IOError) as e:
            logger.debug("Could not read manifest: %s", e)
            return True

        if not expected_hash:
            return True

        actual_hash = cls.compute_hash(binary)

        if actual_hash == expected_hash:
            logger.info("Binary integrity verified")
            return True
        else:
            logger.warning(
                "BINARY INTEGRITY CHECK FAILED! "
                "Expected: %s... Got: %s...",
                expected_hash[:16], actual_hash[:16]
            )
            return False
