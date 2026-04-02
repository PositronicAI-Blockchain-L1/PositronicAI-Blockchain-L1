"""
Positronic - Address Book

Named address management for the wallet. Allows users to assign
human-readable names to frequently used addresses. Persists to
a JSON file.

Phase 17 GOD CHAIN addition.
"""

import json
import time
from typing import List, Dict, Optional

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


class AddressBook:
    """Named address book for the wallet.

    Supports add, remove, search, and name-to-address resolution.
    Optionally persists to a JSON file.

    Example::

        book = AddressBook()
        book.add("alice", "0x1234...", label="Alice's main wallet")
        addr = book.resolve("alice")  # → "0x1234..."
        addr = book.resolve("0x5678...")  # → "0x5678..." (passthrough)
    """

    def __init__(self, filepath: Optional[str] = None):
        """Initialize the address book.

        Args:
            filepath: Optional path to JSON file for persistence.
                If None, the book is in-memory only.
        """
        self._filepath = filepath
        self._entries: Dict[str, dict] = {}  # name → {address, label, tags, created_at}
        if filepath:
            self._load()

    def add(
        self,
        name: str,
        address: str,
        label: str = "",
        tags: Optional[List[str]] = None,
    ):
        """Add or update a named address.

        Args:
            name: Human-readable name (case-insensitive).
            address: Blockchain address (hex string).
            label: Optional description.
            tags: Optional list of tags for categorization.
        """
        key = name.lower().strip()
        if not key:
            raise ValueError("Name cannot be empty")

        self._entries[key] = {
            "name": name.strip(),
            "address": self._normalize_addr(address),
            "label": label,
            "tags": tags or [],
            "created_at": time.time(),
        }
        self._save()

    def remove(self, name: str) -> bool:
        """Remove a named address.

        Args:
            name: Name to remove (case-insensitive).

        Returns:
            True if the entry was removed, False if not found.
        """
        key = name.lower().strip()
        if key in self._entries:
            del self._entries[key]
            self._save()
            return True
        return False

    def get(self, name: str) -> Optional[dict]:
        """Get an address book entry by name.

        Args:
            name: Name to look up (case-insensitive).

        Returns:
            Entry dictionary or None if not found.
        """
        key = name.lower().strip()
        return self._entries.get(key)

    def resolve(self, name_or_address: str) -> str:
        """Resolve a name to an address, or return the address as-is.

        If the input looks like an address (starts with 0x and is 42 chars),
        it is returned unchanged. Otherwise, it is looked up in the book.

        Args:
            name_or_address: Name or hex address.

        Returns:
            Resolved address string.

        Raises:
            KeyError: If the name is not found in the address book.
        """
        s = name_or_address.strip()
        # If it looks like an address, return as-is
        if s.startswith("0x") and len(s) == 42:
            return s.lower()

        key = s.lower()
        entry = self._entries.get(key)
        if entry:
            return entry["address"]

        raise KeyError(f"Name '{s}' not found in address book")

    def search(self, query: str) -> List[dict]:
        """Search entries by name, address, label, or tag.

        Args:
            query: Search string (case-insensitive).

        Returns:
            List of matching entries.
        """
        q = query.lower().strip()
        results = []
        for entry in self._entries.values():
            if (q in entry["name"].lower()
                    or q in entry["address"].lower()
                    or q in entry["label"].lower()
                    or any(q in tag.lower() for tag in entry.get("tags", []))):
                results.append(entry.copy())
        return results

    def list_all(self) -> List[dict]:
        """Get all address book entries sorted by name.

        Returns:
            Sorted list of all entries.
        """
        return sorted(
            [e.copy() for e in self._entries.values()],
            key=lambda e: e["name"].lower(),
        )

    @property
    def count(self) -> int:
        """Number of entries in the address book."""
        return len(self._entries)

    def _save(self):
        """Persist to JSON file (if filepath was provided)."""
        if not self._filepath:
            return
        try:
            with open(self._filepath, "w") as f:
                json.dump(self._entries, f, indent=2)
        except Exception as e:
            logger.warning("Address book save failed: %s", e)

    def _load(self):
        """Load from JSON file (if filepath was provided)."""
        if not self._filepath:
            return
        try:
            with open(self._filepath, "r") as f:
                self._entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._entries = {}

    @staticmethod
    def _normalize_addr(address: str) -> str:
        """Normalize address to lowercase with 0x prefix."""
        addr = address.strip().lower()
        if not addr.startswith("0x"):
            addr = "0x" + addr
        return addr
