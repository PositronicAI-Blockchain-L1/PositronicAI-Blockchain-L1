"""
Positronic - PositronicVM Contract Storage

Interface layer between the VM and the world state (StateManager).
Provides key-value storage scoped per contract address.
All keys and values are 32 bytes (256 bits).
"""

from typing import Dict, Optional, Set

from positronic.core.state import StateManager


class ContractStorage:
    """
    Contract storage interface bridging the VM to StateManager.

    Provides per-contract persistent key-value storage where both
    keys and values are 32-byte (256-bit) quantities. Maintains a
    write cache for uncommitted changes and tracks original values
    for gas refund calculations.

    Usage:
        storage = ContractStorage(state_manager, contract_address)
        value = storage.load(key)
        storage.store(key, new_value)
    """

    SLOT_SIZE = 32  # 32 bytes per storage slot

    def __init__(self, state: StateManager, contract_address: bytes):
        """
        Initialize storage for a specific contract.

        Args:
            state: The global StateManager instance.
            contract_address: The 20-byte address of the contract.
        """
        self._state = state
        self._address = contract_address
        self._cache: Dict[bytes, bytes] = {}
        self._original: Dict[bytes, bytes] = {}
        self._dirty_keys: Set[bytes] = set()

    @property
    def address(self) -> bytes:
        """The contract address this storage is scoped to."""
        return self._address

    def _normalize_key(self, key: int) -> bytes:
        """
        Convert a uint256 key to a 32-byte bytes key.

        Args:
            key: 256-bit unsigned integer key.

        Returns:
            32-byte big-endian representation.
        """
        return key.to_bytes(self.SLOT_SIZE, "big")

    def _normalize_value(self, value: int) -> bytes:
        """
        Convert a uint256 value to 32-byte bytes.

        Args:
            value: 256-bit unsigned integer value.

        Returns:
            32-byte big-endian representation.
        """
        return (value & ((1 << 256) - 1)).to_bytes(self.SLOT_SIZE, "big")

    def _bytes_to_int(self, data: bytes) -> int:
        """Convert 32-byte value back to uint256."""
        return int.from_bytes(data, "big")

    def load(self, key: int) -> int:
        """
        Load a value from storage.

        First checks the write cache, then falls back to the StateManager.

        Args:
            key: 256-bit storage slot key.

        Returns:
            256-bit unsigned integer value (0 if slot is empty).
        """
        key_bytes = self._normalize_key(key)

        # Check write cache first
        if key_bytes in self._cache:
            return self._bytes_to_int(self._cache[key_bytes])

        # Load from state
        value = self._state.get_storage(self._address, key_bytes)
        self._cache[key_bytes] = value

        # Track original value for gas refund calculation
        if key_bytes not in self._original:
            self._original[key_bytes] = value

        return self._bytes_to_int(value)

    def store(self, key: int, value: int) -> None:
        """
        Store a value to storage.

        Writes are cached and only flushed to StateManager on commit.

        Args:
            key: 256-bit storage slot key.
            value: 256-bit unsigned integer value to store.
        """
        key_bytes = self._normalize_key(key)
        value_bytes = self._normalize_value(value)

        # Track original value if not already tracked
        if key_bytes not in self._original:
            original = self._state.get_storage(self._address, key_bytes)
            self._original[key_bytes] = original

        self._cache[key_bytes] = value_bytes
        self._dirty_keys.add(key_bytes)

    def is_slot_warm(self, key: int) -> bool:
        """
        Check if a storage slot has been accessed in this execution.
        Warm slots have lower gas costs.

        Args:
            key: 256-bit storage slot key.

        Returns:
            True if the slot is in the cache (warm).
        """
        return self._normalize_key(key) in self._cache

    def get_original_value(self, key: int) -> int:
        """
        Get the original value of a slot (before any writes in this execution).
        Used for gas refund calculations (EIP-2200 style).

        Args:
            key: 256-bit storage slot key.

        Returns:
            Original 256-bit value.
        """
        key_bytes = self._normalize_key(key)
        if key_bytes in self._original:
            return self._bytes_to_int(self._original[key_bytes])
        # Not yet accessed; load from state
        value = self._state.get_storage(self._address, key_bytes)
        self._original[key_bytes] = value
        return self._bytes_to_int(value)

    def calculate_refund(self) -> int:
        """
        Calculate gas refund for storage operations.

        Refund is granted when a non-zero slot is set to zero.
        Based on EIP-3529 rules (reduced refund post-London).

        Returns:
            Total gas refund amount.
        """
        from positronic.vm.opcodes import SSTORE_REFUND_GAS

        refund = 0
        zero_bytes = b"\x00" * self.SLOT_SIZE

        for key_bytes in self._dirty_keys:
            original = self._original.get(key_bytes, zero_bytes)
            current = self._cache.get(key_bytes, zero_bytes)

            # Refund when clearing a slot (non-zero -> zero)
            if original != zero_bytes and current == zero_bytes:
                refund += SSTORE_REFUND_GAS

        return refund

    def commit(self) -> None:
        """
        Flush all dirty storage writes to the StateManager.
        Called after successful execution.
        """
        for key_bytes in self._dirty_keys:
            value_bytes = self._cache[key_bytes]
            self._state.set_storage(self._address, key_bytes, value_bytes)
        self._dirty_keys.clear()

    def revert(self) -> None:
        """
        Discard all uncommitted storage writes.
        Called on execution failure or REVERT.
        """
        # Restore original values in cache
        for key_bytes in self._dirty_keys:
            if key_bytes in self._original:
                self._cache[key_bytes] = self._original[key_bytes]
            else:
                self._cache.pop(key_bytes, None)
        self._dirty_keys.clear()

    @property
    def dirty_count(self) -> int:
        """Number of storage slots modified but not yet committed."""
        return len(self._dirty_keys)

    def clear_cache(self) -> None:
        """Clear all cached data and dirty tracking."""
        self._cache.clear()
        self._original.clear()
        self._dirty_keys.clear()

    def __repr__(self) -> str:
        addr_hex = "0x" + self._address.hex()[:10] + "..."
        return (
            f"ContractStorage(contract={addr_hex}, "
            f"cached={len(self._cache)}, dirty={self.dirty_count})"
        )
