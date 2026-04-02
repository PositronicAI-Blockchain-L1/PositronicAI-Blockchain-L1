"""
Positronic - PositronicVM Memory

Expandable byte-addressable memory for the virtual machine.
Maximum size is 1MB as defined in positronic.constants.MAX_MEMORY.
Memory is zero-initialized and grows in 32-byte word increments.
"""

from positronic.constants import MAX_MEMORY


class MemoryLimitError(Exception):
    """Raised when memory expansion exceeds the maximum allowed size."""
    pass


class MemoryAccessError(Exception):
    """Raised on invalid memory access."""
    pass


class VMMemory:
    """
    PositronicVM byte-addressable memory.

    Memory is expandable up to MAX_MEMORY (1MB) bytes.
    Grows in 32-byte word boundaries. Zero-initialized on expansion.
    Tracks the highest accessed offset for gas cost calculation.
    """

    WORD_SIZE = 32

    def __init__(self, max_size: int = MAX_MEMORY):
        self._data = bytearray()
        self._max_size = max_size

    @property
    def size(self) -> int:
        """Current size of memory in bytes (always a multiple of 32)."""
        return len(self._data)

    @property
    def word_count(self) -> int:
        """Current size of memory in 32-byte words."""
        return len(self._data) // self.WORD_SIZE

    def _expand_to(self, offset: int, length: int) -> int:
        """
        Expand memory to accommodate access at offset with given length.
        Memory grows to the next 32-byte word boundary.

        Args:
            offset: Starting byte offset of the access.
            length: Number of bytes being accessed.

        Returns:
            Number of new words allocated (for gas cost calculation).

        Raises:
            MemoryLimitError: If expansion would exceed max memory size.
        """
        if length == 0:
            return 0

        end = offset + length
        if end <= len(self._data):
            return 0

        if end > self._max_size:
            raise MemoryLimitError(
                f"Memory expansion to {end} bytes exceeds limit of {self._max_size} bytes"
            )

        # Calculate new size (rounded up to 32-byte words)
        old_words = len(self._data) // self.WORD_SIZE
        new_words = (end + self.WORD_SIZE - 1) // self.WORD_SIZE
        new_size = new_words * self.WORD_SIZE

        # Expand with zeros
        self._data.extend(b"\x00" * (new_size - len(self._data)))

        return new_words - old_words

    def memory_cost(self, word_count: int) -> int:
        """
        Calculate memory gas cost for a given number of words.
        Cost = 3 * words + words^2 // 512

        Args:
            word_count: Total number of 32-byte words.

        Returns:
            Total gas cost for this amount of memory.
        """
        return (3 * word_count) + (word_count * word_count) // 512

    def expansion_cost(self, offset: int, length: int) -> int:
        """
        Calculate the gas cost for expanding memory to cover [offset, offset+length).

        Args:
            offset: Starting byte offset.
            length: Number of bytes.

        Returns:
            Incremental gas cost for this expansion (0 if no expansion needed).
        """
        if length == 0:
            return 0

        end = offset + length
        if end <= len(self._data):
            return 0

        old_words = len(self._data) // self.WORD_SIZE
        new_words = (end + self.WORD_SIZE - 1) // self.WORD_SIZE

        old_cost = self.memory_cost(old_words)
        new_cost = self.memory_cost(new_words)

        return new_cost - old_cost

    def load(self, offset: int) -> int:
        """
        Load a 32-byte word from memory at the given offset.
        Returns the word as a big-endian unsigned integer.

        Args:
            offset: Byte offset to read from.

        Returns:
            256-bit unsigned integer value.
        """
        self._expand_to(offset, self.WORD_SIZE)
        word_bytes = self._data[offset:offset + self.WORD_SIZE]
        return int.from_bytes(word_bytes, "big")

    def store(self, offset: int, value: int) -> None:
        """
        Store a 32-byte word to memory at the given offset.
        The value is stored as big-endian.

        Args:
            offset: Byte offset to write to.
            value: 256-bit unsigned integer to store.
        """
        self._expand_to(offset, self.WORD_SIZE)
        word_bytes = (value & ((1 << 256) - 1)).to_bytes(self.WORD_SIZE, "big")
        self._data[offset:offset + self.WORD_SIZE] = word_bytes

    def store8(self, offset: int, value: int) -> None:
        """
        Store a single byte to memory at the given offset.

        Args:
            offset: Byte offset to write to.
            value: Value to store (only lowest byte used).
        """
        self._expand_to(offset, 1)
        self._data[offset] = value & 0xFF

    def load_bytes(self, offset: int, length: int) -> bytes:
        """
        Load a range of bytes from memory.

        Args:
            offset: Starting byte offset.
            length: Number of bytes to read.

        Returns:
            Bytes read from memory.
        """
        if length == 0:
            return b""
        self._expand_to(offset, length)
        return bytes(self._data[offset:offset + length])

    def store_bytes(self, offset: int, data: bytes) -> None:
        """
        Store a sequence of bytes to memory starting at offset.

        Args:
            offset: Starting byte offset.
            data: Bytes to write.
        """
        if len(data) == 0:
            return
        self._expand_to(offset, len(data))
        self._data[offset:offset + len(data)] = data

    def copy(self, dst_offset: int, src_offset: int, length: int) -> None:
        """
        Copy bytes within memory from one region to another.
        Handles overlapping regions correctly.

        Args:
            dst_offset: Destination byte offset.
            src_offset: Source byte offset.
            length: Number of bytes to copy.
        """
        if length == 0:
            return
        self._expand_to(src_offset, length)
        self._expand_to(dst_offset, length)
        # Use temporary copy to handle overlaps
        temp = bytes(self._data[src_offset:src_offset + length])
        self._data[dst_offset:dst_offset + length] = temp

    def clear(self) -> None:
        """Reset memory to empty state."""
        self._data = bytearray()

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"VMMemory(size={self.size}, words={self.word_count})"
