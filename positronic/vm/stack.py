"""
Positronic - PositronicVM Stack

Stack-based data structure for the virtual machine.
Maximum depth of 1024 items, each item a 256-bit (32-byte) unsigned integer.
"""

from typing import List

from positronic.constants import MAX_STACK_DEPTH


class StackOverflowError(Exception):
    """Raised when pushing to a full stack."""
    pass


class StackUnderflowError(Exception):
    """Raised when popping from an empty stack."""
    pass


class VMStack:
    """
    PositronicVM execution stack.

    Stores unsigned 256-bit integers (Python ints, masked to 2^256 - 1).
    Maximum depth is 1024 as defined in positronic.constants.MAX_STACK_DEPTH.
    """

    UINT256_MAX = (1 << 256) - 1

    def __init__(self, max_depth: int = MAX_STACK_DEPTH):
        self._items: List[int] = []
        self._max_depth = max_depth

    @property
    def depth(self) -> int:
        """Current number of items on the stack."""
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        """Check if stack has no items."""
        return len(self._items) == 0

    @property
    def is_full(self) -> bool:
        """Check if stack is at maximum depth."""
        return len(self._items) >= self._max_depth

    def _clamp(self, value: int) -> int:
        """Clamp a value to the uint256 range [0, 2^256 - 1]."""
        return value & self.UINT256_MAX

    def push(self, value: int) -> None:
        """
        Push a value onto the stack.

        Args:
            value: Integer value to push. Will be clamped to uint256.

        Raises:
            StackOverflowError: If stack depth would exceed 1024.
        """
        if len(self._items) >= self._max_depth:
            raise StackOverflowError(
                f"Stack overflow: depth {len(self._items)} >= max {self._max_depth}"
            )
        self._items.append(self._clamp(value))

    def pop(self) -> int:
        """
        Pop the top value from the stack.

        Returns:
            The top stack value as an unsigned integer.

        Raises:
            StackUnderflowError: If stack is empty.
        """
        if not self._items:
            raise StackUnderflowError("Stack underflow: cannot pop from empty stack")
        return self._items.pop()

    def pop_n(self, n: int) -> List[int]:
        """
        Pop n values from the stack.

        Args:
            n: Number of values to pop.

        Returns:
            List of n values, first element is the topmost popped value.

        Raises:
            StackUnderflowError: If stack has fewer than n items.
        """
        if len(self._items) < n:
            raise StackUnderflowError(
                f"Stack underflow: need {n} items, have {len(self._items)}"
            )
        result = []
        for _ in range(n):
            result.append(self._items.pop())
        return result

    def peek(self, depth: int = 0) -> int:
        """
        Peek at a stack item without removing it.

        Args:
            depth: How deep to look. 0 = top of stack, 1 = second from top, etc.

        Returns:
            The value at the given depth.

        Raises:
            StackUnderflowError: If depth exceeds stack size.
        """
        if depth >= len(self._items):
            raise StackUnderflowError(
                f"Stack underflow: peek depth {depth} >= stack size {len(self._items)}"
            )
        return self._items[-(depth + 1)]

    def dup(self, depth: int) -> None:
        """
        Duplicate a stack item at the given depth and push it on top.
        DUP1 = dup(1) duplicates the top item.
        DUP2 = dup(2) duplicates the second item.

        Args:
            depth: 1-indexed depth of the item to duplicate.
                   1 = top of stack, 2 = second from top, etc.

        Raises:
            StackUnderflowError: If depth exceeds stack size.
            StackOverflowError: If stack is full.
        """
        if depth < 1 or depth > len(self._items):
            raise StackUnderflowError(
                f"Stack underflow: DUP{depth} requires {depth} items, "
                f"have {len(self._items)}"
            )
        if len(self._items) >= self._max_depth:
            raise StackOverflowError(
                f"Stack overflow: cannot DUP, depth {len(self._items)} >= max {self._max_depth}"
            )
        value = self._items[-depth]
        self._items.append(value)

    def swap(self, depth: int) -> None:
        """
        Swap the top stack item with the item at the given depth.
        SWAP1 = swap(1) swaps top with second item.
        SWAP2 = swap(2) swaps top with third item.

        Args:
            depth: 1-indexed depth of the item to swap with the top.
                   1 = swap with second item, 2 = swap with third, etc.

        Raises:
            StackUnderflowError: If depth + 1 exceeds stack size.
        """
        if depth < 1 or depth + 1 > len(self._items):
            raise StackUnderflowError(
                f"Stack underflow: SWAP{depth} requires {depth + 1} items, "
                f"have {len(self._items)}"
            )
        # Swap top with the item at -(depth + 1)
        top_idx = len(self._items) - 1
        swap_idx = top_idx - depth
        self._items[top_idx], self._items[swap_idx] = (
            self._items[swap_idx],
            self._items[top_idx],
        )

    def clear(self) -> None:
        """Remove all items from the stack."""
        self._items.clear()

    def to_list(self) -> List[int]:
        """
        Return a copy of the stack as a list.
        Index 0 is the bottom of the stack, last index is the top.
        """
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        top_items = self._items[-5:] if len(self._items) > 5 else self._items
        top_hex = [f"0x{v:04x}" for v in reversed(top_items)]
        return f"VMStack(depth={self.depth}, top={top_hex})"
