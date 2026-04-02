"""
Positronic - Priority Replay Buffer

Stores training examples with priority weighting for importance sampling.
This is a core data structure for the online learning pipeline, enabling
the system to balance rare fraud examples against common normal transactions.

In fraud detection, confirmed malicious transactions are extremely rare
compared to normal ones. Without priority sampling, these critical examples
would be underrepresented in training batches. The priority replay buffer
solves this by assigning higher priority to rare/important examples,
making them more likely to be selected during training.

The priority exponent ``alpha`` controls the degree of prioritization:
    - alpha = 0: Uniform random sampling (all priorities equal).
    - alpha = 1: Full priority sampling (probability proportional to priority).
    - alpha = 0.6 (default): Balanced compromise between uniform and
      full priority sampling.

Dependencies:
    - numpy for numerical operations
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import deque


class PriorityReplayBuffer:
    """Priority replay buffer for experience replay with importance sampling.

    The buffer maintains a fixed-capacity circular array of training examples,
    each associated with a priority value. When sampling, examples are drawn
    with probability proportional to ``priority^alpha``.

    When the buffer reaches capacity, new items overwrite the oldest entries
    in FIFO order (circular buffer behavior).

    Args:
        capacity: Maximum number of items the buffer can hold. Once full,
            new items overwrite the oldest entries.
        alpha: Priority exponent controlling the degree of prioritization.
            0 = uniform sampling, 1 = fully priority-weighted sampling.

    Example::

        from positronic.ai.training.data_buffer import PriorityReplayBuffer

        buffer = PriorityReplayBuffer(capacity=5000, alpha=0.6)

        # Add normal transactions with default priority
        for tx_features in normal_transactions:
            buffer.add(tx_features, priority=1.0)

        # Add rare fraud examples with high priority
        for tx_features in fraud_transactions:
            buffer.add(tx_features, priority=5.0)

        # Sample a batch (fraud examples are overrepresented)
        batch = buffer.sample(batch_size=64)
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha

        self._data: List[np.ndarray] = []
        self._priorities: List[float] = []
        self._position: int = 0
        self._full: bool = False

    @property
    def size(self) -> int:
        """Current number of items in the buffer.

        Returns:
            Number of stored items (always <= capacity).
        """
        return len(self._data)

    def add(self, item: np.ndarray, priority: float = 1.0) -> None:
        """Add an item to the buffer with the given priority.

        Items are copied to prevent external mutation. When the buffer is
        full, the oldest item is overwritten.

        Args:
            item: A numpy array representing a training example (typically
                a 1-D feature vector).
            priority: Sampling priority for this item. Higher priority
                means the item is more likely to be selected during
                ``sample()``. Must be positive.
        """
        if len(self._data) < self.capacity:
            self._data.append(item.copy())
            self._priorities.append(priority)
        else:
            self._data[self._position] = item.copy()
            self._priorities[self._position] = priority

        self._position = (self._position + 1) % self.capacity
        if self._position == 0 and len(self._data) >= self.capacity:
            self._full = True

    def sample(self, batch_size: int) -> Optional[List[np.ndarray]]:
        """Sample a batch with priority-weighted probabilities.

        Sampling probability for item i is proportional to
        ``priority_i^alpha``. Items are sampled without replacement.

        If the buffer contains fewer items than ``batch_size``, all
        available items are returned. If the buffer is empty, returns None.

        Args:
            batch_size: Desired number of items to sample.

        Returns:
            List of numpy arrays (the sampled items), or None if the
            buffer is empty.
        """
        if self.size == 0:
            return None

        if self.size < batch_size:
            batch_size = self.size

        priorities = np.array(self._priorities[: self.size], dtype=np.float64)
        priorities = priorities ** self.alpha
        probs = priorities / (priorities.sum() + 1e-8)

        indices = np.random.choice(
            self.size, size=batch_size, replace=False, p=probs
        )
        return [self._data[i] for i in indices]

    def sample_with_indices(
        self, batch_size: int
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """Sample a batch and return both items and their indices.

        This is useful when priorities need to be updated after training
        (e.g., based on the loss each sample produced). The returned
        indices can be passed to ``update_priorities()``.

        Args:
            batch_size: Desired number of items to sample.

        Returns:
            A tuple of (items, indices):
                - items: List of sampled numpy arrays, or None if empty.
                - indices: 1-D numpy array of buffer indices, or None if empty.
        """
        if self.size == 0:
            return None, None

        if self.size < batch_size:
            batch_size = self.size

        priorities = np.array(self._priorities[: self.size], dtype=np.float64)
        priorities = priorities ** self.alpha
        probs = priorities / (priorities.sum() + 1e-8)

        indices = np.random.choice(
            self.size, size=batch_size, replace=False, p=probs
        )
        items = [self._data[i] for i in indices]
        return items, indices

    def update_priorities(
        self, indices: np.ndarray, new_priorities: np.ndarray
    ) -> None:
        """Update priorities for items at the given indices.

        Typically called after training to adjust priorities based on
        per-sample loss. Samples with higher loss (harder examples)
        receive higher priority for future sampling.

        Args:
            indices: 1-D array of buffer indices (from ``sample_with_indices``).
            new_priorities: 1-D array of new priority values, one per index.
        """
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < self.size:
                self._priorities[idx] = float(priority)

    def clear(self) -> None:
        """Clear all items and priorities from the buffer.

        Resets the buffer to its initial empty state.
        """
        self._data.clear()
        self._priorities.clear()
        self._position = 0
        self._full = False

    def get_stats(self) -> Dict:
        """Return a summary dictionary of the buffer state.

        Returns:
            Dictionary with keys: size, capacity, is_full, avg_priority,
            max_priority.
        """
        return {
            "size": self.size,
            "capacity": self.capacity,
            "is_full": self._full,
            "avg_priority": (
                float(np.mean(self._priorities[: self.size]))
                if self.size > 0
                else 0.0
            ),
            "max_priority": (
                float(max(self._priorities[: self.size]))
                if self.size > 0
                else 0.0
            ),
        }
