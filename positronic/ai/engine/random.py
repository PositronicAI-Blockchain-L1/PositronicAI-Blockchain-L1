"""
Positronic - Deterministic Random Number Generator for Neural Consensus

Provides a deterministic, thread-safe random number generation system that
guarantees all validator nodes produce **identical** pseudo-random sequences
for the same seed.  This is a hard requirement for Proof-of-Neural-Consensus:
every node must arrive at the same AI validation result, which means every
source of randomness (weight initialisation, dropout masks, data shuffling)
must be reproducible and auditable.

Usage::

    from positronic.ai.engine.random import DeterministicContext, get_global_rng

    # Inside consensus-critical code paths:
    with DeterministicContext():
        weights = get_global_rng().randn(128, 64)
        # All nodes will produce the exact same `weights` array.

    # Or with a custom seed:
    with DeterministicContext(seed=12345):
        mask = get_global_rng().rand(32) > 0.5
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AI_CONSENSUS_SEED: int = 42
"""Default seed used across the network to ensure deterministic AI inference
during consensus validation rounds."""

# ---------------------------------------------------------------------------
# Thread-local state
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_state() -> threading.local:
    """Return the thread-local state, initialising defaults if needed."""
    if not hasattr(_thread_local, "global_seed"):
        _thread_local.global_seed = AI_CONSENSUS_SEED
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = np.random.RandomState(AI_CONSENSUS_SEED)
    if not hasattr(_thread_local, "saved_states"):
        _thread_local.saved_states: list = []
    return _thread_local


# ---------------------------------------------------------------------------
# Public API — seed management
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Set the global RNG seed for the current thread.

    This re-seeds the thread-local ``RandomState`` so that all subsequent
    calls to :func:`get_global_rng` (in the same thread) produce a fresh
    deterministic sequence starting from *seed*.

    Parameters
    ----------
    seed : int
        The new seed value.
    """
    state = _get_state()
    state.global_seed = seed
    state.rng = np.random.RandomState(seed)


def get_global_rng() -> np.random.RandomState:
    """Return the thread-local ``numpy.random.RandomState`` instance.

    If no seed has been set explicitly, the default
    :data:`AI_CONSENSUS_SEED` is used.

    Returns
    -------
    numpy.random.RandomState
        A seeded, deterministic random number generator.
    """
    return _get_state().rng


def get_rng(seed: int) -> np.random.RandomState:
    """Create and return a **new** ``RandomState`` seeded with *seed*.

    Unlike :func:`get_global_rng`, this does **not** affect any global or
    thread-local state.  Use this when you need an isolated RNG that will
    not interfere with other random streams.

    Parameters
    ----------
    seed : int
        Seed for the new generator.

    Returns
    -------
    numpy.random.RandomState
        A fresh, independently seeded RNG.
    """
    return np.random.RandomState(seed)


# ---------------------------------------------------------------------------
# DeterministicRNG helper class
# ---------------------------------------------------------------------------

class DeterministicRNG:
    """Convenience wrapper around a seeded ``numpy.random.RandomState``.

    This class is useful when you want an object that carries its own seed
    and can be re-seeded or forked without touching global state.

    Parameters
    ----------
    seed : int, optional
        Initial seed (defaults to :data:`AI_CONSENSUS_SEED`).
    """

    def __init__(self, seed: int = AI_CONSENSUS_SEED) -> None:
        self.seed: int = seed
        self._rng: np.random.RandomState = np.random.RandomState(seed)

    def reset(self) -> None:
        """Re-seed the internal RNG back to the original seed."""
        self._rng = np.random.RandomState(self.seed)

    def set_seed(self, seed: int) -> None:
        """Change the seed and reset the internal RNG."""
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def rng(self) -> np.random.RandomState:
        """The underlying ``numpy.random.RandomState`` instance."""
        return self._rng

    def randn(self, *shape: int) -> np.ndarray:
        """Draw samples from the standard normal distribution."""
        return self._rng.randn(*shape)

    def rand(self, *shape: int) -> np.ndarray:
        """Draw uniform samples from [0, 1)."""
        return self._rng.rand(*shape)

    def randint(self, low: int, high: int, size: Optional[int] = None) -> np.ndarray:
        """Draw random integers from [low, high)."""
        return self._rng.randint(low, high, size=size)

    def choice(self, a, size: Optional[int] = None, replace: bool = True) -> np.ndarray:
        """Random selection from an array or range."""
        return self._rng.choice(a, size=size, replace=replace)

    def shuffle(self, arr: np.ndarray) -> None:
        """Shuffle *arr* in-place deterministically."""
        self._rng.shuffle(arr)

    def fork(self, child_seed: Optional[int] = None) -> "DeterministicRNG":
        """Create a child RNG derived from this one.

        If *child_seed* is ``None``, a seed is drawn from the current RNG
        so the child is deterministic relative to the parent sequence.

        Parameters
        ----------
        child_seed : int, optional
            Explicit seed for the child.  If ``None``, one is generated.

        Returns
        -------
        DeterministicRNG
            A new, independent RNG.
        """
        if child_seed is None:
            child_seed = int(self._rng.randint(0, 2**31))
        return DeterministicRNG(seed=child_seed)

    def __repr__(self) -> str:
        return f"DeterministicRNG(seed={self.seed})"


# ---------------------------------------------------------------------------
# DeterministicContext — context manager
# ---------------------------------------------------------------------------

class DeterministicContext:
    """Context manager that enforces deterministic random state.

    On entry, the current NumPy global random state **and** the thread-local
    Positronic RNG state are saved, then replaced with a fresh state seeded from
    *seed*.  On exit, the previous states are restored so that code outside
    the block is not affected.

    This is the recommended way to wrap any consensus-critical computation
    so that non-determinism cannot leak in.

    Parameters
    ----------
    seed : int, optional
        Seed to use inside the block.  Defaults to :data:`AI_CONSENSUS_SEED`.

    Example
    -------
    ::

        with DeterministicContext(seed=99):
            x = np.random.randn(4)  # deterministic
            y = get_global_rng().randn(4)  # also deterministic
        # Outside the block, original state is restored.
    """

    def __init__(self, seed: int = AI_CONSENSUS_SEED) -> None:
        self.seed: int = seed
        self._saved_numpy_state: Optional[dict] = None
        self._saved_thread_rng: Optional[np.random.RandomState] = None
        self._saved_thread_seed: Optional[int] = None

    def __enter__(self) -> "DeterministicContext":
        # Save NumPy global random state.
        self._saved_numpy_state = np.random.get_state()

        # Save thread-local Positronic RNG state.
        tls = _get_state()
        self._saved_thread_rng = tls.rng
        self._saved_thread_seed = tls.global_seed

        # Apply deterministic seed to both.
        np.random.seed(self.seed)
        set_global_seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore NumPy global random state.
        if self._saved_numpy_state is not None:
            np.random.set_state(self._saved_numpy_state)

        # Restore thread-local Positronic RNG state.
        tls = _get_state()
        if self._saved_thread_rng is not None:
            tls.rng = self._saved_thread_rng
        if self._saved_thread_seed is not None:
            tls.global_seed = self._saved_thread_seed

        return None  # Do not suppress exceptions.
