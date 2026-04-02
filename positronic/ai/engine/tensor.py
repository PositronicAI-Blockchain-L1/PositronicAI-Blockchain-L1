"""
Positronic - Autograd Tensor Engine

The foundational automatic differentiation engine for the Positronic neural
consensus system. Provides a Tensor class that wraps numpy.ndarray and tracks
computational graphs for reverse-mode automatic differentiation (backprop).

Every arithmetic operation creates a new Tensor node in the graph and attaches
a ``_backward`` closure that knows how to propagate gradients to its parents.
Calling ``backward()`` on a scalar loss performs a topological sort of the
graph and walks it in reverse, accumulating gradients via the chain rule.

This module intentionally has **zero** dependency on PyTorch, TensorFlow, or
any other deep learning framework. It is built purely on NumPy so that every
validator node can reproduce identical computations without GPU drivers or
heavy runtime dependencies.

Design goals:
    - Correctness: gradient numerics verified against finite-difference checks.
    - Determinism: identical inputs always produce identical outputs (critical
      for Proof-of-Neural-Consensus).
    - Clarity: the code is the documentation; every backward closure is
      self-contained and readable.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Set, Callable, Union, Sequence

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, float, int, list]
Shape = Tuple[int, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_array(data: ArrayLike) -> np.ndarray:
    """Convert *data* to a numpy float64 array if it is not already one."""
    if isinstance(data, np.ndarray):
        if data.dtype != np.float64:
            return data.astype(np.float64)
        return data
    return np.array(data, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, shape: Shape) -> np.ndarray:
    """Sum *grad* over dimensions that were broadcast to match *shape*.

    When two tensors with different shapes are combined via a NumPy
    broadcasting operation, the output gradient has the broadcasted shape.
    To propagate back to the original (smaller) operand we must sum out any
    axes that were added or expanded.

    Parameters
    ----------
    grad : np.ndarray
        The gradient with the (larger) broadcasted shape.
    shape : tuple of int
        The target shape that the gradient must be reduced to.

    Returns
    -------
    np.ndarray
        Gradient reduced to *shape*.
    """
    if grad.shape == shape:
        return grad

    # Number of leading dimensions that were prepended during broadcast.
    ndim_added = grad.ndim - len(shape)

    # Sum over all prepended dimensions first.
    for _ in range(ndim_added):
        grad = grad.sum(axis=0)

    # Now handle dimensions that were size-1 and got expanded.
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """A multidimensional array with automatic differentiation support.

    Parameters
    ----------
    data : ArrayLike
        The numeric payload (scalar, list, or ndarray).
    requires_grad : bool, optional
        Whether this tensor should track gradients (default ``False``).
    _children : tuple of Tensor, optional
        Parent tensors in the computational graph (internal use only).

    Attributes
    ----------
    data : np.ndarray
        The underlying NumPy array (float64).
    grad : np.ndarray
        Accumulated gradient, same shape as *data*. Initialised to zeros.
    requires_grad : bool
        Whether this node participates in gradient computation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
    ) -> None:
        self.data: np.ndarray = _ensure_array(data)
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)

        # Computational-graph bookkeeping.
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set(_children)

        # Propagate ``requires_grad`` from children: if any parent requires
        # grad, so does the output.
        if not requires_grad and any(c.requires_grad for c in _children):
            self.requires_grad = True

    # ------------------------------------------------------------------
    # Static factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(*shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor filled with zeros."""
        return Tensor(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)

    @staticmethod
    def ones(*shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor filled with ones."""
        return Tensor(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)

    @staticmethod
    def randn(*shape: int, requires_grad: bool = False) -> "Tensor":
        """Return a tensor filled with samples from the standard normal."""
        return Tensor(np.random.randn(*shape).astype(np.float64), requires_grad=requires_grad)

    @staticmethod
    def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Wrap an existing NumPy array in a Tensor."""
        return Tensor(arr, requires_grad=requires_grad)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Shape:
        """Shape of the underlying data array."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        """Data type of the underlying array."""
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        """Shorthand for ``self.transpose()``."""
        return self.transpose()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array (detached from the graph)."""
        return self.data

    def detach(self) -> "Tensor":
        """Return a new Tensor that shares the same data but is detached
        from the computational graph (no gradient tracking)."""
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self) -> None:
        """Reset the gradient to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def item(self) -> float:
        """Return the scalar value for a single-element tensor."""
        return float(self.data.item())

    # ------------------------------------------------------------------
    # Backward pass (reverse-mode autodiff)
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Compute gradients via reverse-mode automatic differentiation.

        Must be called on a scalar (0-d or single-element) tensor, typically
        the loss value.  Performs a topological sort of the computational
        graph and then walks it in reverse, calling each node's ``_backward``
        closure to accumulate gradients.
        """
        if self.data.size != 1:
            raise RuntimeError(
                "backward() can only be called on a scalar tensor, "
                f"but got shape {self.shape}"
            )

        # Topological ordering via depth-first search.
        topo: list["Tensor"] = []
        visited: Set[int] = set()

        def _build_topo(node: "Tensor") -> None:
            node_id = id(node)
            if node_id not in visited:
                visited.add(node_id)
                for parent in node._prev:
                    _build_topo(parent)
                topo.append(node)

        _build_topo(self)

        # Seed the gradient of the loss itself.
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # Walk in reverse topological order.
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Element-wise addition with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Right-hand addition (supports ``scalar + Tensor``)."""
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Element-wise subtraction."""
        return self + (-other if isinstance(other, Tensor) else Tensor(-np.asarray(other, dtype=np.float64)))

    def __rsub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Right-hand subtraction (``scalar - Tensor``)."""
        return (-self) + other

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Element-wise multiplication with broadcasting support."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad * other.data, self.shape)
            if other.requires_grad:
                other.grad = other.grad + _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Right-hand multiplication (``scalar * Tensor``)."""
        return self.__mul__(other)

    def __neg__(self) -> "Tensor":
        """Unary negation."""
        return self * (-1.0)

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Element-wise division: ``self / other`` implemented as
        ``self * other ** -1``."""
        return self * (other ** (-1.0))

    def __rtruediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Right-hand division (``scalar / Tensor``)."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self

    def __pow__(self, exponent: Union[float, int]) -> "Tensor":
        """Raise every element to a scalar *exponent*.

        Only scalar (non-Tensor) exponents are supported because the gradient
        w.r.t. the exponent (``x^n * ln(x)``) is rarely needed in typical
        neural-network workloads.
        """
        assert isinstance(exponent, (int, float)), (
            "Tensor.__pow__ only supports scalar exponents"
        )
        out = Tensor(self.data ** exponent, _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                local_grad = exponent * (self.data ** (exponent - 1))
                self.grad = self.grad + _unbroadcast(out.grad * local_grad, self.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication (``self @ other``).

        Supports 2-D @ 2-D and batched (N-D @ N-D) via ``np.matmul``.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), _children=(self, other))

        def _backward() -> None:
            if self.requires_grad:
                # grad_self = out.grad @ other^T
                g = np.matmul(out.grad, _swap_last_two(other.data))
                self.grad = self.grad + _unbroadcast(g, self.shape)
            if other.requires_grad:
                # grad_other = self^T @ out.grad
                g = np.matmul(_swap_last_two(self.data), out.grad)
                other.grad = other.grad + _unbroadcast(g, other.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        """Right-hand matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__matmul__(self)

    # ------------------------------------------------------------------
    # Reduction operations
    # ------------------------------------------------------------------

    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Sum elements, optionally along *axis*.

        Parameters
        ----------
        axis : int or tuple of int, optional
            Axis or axes to sum over.  ``None`` sums all elements.
        keepdims : bool
            If ``True``, reduced axes are kept as size-1 dimensions.
        """
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                grad = out.grad
                # Expand the gradient back to the original shape.
                if axis is not None and not keepdims:
                    # Re-insert the summed-over axes so broadcasting works.
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(a % self.ndim for a in axes):
                        grad = np.expand_dims(grad, axis=ax)
                # Broadcast to full shape.
                self.grad = self.grad + np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        """Compute the arithmetic mean, optionally along *axis*.

        Implemented as ``sum / n`` so that gradients flow correctly.
        """
        if axis is None:
            n = float(self.data.size)
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            n = float(np.prod([self.data.shape[a] for a in axes]))

        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, *shape: int) -> "Tensor":
        """Return a tensor with a new shape (gradient-aware).

        Parameters
        ----------
        *shape : int
            The desired shape.  Exactly one dimension may be ``-1``.
        """
        # Accept both reshape(2, 3) and reshape((2, 3)).
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        out = Tensor(self.data.reshape(shape), _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        """Transpose dimensions.

        With no arguments, reverses the order of all axes (standard matrix
        transpose).  Otherwise, permutes axes according to *axes*.
        """
        if not axes:
            perm = None  # numpy default: reverse all axes
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = axes

        out = Tensor(self.data.transpose(perm), _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                if perm is None:
                    inv_perm = None
                else:
                    # Invert the permutation.
                    inv_perm = [0] * len(perm)
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                    inv_perm = tuple(inv_perm)
                self.grad = self.grad + out.grad.transpose(inv_perm)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation helpers (commonly needed in neural networks)
    # ------------------------------------------------------------------

    def relu(self) -> "Tensor":
        """Rectified Linear Unit: ``max(0, x)``."""
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * (self.data > 0).astype(np.float64)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        clipped = np.clip(self.data, -500, 500)
        out_data = np.exp(clipped)
        out = Tensor(out_data, _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * out_data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Element-wise natural logarithm."""
        safe_data = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(safe_data), _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad / safe_data

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """Element-wise hyperbolic tangent."""
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        """Element-wise logistic sigmoid: ``1 / (1 + exp(-x))``."""
        clipped = np.clip(self.data, -500, 500)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(s, _children=(self,))

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * s * (1.0 - s)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Comparison (no gradient — used for masks / assertions)
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> np.ndarray:  # type: ignore[override]
        if isinstance(other, Tensor):
            return self.data == other.data
        return self.data == np.asarray(other)

    def __lt__(self, other: Union["Tensor", ArrayLike]) -> np.ndarray:
        if isinstance(other, Tensor):
            return self.data < other.data
        return self.data < np.asarray(other)

    def __gt__(self, other: Union["Tensor", ArrayLike]) -> np.ndarray:
        if isinstance(other, Tensor):
            return self.data > other.data
        return self.data > np.asarray(other)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_str})"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, idx):
        """Basic indexing / slicing (forward-only, no backward)."""
        return Tensor(self.data[idx])


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _swap_last_two(arr: np.ndarray) -> np.ndarray:
    """Swap the last two axes of *arr* (generalised matrix transpose).

    Works for 2-D arrays (plain transpose) as well as batched N-D arrays
    used in batched matmul.
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return arr.transpose(axes)
