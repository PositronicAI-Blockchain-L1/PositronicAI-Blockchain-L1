"""
Positronic - Neural Engine Model Base Class

Base class for all neural models in the Positronic AI engine.
Provides parameter discovery, save/load, training/eval modes,
gradient management, and model introspection utilities.

Dependencies:
    - positronic.ai.engine.tensor.Tensor: Core tensor class with autograd support.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Iterator
from positronic.ai.engine.tensor import Tensor


class Model:
    """
    Base class for all neural models.

    Subclasses should:
        1. Define layers as attributes in ``__init__`` (call ``super().__init__()``).
        2. Implement :meth:`forward` for the computation graph.
        3. Optionally override :meth:`train_step` for custom training logic.

    Example::

        class MyModel(Model):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.weight = Tensor(np.random.randn(in_dim, out_dim) * 0.01,
                                     requires_grad=True)
                self.bias = Tensor(np.zeros(out_dim), requires_grad=True)

            def forward(self, x):
                return x @ self.weight + self.bias
    """

    def __init__(self) -> None:
        self._training: bool = True
        self._name: str = self.__class__.__name__
        self._version: int = 1

    # ------------------------------------------------------------------
    # Forward / call
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Define the forward computation.

        Must be overridden by every concrete subclass.

        Returns:
            Tensor produced by the forward pass.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{self._name}.forward() is not implemented. "
            "Subclasses must override forward()."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """Make the model callable, delegating to :meth:`forward`."""
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Parameter discovery
    # ------------------------------------------------------------------

    def parameters(self) -> List[Tensor]:
        """
        Auto-discover all trainable :class:`Tensor` parameters.

        Walks instance attributes and recursively collects parameters from
        sub-layers that expose a ``parameters()`` method.  Handles plain
        attributes, lists, tuples, and dicts of layers.

        Returns:
            List of unique :class:`Tensor` objects with ``requires_grad=True``.
        """
        params: List[Tensor] = []
        visited: set = set()

        def _collect(obj: Any) -> None:
            """Recursively collect parameters from an object."""
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, Tensor) and obj.requires_grad:
                params.append(obj)
            elif hasattr(obj, "parameters") and callable(obj.parameters):
                for p in obj.parameters():
                    if id(p) not in visited:
                        visited.add(id(p))
                        params.append(p)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _collect(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)

        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue
            _collect(attr)

        return params

    def named_parameters(self) -> List[Tuple[str, Tensor]]:
        """
        Return ``(name, tensor)`` pairs for every trainable parameter.

        The name is a dot-separated path reflecting the attribute hierarchy
        (e.g. ``"encoder.weight"``).

        Returns:
            List of (name, Tensor) tuples.
        """
        named: List[Tuple[str, Tensor]] = []
        visited: set = set()

        def _collect(prefix: str, obj: Any) -> None:
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, Tensor) and obj.requires_grad:
                named.append((prefix, obj))
            elif hasattr(obj, "named_parameters") and callable(obj.named_parameters):
                for sub_name, p in obj.named_parameters():
                    full = f"{prefix}.{sub_name}" if prefix else sub_name
                    if id(p) not in visited:
                        visited.add(id(p))
                        named.append((full, p))
            elif hasattr(obj, "parameters") and callable(obj.parameters):
                for i, p in enumerate(obj.parameters()):
                    full = f"{prefix}.param_{i}" if prefix else f"param_{i}"
                    if id(p) not in visited:
                        visited.add(id(p))
                        named.append((full, p))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _collect(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    _collect(f"{prefix}.{i}" if prefix else str(i), item)

        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue
            _collect(name, attr)

        return named

    def num_parameters(self) -> int:
        """
        Count the total number of trainable scalar parameters.

        Returns:
            Integer count of all scalar values across every parameter tensor.
        """
        return sum(p.data.size for p in self.parameters())

    # ------------------------------------------------------------------
    # Training / evaluation mode
    # ------------------------------------------------------------------

    def train(self) -> "Model":
        """
        Set the model (and all sub-layers) to training mode.

        In training mode stochastic layers like dropout are active.

        Returns:
            ``self``, for method chaining.
        """
        self._training = True
        self._set_training_recursive(True)
        return self

    def eval(self) -> "Model":
        """
        Set the model (and all sub-layers) to evaluation mode.

        Disables dropout and similar stochastic behaviours.

        Returns:
            ``self``, for method chaining.
        """
        self._training = False
        self._set_training_recursive(False)
        return self

    def _set_training_recursive(self, mode: bool) -> None:
        """Propagate training mode to all child layers."""
        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue
            self._set_training_on(attr, mode)

    @staticmethod
    def _set_training_on(obj: Any, mode: bool) -> None:
        """Set training flag on a single object and its children."""
        if hasattr(obj, "training"):
            obj.training = mode
        if hasattr(obj, "_set_training_recursive"):
            obj._set_training_recursive(mode)
        if isinstance(obj, (list, tuple)):
            for item in obj:
                if hasattr(item, "training"):
                    item.training = mode
                if hasattr(item, "_set_training_recursive"):
                    item._set_training_recursive(mode)
        if isinstance(obj, dict):
            for v in obj.values():
                if hasattr(v, "training"):
                    v.training = mode
                if hasattr(v, "_set_training_recursive"):
                    v._set_training_recursive(mode)

    @property
    def training(self) -> bool:
        """Whether the model is in training mode."""
        return self._training

    @training.setter
    def training(self, val: bool) -> None:
        self._training = val

    # ------------------------------------------------------------------
    # Gradient utilities
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """
        Reset gradients of all parameters to zero.

        Should be called before each training step to prevent gradient
        accumulation across mini-batches.
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad = np.zeros_like(p.grad)

    def clip_grad_norm(self, max_norm: float, eps: float = 1e-6) -> float:
        """
        Clip gradients by global L2 norm.

        If the combined L2 norm of all parameter gradients exceeds
        *max_norm*, every gradient is rescaled so the norm equals
        *max_norm*.

        Args:
            max_norm: Maximum allowed gradient norm.
            eps: Small constant for numerical stability.

        Returns:
            The original (unclipped) total gradient norm.
        """
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}")

        total_norm_sq = 0.0
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                total_norm_sq += float(np.sum(p.grad ** 2))
                grads.append(p)
        total_norm = float(np.sqrt(total_norm_sq))

        clip_coef = max_norm / (total_norm + eps)
        if clip_coef < 1.0:
            for p in grads:
                p.grad = p.grad * clip_coef

        return total_norm

    def freeze(self) -> None:
        """Freeze all parameters (disable gradient computation)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (enable gradient computation)."""
        for p in self.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Export model state as a flat dictionary of NumPy arrays.

        Keys follow dot-separated naming (e.g. ``"encoder.weight"``).

        Returns:
            Dictionary mapping parameter names to copies of their data.
        """
        state: Dict[str, np.ndarray] = {}

        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(attr, Tensor) and attr.requires_grad:
                state[name] = attr.data.copy()
            elif hasattr(attr, "state_dict") and callable(attr.state_dict):
                sub_state = attr.state_dict()
                for k, v in sub_state.items():
                    state[f"{name}.{k}"] = v
            elif hasattr(attr, "parameters") and callable(attr.parameters):
                for i, p in enumerate(attr.parameters()):
                    state[f"{name}.param_{i}"] = p.data.copy()
            elif isinstance(attr, (list, tuple)):
                for idx, item in enumerate(attr):
                    if hasattr(item, "state_dict") and callable(item.state_dict):
                        for k, v in item.state_dict().items():
                            state[f"{name}.{idx}.{k}"] = v
                    elif hasattr(item, "parameters") and callable(item.parameters):
                        for i, p in enumerate(item.parameters()):
                            state[f"{name}.{idx}.param_{i}"] = p.data.copy()

        return state

    def load_state_dict(
        self,
        state: Dict[str, np.ndarray],
        strict: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """
        Load model state from a flat dictionary.

        Args:
            state: Dictionary mapping parameter names to NumPy arrays.
            strict: If ``True``, raise on mismatched or missing keys.

        Returns:
            Tuple of (missing_keys, unexpected_keys).

        Raises:
            RuntimeError: When *strict* is ``True`` and keys do not match.
        """
        own_state = self.state_dict()
        missing = [k for k in own_state if k not in state]
        unexpected = [k for k in state if k not in own_state]

        if strict and (missing or unexpected):
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing keys: {missing}")
            if unexpected:
                msg_parts.append(f"Unexpected keys: {unexpected}")
            raise RuntimeError(
                "Error loading state_dict. " + "; ".join(msg_parts)
            )

        # Apply matching parameters
        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(attr, Tensor) and attr.requires_grad and name in state:
                if attr.data.shape != state[name].shape:
                    raise ValueError(
                        f"Shape mismatch for '{name}': "
                        f"expected {attr.data.shape}, got {state[name].shape}"
                    )
                attr.data = state[name].copy()
            elif hasattr(attr, "load_state_dict") and callable(attr.load_state_dict):
                prefix = f"{name}."
                sub_state = {
                    k[len(prefix):]: v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
                if sub_state:
                    attr.load_state_dict(sub_state)
            elif hasattr(attr, "parameters") and callable(attr.parameters):
                for i, p in enumerate(attr.parameters()):
                    key = f"{name}.param_{i}"
                    if key in state:
                        if p.data.shape != state[key].shape:
                            raise ValueError(
                                f"Shape mismatch for '{key}': "
                                f"expected {p.data.shape}, got {state[key].shape}"
                            )
                        p.data = state[key].copy()
            elif isinstance(attr, (list, tuple)):
                for idx, item in enumerate(attr):
                    if hasattr(item, "load_state_dict") and callable(item.load_state_dict):
                        prefix = f"{name}.{idx}."
                        sub_state = {
                            k[len(prefix):]: v
                            for k, v in state.items()
                            if k.startswith(prefix)
                        }
                        if sub_state:
                            item.load_state_dict(sub_state)
                    elif hasattr(item, "parameters") and callable(item.parameters):
                        for i, p in enumerate(item.parameters()):
                            key = f"{name}.{idx}.param_{i}"
                            if key in state:
                                if p.data.shape != state[key].shape:
                                    raise ValueError(
                                        f"Shape mismatch for '{key}': "
                                        f"expected {p.data.shape}, got {state[key].shape}"
                                    )
                                p.data = state[key].copy()

        return missing, unexpected

    # ------------------------------------------------------------------
    # High-level save / load
    # ------------------------------------------------------------------

    def save(self) -> dict:
        """
        Serialize the model to a plain dictionary.

        The dictionary includes architecture metadata, version, and all
        trainable parameter data.  Suitable for passing to
        :func:`positronic.ai.engine.serialization.serialize_state` for binary
        encoding.

        Returns:
            Dictionary with keys ``model_type``, ``version``, ``state``,
            and ``num_parameters``.
        """
        return {
            "model_type": self._name,
            "version": self._version,
            "state": self.state_dict(),
            "num_parameters": self.num_parameters(),
        }

    @classmethod
    def load(cls, data: dict) -> "Model":
        """
        Reconstruct a model from a dictionary produced by :meth:`save`.

        Subclasses **must** override this method to instantiate the correct
        architecture before loading weights.

        Raises:
            NotImplementedError: If not overridden by a subclass.
        """
        raise NotImplementedError(
            f"{cls.__name__}.load() is not implemented. "
            "Subclasses must override load() to reconstruct architecture."
        )

    # ------------------------------------------------------------------
    # Training step (default implementation)
    # ------------------------------------------------------------------

    def train_step(
        self,
        loss_fn: Any,
        optimizer: Any,
        *inputs: Any,
        **kwargs: Any,
    ) -> float:
        """
        Execute one training step: forward, backward, update.

        This is a convenience wrapper.  Override for custom training logic
        (e.g. gradient accumulation, mixed precision).

        Args:
            loss_fn: Callable that accepts model output and returns a
                scalar :class:`Tensor` loss.
            optimizer: Object with a ``step()`` method that updates
                parameters using their gradients.
            *inputs: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments forwarded to :meth:`forward`.

        Returns:
            The scalar loss value as a Python float.
        """
        self.zero_grad()
        output = self.forward(*inputs, **kwargs)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
        return float(loss.data)

    # ------------------------------------------------------------------
    # Introspection / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Generate a human-readable model summary.

        Includes model name, version, parameter count, training mode,
        and per-layer breakdowns.

        Returns:
            Multi-line summary string.
        """
        sep = "-" * 60
        lines = [
            sep,
            f"Model: {self._name}  (v{self._version})",
            sep,
            f"Training mode : {self._training}",
            f"Total params  : {self.num_parameters():,}",
            sep,
        ]

        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue

            if isinstance(attr, Tensor) and attr.requires_grad:
                lines.append(
                    f"  {name:<24s}  Tensor  shape={attr.data.shape}  "
                    f"({attr.data.size:,} params)"
                )
            elif hasattr(attr, "parameters") and callable(attr.parameters):
                n_params = sum(p.data.size for p in attr.parameters())
                lines.append(
                    f"  {name:<24s}  {type(attr).__name__:<16s}  "
                    f"({n_params:,} params)"
                )
            elif isinstance(attr, (list, tuple)):
                for idx, item in enumerate(attr):
                    if hasattr(item, "parameters") and callable(item.parameters):
                        n_params = sum(p.data.size for p in item.parameters())
                        lines.append(
                            f"  {name}[{idx}]"
                            f"{'':>{20 - len(name) - len(str(idx))}s}"
                            f"  {type(item).__name__:<16s}  "
                            f"({n_params:,} params)"
                        )

        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self._name}("
            f"params={self.num_parameters():,}, "
            f"training={self._training})"
        )

    def __str__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------------
    # Copy utilities
    # ------------------------------------------------------------------

    def copy_weights_from(self, other: "Model") -> None:
        """
        Copy weights from *other* model into this model.

        Both models must share the same architecture (matching state_dict
        keys and shapes).

        Args:
            other: Source model to copy weights from.

        Raises:
            ValueError: On shape mismatches between corresponding params.
        """
        self.load_state_dict(other.state_dict(), strict=True)

    def apply(self, fn: Any) -> "Model":
        """
        Apply a function to every sub-layer and to ``self``.

        Useful for custom weight initialization schemes.

        Args:
            fn: Callable that takes a single argument (a layer or model).

        Returns:
            ``self``, for method chaining.
        """
        for name, attr in self.__dict__.items():
            if name.startswith("_"):
                continue
            if hasattr(attr, "apply") and callable(attr.apply):
                attr.apply(fn)
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if hasattr(item, "apply") and callable(item.apply):
                        item.apply(fn)
        fn(self)
        return self
