"""
Positronic Neural Engine - Gradient-Based Optimizers

Provides optimizer implementations for updating Tensor parameters during
training. All optimizers operate on parameters from positronic.ai.engine.tensor
and support standard features like momentum, adaptive learning rates,
weight decay, and gradient clipping.

Optimizers:
    - SGD: Stochastic Gradient Descent with optional momentum
    - Adam: Adaptive Moment Estimation
    - AdamW: Adam with decoupled weight decay regularization
    - RMSProp: Root Mean Square Propagation

Utilities:
    - clip_grad_norm: Global gradient norm clipping
    - clip_grad_value: Per-element gradient value clipping
"""

from typing import List, Optional

import numpy as np

from positronic.ai.engine.tensor import Tensor


class Optimizer:
    """Base class for all optimizers.

    All optimizers operate on a list of Tensor parameters, updating their
    ``.data`` attribute in-place based on accumulated ``.grad`` values.

    Args:
        parameters: Iterable of Tensor parameters to optimize. Only
            parameters with ``requires_grad=True`` are retained.
        lr: Learning rate (step size). Must be positive.
    """

    def __init__(self, parameters: List[Tensor], lr: float = 0.001) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be >= 0.")
        self.parameters: List[Tensor] = [p for p in parameters if p.requires_grad]
        self.lr: float = lr

    def step(self) -> None:
        """Perform a single optimization step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def zero_grad(self) -> None:
        """Reset gradients of all managed parameters to zero."""
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum and weight decay.

    Implements the update rule::

        v_t = momentum * v_{t-1} + grad + weight_decay * param
        param = param - lr * v_t

    When ``momentum=0``, this reduces to vanilla SGD.

    Args:
        parameters: Iterable of Tensor parameters to optimize.
        lr: Learning rate. Default: ``0.01``.
        momentum: Momentum factor. Default: ``0.0`` (no momentum).
        weight_decay: L2 regularization coefficient. Default: ``0.0``.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Initialize velocity buffers to zero for each parameter
        self.velocity: List[np.ndarray] = [
            np.zeros_like(p.data) for p in self.parameters
        ]

    def step(self) -> None:
        """Perform a single SGD update step."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0.0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                p.data -= self.lr * self.velocity[i]
            else:
                p.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation).

    Maintains per-parameter exponential moving averages of the gradient
    (first moment) and squared gradient (second moment), with bias
    correction applied at each step.

    Args:
        parameters: Iterable of Tensor parameters to optimize.
        lr: Learning rate. Default: ``0.001``.
        beta1: Exponential decay rate for the first moment. Default: ``0.9``.
        beta2: Exponential decay rate for the second moment. Default: ``0.999``.
        eps: Small constant for numerical stability. Default: ``1e-8``.
        weight_decay: L2 regularization coefficient applied to the
            gradient before moment updates. Default: ``0.0``.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        # First moment estimates (mean of gradients)
        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]
        # Second moment estimates (mean of squared gradients)
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]
        # Global timestep counter for bias correction
        self.t: int = 0

    def step(self) -> None:
        """Perform a single Adam update step."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            # Apply L2 regularization to gradient
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Bias-corrected moment estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay regularization.

    Unlike standard Adam where weight decay is applied to the gradient,
    AdamW applies weight decay directly to the parameters before the
    Adam update. This decoupling leads to better generalization in
    practice (Loshchilov & Hutter, 2019).

    The update rule is::

        param = param - lr * weight_decay * param      (decoupled decay)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        parameters: Iterable of Tensor parameters to optimize.
        lr: Learning rate. Default: ``0.001``.
        beta1: Exponential decay rate for the first moment. Default: ``0.9``.
        beta2: Exponential decay rate for the second moment. Default: ``0.999``.
        eps: Small constant for numerical stability. Default: ``1e-8``.
        weight_decay: Decoupled weight decay coefficient. Default: ``0.01``.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]
        self.t: int = 0

    def step(self) -> None:
        """Perform a single AdamW update step."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            # Decoupled weight decay: applied directly to parameters
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data

            grad = p.grad

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad ** 2)

            # Bias-corrected moment estimates
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """RMSProp optimizer.

    Maintains a moving average of squared gradients to normalize the
    gradient, effectively adapting the learning rate per parameter.

    Args:
        parameters: Iterable of Tensor parameters to optimize.
        lr: Learning rate. Default: ``0.01``.
        alpha: Smoothing constant (decay rate). Default: ``0.99``.
        eps: Small constant for numerical stability. Default: ``1e-8``.
        weight_decay: L2 regularization coefficient. Default: ``0.0``.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        """Perform a single RMSProp update step."""
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            # Update running average of squared gradients
            self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * (grad ** 2)

            # Update parameters
            p.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


# ---------------------------------------------------------------------------
# Gradient clipping utilities
# ---------------------------------------------------------------------------


def clip_grad_norm(
    parameters: List[Tensor], max_norm: float, norm_type: float = 2.0
) -> float:
    """Clip gradients by their global norm.

    Computes the global norm of all parameter gradients concatenated
    together. If the total norm exceeds ``max_norm``, all gradients are
    scaled down proportionally so the total norm equals ``max_norm``.

    Args:
        parameters: Iterable of Tensor parameters whose gradients will
            be clipped.
        max_norm: Maximum allowed global norm.
        norm_type: Type of norm to use. Default: ``2.0`` (L2 norm).

    Returns:
        The total gradient norm (before clipping).
    """
    if max_norm < 0.0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")

    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return 0.0

    if norm_type == float("inf"):
        total_norm = max(np.max(np.abs(g)) for g in grads)
    else:
        total_norm = 0.0
        for g in grads:
            total_norm += np.sum(np.abs(g) ** norm_type)
        total_norm = float(total_norm ** (1.0 / norm_type))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad = p.grad * scale

    return total_norm


def clip_grad_value(parameters: List[Tensor], clip_value: float) -> None:
    """Clip gradients by clamping each element to ``[-clip_value, clip_value]``.

    Args:
        parameters: Iterable of Tensor parameters whose gradients will
            be clamped.
        clip_value: Maximum absolute value for any gradient element.
    """
    if clip_value < 0.0:
        raise ValueError(f"clip_value must be non-negative, got {clip_value}")

    for p in parameters:
        if p.grad is not None:
            p.grad = np.clip(p.grad, -clip_value, clip_value)
