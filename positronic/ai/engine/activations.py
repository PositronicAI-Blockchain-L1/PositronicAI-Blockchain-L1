"""
Positronic Neural Engine - Activation Functions
================================================

Callable activation function classes that operate on Tensor objects and
support automatic differentiation through backward closures.

Each activation takes a Tensor input and returns a new Tensor with a
properly defined `_backward` closure for backpropagation.
"""

import numpy as np
from positronic.ai.engine.tensor import Tensor


class ReLU:
    """Rectified Linear Unit activation: max(0, x).

    Gradient is 1 where x > 0, and 0 elsewhere.
    """

    def __call__(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + out.grad * (x.data > 0)

        out._backward = _backward
        return out


class LeakyReLU:
    """Leaky ReLU activation: x if x > 0, else alpha * x.

    Allows a small gradient when the unit is not active, which can help
    prevent dead neurons during training.

    Args:
        alpha: Slope for negative values. Default is 0.01.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: Tensor) -> Tensor:
        out_data = np.where(x.data > 0, x.data, self.alpha * x.data)
        out = Tensor(out_data, _children=(x,))

        def _backward():
            if x.requires_grad:
                grad_mask = np.where(x.data > 0, 1.0, self.alpha)
                x.grad = x.grad + out.grad * grad_mask

        out._backward = _backward
        return out


class Sigmoid:
    """Logistic sigmoid activation: 1 / (1 + exp(-x)).

    Gradient: sigmoid(x) * (1 - sigmoid(x)).
    """

    def __call__(self, x: Tensor) -> Tensor:
        # Numerically stable sigmoid using np.clip to avoid overflow
        clipped = np.clip(x.data, -500, 500)
        sig = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(sig, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out


class Tanh:
    """Hyperbolic tangent activation.

    Gradient: 1 - tanh(x)^2.
    """

    def __call__(self, x: Tensor) -> Tensor:
        t = np.tanh(x.data)
        out = Tensor(t, _children=(x,))

        def _backward():
            if x.requires_grad:
                x.grad = x.grad + out.grad * (1.0 - t * t)

        out._backward = _backward
        return out


class GELU:
    """Gaussian Error Linear Unit (approximate).

    Uses the tanh approximation:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    This is the activation used in BERT and GPT-family models.
    """

    def __call__(self, x: Tensor) -> Tensor:
        c = np.sqrt(2.0 / np.pi)
        x3 = x.data ** 3
        inner = c * (x.data + 0.044715 * x3)
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * x.data * (1.0 + tanh_inner)
        out = Tensor(out_data, _children=(x,))

        def _backward():
            if x.requires_grad:
                # d/dx GELU(x) = 0.5*(1+tanh(s)) + 0.5*x*(1-tanh(s)^2)*ds/dx
                # where s = c*(x + 0.044715*x^3), ds/dx = c*(1 + 0.134145*x^2)
                sech2 = 1.0 - tanh_inner * tanh_inner
                ds_dx = c * (1.0 + 3.0 * 0.044715 * x.data ** 2)
                grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x.data * sech2 * ds_dx
                x.grad = x.grad + out.grad * grad

        out._backward = _backward
        return out


class Softmax:
    """Softmax activation along a specified axis.

    Computes exp(x_i) / sum(exp(x_j)) along the given axis, with numerical
    stability via the max-subtraction trick.

    Args:
        axis: The axis along which to compute softmax. Default is -1.
    """

    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, x: Tensor) -> Tensor:
        # Numerically stable softmax: subtract max
        shifted = x.data - np.max(x.data, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted)
        s = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        out = Tensor(s, _children=(x,))

        axis = self.axis

        def _backward():
            if x.requires_grad:
                # Jacobian-vector product for softmax:
                # dx_i = s_i * (dout_i - sum_j(s_j * dout_j))
                dot = np.sum(out.grad * s, axis=axis, keepdims=True)
                x.grad = x.grad + s * (out.grad - dot)

        out._backward = _backward
        return out


class Swish:
    """Swish activation: x * sigmoid(x).

    Also known as SiLU (Sigmoid Linear Unit).
    Gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """

    def __call__(self, x: Tensor) -> Tensor:
        clipped = np.clip(x.data, -500, 500)
        sig = 1.0 / (1.0 + np.exp(-clipped))
        out_data = x.data * sig
        out = Tensor(out_data, _children=(x,))

        def _backward():
            if x.requires_grad:
                grad = sig * (1.0 + x.data * (1.0 - sig))
                x.grad = x.grad + out.grad * grad

        out._backward = _backward
        return out
