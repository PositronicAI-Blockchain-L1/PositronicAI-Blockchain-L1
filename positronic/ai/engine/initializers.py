"""
Positronic Neural Engine - Weight Initializers

Provides functions for initializing neural network weight matrices as
NumPy arrays. Proper initialization is critical for stable training:
poor initialization can lead to vanishing or exploding gradients.

Guidelines:
    - Use ``he_normal`` or ``he_uniform`` for layers followed by ReLU.
    - Use ``xavier_uniform`` or ``xavier_normal`` for layers followed
      by sigmoid or tanh activations.
    - Use ``orthogonal`` for recurrent layers (LSTMs, GRUs).
    - Use ``zeros`` for biases (default) and ``ones`` sparingly.

All functions accept a ``shape`` tuple and return a ``numpy.ndarray``.
"""

from typing import Tuple, Union

import numpy as np

Shape = Union[Tuple[int, ...], list]


def _compute_fans(shape: Shape) -> Tuple[int, int]:
    """Compute fan-in and fan-out for a weight tensor shape.

    For 1-D shapes, fan_in and fan_out are both set to the single
    dimension. For 2-D shapes, fan_in is the first dimension and
    fan_out is the second. For higher-dimensional shapes (e.g.,
    convolutional kernels), the receptive field size is factored in.

    Args:
        shape: Shape tuple of the weight tensor.

    Returns:
        Tuple of (fan_in, fan_out).
    """
    if len(shape) == 1:
        return shape[0], shape[0]
    elif len(shape) == 2:
        return shape[0], shape[1]
    else:
        # Convolutional kernels: (out_channels, in_channels, *kernel_size)
        receptive_field = int(np.prod(shape[2:]))
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
        return fan_in, fan_out


def xavier_uniform(shape: Shape) -> np.ndarray:
    """Xavier/Glorot uniform initialization.

    Draws samples from a uniform distribution U(-limit, limit) where::

        limit = sqrt(6 / (fan_in + fan_out))

    Designed to preserve signal variance through layers with linear or
    sigmoid activations (Glorot & Bengio, 2010).

    Args:
        shape: Shape of the weight tensor.

    Returns:
        Initialized NumPy array.
    """
    fan_in, fan_out = _compute_fans(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def xavier_normal(shape: Shape) -> np.ndarray:
    """Xavier/Glorot normal initialization.

    Draws samples from a normal distribution N(0, std^2) where::

        std = sqrt(2 / (fan_in + fan_out))

    Args:
        shape: Shape of the weight tensor.

    Returns:
        Initialized NumPy array.
    """
    fan_in, fan_out = _compute_fans(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std


def he_normal(shape: Shape) -> np.ndarray:
    """He/Kaiming normal initialization for ReLU networks.

    Draws samples from a normal distribution N(0, std^2) where::

        std = sqrt(2 / fan_in)

    Designed to account for the zero-killing property of ReLU which
    halves the variance (He et al., 2015).

    Args:
        shape: Shape of the weight tensor.

    Returns:
        Initialized NumPy array.
    """
    fan_in, _ = _compute_fans(shape)
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def he_uniform(shape: Shape) -> np.ndarray:
    """He/Kaiming uniform initialization for ReLU networks.

    Draws samples from a uniform distribution U(-limit, limit) where::

        limit = sqrt(6 / fan_in)

    Args:
        shape: Shape of the weight tensor.

    Returns:
        Initialized NumPy array.
    """
    fan_in, _ = _compute_fans(shape)
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, shape)


def lecun_normal(shape: Shape) -> np.ndarray:
    """LeCun normal initialization.

    Draws samples from N(0, std^2) where ``std = sqrt(1 / fan_in)``.
    Appropriate for SELU activation functions.

    Args:
        shape: Shape of the weight tensor.

    Returns:
        Initialized NumPy array.
    """
    fan_in, _ = _compute_fans(shape)
    std = np.sqrt(1.0 / fan_in)
    return np.random.randn(*shape) * std


def orthogonal(shape: Shape, gain: float = 1.0) -> np.ndarray:
    """Orthogonal initialization.

    Generates an orthogonal (or semi-orthogonal) matrix via SVD
    decomposition of a random Gaussian matrix. Particularly effective
    for recurrent networks where it helps preserve gradient norms across
    many time steps (Saxe et al., 2014).

    Args:
        shape: Shape of the weight tensor. Must have at least 2 dimensions.
        gain: Multiplicative scaling factor. Default: ``1.0``.

    Returns:
        Initialized NumPy array with orthogonal rows or columns.
    """
    if len(shape) < 2:
        raise ValueError(
            f"Orthogonal initialization requires at least 2 dimensions, "
            f"got shape {shape}"
        )
    rows = shape[0]
    cols = int(np.prod(shape[1:]))
    flat_shape = (rows, cols)

    # Generate a random matrix and compute its SVD
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)

    # Choose u or v to match the desired flat shape
    q = u if u.shape == flat_shape else v

    return (gain * q).reshape(shape)


def zeros(shape: Shape) -> np.ndarray:
    """Initialize all elements to zero.

    Commonly used for bias vectors.

    Args:
        shape: Shape of the tensor.

    Returns:
        NumPy array of zeros.
    """
    return np.zeros(shape)


def ones(shape: Shape) -> np.ndarray:
    """Initialize all elements to one.

    Args:
        shape: Shape of the tensor.

    Returns:
        NumPy array of ones.
    """
    return np.ones(shape)


def constant(shape: Shape, value: float) -> np.ndarray:
    """Initialize all elements to a constant value.

    Args:
        shape: Shape of the tensor.
        value: Fill value.

    Returns:
        NumPy array filled with ``value``.
    """
    return np.full(shape, value)


def uniform(shape: Shape, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Initialize from a uniform distribution U(low, high).

    Args:
        shape: Shape of the tensor.
        low: Lower bound (inclusive). Default: ``0.0``.
        high: Upper bound (exclusive). Default: ``1.0``.

    Returns:
        Initialized NumPy array.
    """
    return np.random.uniform(low, high, shape)


def normal(shape: Shape, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    """Initialize from a normal distribution N(mean, std^2).

    Args:
        shape: Shape of the tensor.
        mean: Mean of the distribution. Default: ``0.0``.
        std: Standard deviation. Default: ``1.0``.

    Returns:
        Initialized NumPy array.
    """
    return np.random.randn(*shape) * std + mean
