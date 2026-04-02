"""
Positronic - Functional Operations

Stateless utility functions for the neural engine.  All functions operate
on plain NumPy arrays and carry no internal state, making them safe to
call from any context (model forward passes, data pre-processing,
evaluation pipelines, etc.).
"""

import numpy as np
from typing import List, Optional


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Compute sinusoidal positional encodings for transformer models.

    Uses the encoding scheme from *Attention Is All You Need*: even
    dimensions receive sine values and odd dimensions receive cosine
    values with geometrically increasing wavelengths.

    Args:
        seq_len: Number of positions (tokens) in the sequence.
        d_model: Dimensionality of the model embeddings.

    Returns:
        Array of shape ``(seq_len, d_model)`` with float64 values.

    Raises:
        ValueError: If *seq_len* or *d_model* is not positive.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")

    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    dim = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
    angles = pos / np.power(10000.0, (2.0 * (dim // 2)) / d_model)

    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create an additive causal (autoregressive) attention mask.

    Positions that should **not** be attended to are set to ``-inf``
    (upper triangle).  Positions that **may** be attended to are ``0``.

    Args:
        seq_len: Sequence length for the square mask.

    Returns:
        Array of shape ``(seq_len, seq_len)`` with float64 values.

    Raises:
        ValueError: If *seq_len* is not positive.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    mask = np.zeros((seq_len, seq_len), dtype=np.float64)
    mask = np.where(np.triu(np.ones((seq_len, seq_len)), k=1) == 1, -np.inf, mask)
    return mask


def pad_sequences(
    sequences: List[np.ndarray],
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad a list of variable-length 1-D sequences to uniform length.

    Args:
        sequences: List of 1-D NumPy arrays (or list-like objects).
        max_len: Target length.  If ``None``, uses the length of the
            longest sequence in *sequences*.
        pad_value: Scalar used for padding.

    Returns:
        Array of shape ``(len(sequences), max_len)`` with the same dtype
        as the first sequence (defaults to float64 for empty inputs).

    Raises:
        ValueError: If *sequences* is empty or *max_len* is non-positive.
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list")

    # Normalise to numpy arrays
    arrays = [np.asarray(s) for s in sequences]
    inferred_len = max(len(a) for a in arrays)

    if max_len is None:
        max_len = inferred_len
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")

    dtype = arrays[0].dtype
    padded = np.full((len(arrays), max_len), pad_value, dtype=dtype)

    for i, arr in enumerate(arrays):
        length = min(len(arr), max_len)
        padded[i, :length] = arr[:length]

    return padded


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode an array of integer class indices.

    Args:
        indices: Integer array of any shape.  Values must be in
            ``[0, num_classes)``.
        num_classes: Number of classes (width of the one-hot dimension).

    Returns:
        Float array with shape ``(*indices.shape, num_classes)``.

    Raises:
        ValueError: If *num_classes* is not positive or indices are out
            of range.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    indices = np.asarray(indices, dtype=np.int64)

    if indices.size > 0:
        if np.any(indices < 0) or np.any(indices >= num_classes):
            raise ValueError(
                f"All indices must be in [0, {num_classes}). "
                f"Got min={indices.min()}, max={indices.max()}"
            )

    flat = indices.ravel()
    result = np.zeros((flat.size, num_classes), dtype=np.float64)
    result[np.arange(flat.size), flat] = 1.0
    return result.reshape((*indices.shape, num_classes))


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute cosine similarity between two arrays along the last axis.

    Args:
        a: First array.  Shape ``(..., D)``.
        b: Second array. Shape ``(..., D)``, broadcastable with *a*.
        eps: Small constant to avoid division by zero.

    Returns:
        Array of cosine similarities with the last dimension reduced.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    dot = np.sum(a * b, axis=-1)
    norm_a = np.sqrt(np.sum(a ** 2, axis=-1))
    norm_b = np.sqrt(np.sum(b ** 2, axis=-1))

    return dot / np.maximum(norm_a * norm_b, eps)


def l2_normalize(
    x: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    L2-normalize an array along the given axis.

    After normalization each slice along *axis* has unit L2 norm
    (up to numerical precision).

    Args:
        x: Input array.
        axis: Axis along which to normalize.
        eps: Small constant to prevent division by zero.

    Returns:
        Normalized array with the same shape and dtype as *x*.
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))
    return x / np.maximum(norm, eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax function along the given axis.

    Numerically stable implementation using the max-subtraction trick.

    Args:
        x: Input array of logits.
        axis: Axis along which to compute softmax.

    Returns:
        Array of the same shape where values along *axis* sum to 1.
    """
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log-softmax along the given axis.

    More numerically stable than ``np.log(softmax(x))``.

    Args:
        x: Input array of logits.
        axis: Axis along which to compute log-softmax.

    Returns:
        Array of the same shape containing log-probabilities.
    """
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.

    Uses the exact formulation:
    ``GELU(x) = x * Phi(x)`` where Phi is the standard Gaussian CDF.

    The approximation ``0.5 * x * (1 + tanh(...))`` is **not** used;
    instead we use ``scipy``-free exact form via ``erf``.

    Args:
        x: Input array.

    Returns:
        Activated array of the same shape.
    """
    from scipy.special import erf  # noqa: local import for optional dep

    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximate GELU activation (no scipy dependency).

    Uses the tanh approximation:
    ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``

    Args:
        x: Input array.

    Returns:
        Activated array of the same shape.
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
