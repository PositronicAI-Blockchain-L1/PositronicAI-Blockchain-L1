"""
Positronic - Neural Engine (Pure NumPy Deep Learning Framework)

The foundational computation engine for the PoNC AI Validation Gate.
Built entirely on NumPy with no external deep learning dependencies.

Components:
- Tensor: Automatic differentiation engine with full operator overloading
- Layers: Dense, Conv1D, LSTM, MultiHeadAttention, LayerNorm, BatchNorm, Dropout, Embedding
- Activations: ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Softmax, Swish
- Optimizers: SGD, Adam, AdamW with gradient clipping
- Losses: MSE, BCE, CrossEntropy, ContrastiveLoss, VAELoss
- Model: Base class for neural models with save/load/train/eval
- Serialization: Save/load model weights to/from bytes
- Random: Deterministic RNG for consensus-safe inference
"""

from positronic.ai.engine.tensor import Tensor
from positronic.ai.engine.random import DeterministicRNG, DeterministicContext
from positronic.ai.engine.model import Model
from positronic.ai.engine.layers import (
    Dense, Conv1D, LSTM, MultiHeadAttention,
    LayerNorm, BatchNorm, Dropout, Embedding,
)
from positronic.ai.engine.activations import (
    ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Softmax, Swish,
)
from positronic.ai.engine.optimizers import SGD, Adam, AdamW, clip_grad_norm
from positronic.ai.engine.losses import MSELoss, BCELoss, CrossEntropyLoss, ContrastiveLoss, VAELoss
from positronic.ai.engine.initializers import (
    xavier_uniform, xavier_normal, he_normal, he_uniform, orthogonal,
)
from positronic.ai.engine.serialization import serialize_state, deserialize_state, compute_checksum
from positronic.ai.engine.functional import (
    positional_encoding, create_causal_mask, pad_sequences,
    one_hot, cosine_similarity, l2_normalize,
)

__all__ = [
    # Core
    "Tensor",
    "Model",
    # Layers
    "Dense", "Conv1D", "LSTM", "MultiHeadAttention",
    "LayerNorm", "BatchNorm", "Dropout", "Embedding",
    # Activations
    "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "GELU", "Softmax", "Swish",
    # Optimizers
    "SGD", "Adam", "AdamW", "clip_grad_norm",
    # Losses
    "MSELoss", "BCELoss", "CrossEntropyLoss", "ContrastiveLoss", "VAELoss",
    # Initializers
    "xavier_uniform", "xavier_normal", "he_normal", "he_uniform", "orthogonal",
    # Serialization
    "serialize_state", "deserialize_state", "compute_checksum",
    # Functional
    "positional_encoding", "create_causal_mask", "pad_sequences",
    "one_hot", "cosine_similarity", "l2_normalize",
    # Random
    "DeterministicRNG", "DeterministicContext",
]
