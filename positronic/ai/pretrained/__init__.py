"""
Positronic - Pre-trained AI Model Weights

Provides baseline weights for AI models so they start with reasonable
initial values instead of random Xavier initialization.
"""

from positronic.ai.pretrained.weights import (
    get_baseline_weights,
    save_weights,
)

__all__ = [
    "get_baseline_weights",
    "save_weights",
]
