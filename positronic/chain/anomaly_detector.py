"""
Positronic - Transaction Anomaly Detector (TAD)
Uses an Autoencoder + statistical analysis to detect unusual transaction patterns.
Part of the PoNC AI Validation Gate.

The autoencoder learns normal transaction patterns. When a transaction's
reconstruction error is high, it's flagged as anomalous.
"""

import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

from positronic.ai.feature_extractor import TransactionFeatures

logger = get_logger(__name__)
from positronic.constants import AI_FEATURE_DIM


@dataclass
class NeuralLayer:
    """A single dense layer in the neural network."""
    weights: List[List[float]] = field(default_factory=list)
    biases: List[float] = field(default_factory=list)
    input_size: int = 0
    output_size: int = 0

    def initialize(self, input_size: int, output_size: int):
        """Xavier initialization for weights (deterministic with fixed seed)."""
        self.input_size = input_size
        self.output_size = output_size
        scale = math.sqrt(2.0 / (input_size + output_size))
        # Use fixed seed for deterministic initialization across all nodes
        rng = random.Random(42 + input_size * 1000 + output_size)
        self.weights = [
            [rng.gauss(0, scale) for _ in range(input_size)]
            for _ in range(output_size)
        ]
        self.biases = [0.0] * output_size

    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through the layer."""
        outputs = []
        for j in range(self.output_size):
            total = self.biases[j]
            for i in range(self.input_size):
                total += self.weights[j][i] * inputs[i]
            outputs.append(total)
        return outputs


def relu(x: List[float]) -> List[float]:
    """ReLU activation function."""
    return [max(0.0, v) for v in x]


def sigmoid(x: List[float]) -> List[float]:
    """Sigmoid activation function."""
    return [1.0 / (1.0 + math.exp(-min(max(v, -500), 500))) for v in x]


def tanh_activation(x: List[float]) -> List[float]:
    """Tanh activation function."""
    return [math.tanh(v) for v in x]


class Autoencoder:
    """
    Simple autoencoder for transaction anomaly detection.

    Architecture: 35 -> 16 -> 8 -> 4 -> 8 -> 16 -> 35
    (encoder: compress features, decoder: reconstruct)

    High reconstruction error = anomalous transaction.
    """

    def __init__(self, input_size: int = AI_FEATURE_DIM):
        self.input_size = input_size

        # Encoder layers
        self.enc1 = NeuralLayer()
        self.enc1.initialize(input_size, 16)
        self.enc2 = NeuralLayer()
        self.enc2.initialize(16, 8)
        self.enc3 = NeuralLayer()
        self.enc3.initialize(8, 4)

        # Decoder layers
        self.dec1 = NeuralLayer()
        self.dec1.initialize(4, 8)
        self.dec2 = NeuralLayer()
        self.dec2.initialize(8, 16)
        self.dec3 = NeuralLayer()
        self.dec3.initialize(16, input_size)

        # Training state
        self.trained = False
        self.training_samples = 0
        self.mean_error: float = 0.0
        self.std_error: float = 1.0
        self._error_history: List[float] = []

        # Real neural model (VAE) - lazy import for graceful degradation
        try:
            from positronic.ai.models.vae import VAE
            from positronic.ai.engine.optimizers import Adam
            self._vae = VAE(input_dim=input_size, latent_dim=8)
            self._optimizer = Adam(self._vae.parameters(), lr=0.001)
            self._neural_available = True
        except (ImportError, Exception):
            self._vae = None
            self._optimizer = None
            self._neural_available = False

        self._use_neural = False
        self._neural_threshold = 500  # Switch to neural after this many samples
        self._neural_errors: int = 0
        self._consecutive_neural_failures: int = 0

    def encode(self, features: List[float]) -> List[float]:
        """Encode features to latent representation."""
        x = relu(self.enc1.forward(features))
        x = relu(self.enc2.forward(x))
        x = tanh_activation(self.enc3.forward(x))
        return x

    def decode(self, latent: List[float]) -> List[float]:
        """Decode latent representation back to features."""
        x = relu(self.dec1.forward(latent))
        x = relu(self.dec2.forward(x))
        x = self.dec3.forward(x)
        return x

    def forward(self, features: List[float]) -> Tuple[List[float], float]:
        """
        Full forward pass: encode then decode.
        Returns (reconstruction, reconstruction_error).
        """
        latent = self.encode(features)
        reconstruction = self.decode(latent)

        # Mean squared error
        mse = sum(
            (a - b) ** 2 for a, b in zip(features, reconstruction)
        ) / len(features)

        return reconstruction, mse

    def compute_anomaly_score(self, features: TransactionFeatures) -> float:
        """
        Compute anomaly score for a transaction (0 = normal, 1 = anomalous).
        Uses reconstruction error compared to historical distribution.
        """
        vector = features.to_vector()

        # Normalize input
        normalized = self._normalize(vector)

        # Heuristic floor: rule-based signals must never be suppressed by any model
        heuristic = self._heuristic_score(features)

        # Try neural (VAE) scoring first
        if self._use_neural and self._vae is not None:
            try:
                x = np.array(normalized, dtype=np.float32).reshape(1, -1)
                score = self._vae.compute_anomaly_score(x)
                self._consecutive_neural_failures = 0
                vae_score = float(min(max(score, 0.0), 1.0))
                return max(vae_score, heuristic)
            except Exception as e:
                logger.debug("VAE scoring error (neural fallback): %s", e)
                self._neural_errors += 1  # Neural fallback: VAE scoring
                self._consecutive_neural_failures += 1
                if self._consecutive_neural_failures > 10:
                    self._use_neural = False  # Deactivate neural on degradation

        # LEGACY: original autoencoder scoring (unchanged)
        _, error = self.forward(normalized)

        if not self.trained or self.std_error == 0:
            # Before training: use heuristic scoring
            return heuristic

        # Z-score of the reconstruction error
        z_score = (error - self.mean_error) / max(self.std_error, 1e-10)

        # Convert z-score to 0-1 probability using sigmoid
        trained_score = 1.0 / (1.0 + math.exp(-z_score + 2))  # Shifted sigmoid
        trained_score = min(max(trained_score, 0.0), 1.0)

        # Heuristic floor (already computed above)
        return max(trained_score, heuristic)

    def train_step(self, features_batch: List[List[float]], learning_rate: float = 0.001):
        """
        One training step (simplified gradient-free approach).
        Uses evolutionary strategy: perturb weights, keep if error decreases.
        """
        for features in features_batch:
            normalized = self._normalize(features)
            _, error = self.forward(normalized)

            self._error_history.append(error)
            if len(self._error_history) > 10000:
                self._error_history = self._error_history[-10000:]

            # Update statistics
            self.training_samples += 1

        if self._error_history:
            self.mean_error = sum(self._error_history) / len(self._error_history)
            variance = sum(
                (e - self.mean_error) ** 2 for e in self._error_history
            ) / len(self._error_history)
            self.std_error = math.sqrt(variance)

        # Perturbation-based training
        self._perturb_weights(learning_rate)

        if self.training_samples >= 100:
            self.trained = True

        # NEW: Real neural (VAE) training via backpropagation
        if self._neural_available and self._vae is not None and len(features_batch) > 0:
            try:
                batch = np.array(
                    [self._normalize(f) for f in features_batch],
                    dtype=np.float32,
                )
                self._vae.train_step(batch, self._optimizer)
            except Exception as e:
                logger.debug("VAE training error (neural fallback): %s", e)
                self._neural_errors += 1  # Neural fallback: VAE training

        # Switch to neural scoring after sufficient training samples
        if self.training_samples >= self._neural_threshold and self._neural_available:
            self._use_neural = True

    def _perturb_weights(self, lr: float):
        """Evolutionary weight update: randomly perturb and keep improvements."""
        all_layers = [self.enc1, self.enc2, self.enc3, self.dec1, self.dec2, self.dec3]

        for layer in all_layers:
            for j in range(layer.output_size):
                for i in range(layer.input_size):
                    layer.weights[j][i] += random.gauss(0, lr)
                layer.biases[j] += random.gauss(0, lr * 0.1)

    def _normalize(self, vector: List[float]) -> List[float]:
        """Simple min-max style normalization."""
        result = []
        for v in vector:
            if abs(v) > 1e6:
                v = math.log10(abs(v) + 1) * (1 if v > 0 else -1)
            result.append(v / max(abs(v), 1.0) if v != 0 else 0.0)
        return result

    def _heuristic_score(self, features: TransactionFeatures) -> float:
        """
        Heuristic scoring using rule-based checks on known suspicious patterns.
        Also serves as a floor for trained models: obvious anomalies must never
        be suppressed by learned weights.
        """
        score = 0.0

        # High value relative to balance — graduated severity
        if features.sender_balance_ratio > 10:
            score += 0.5   # Extreme: value is 10x+ the sender's balance
        elif features.sender_balance_ratio >= 0.9:
            score += 0.3
        elif features.sender_balance_ratio > 0.5:
            score += 0.1

        # Unusual value deviation
        if features.sender_value_deviation > 5.0:
            score += 0.2
        elif features.sender_value_deviation > 2.0:
            score += 0.1

        # Burst activity
        if features.is_burst:
            score += 0.2

        # New account with large value
        if features.sender_nonce < 3 and features.sender_balance_ratio > 0.5:
            score += 0.15

        # Low reputation
        if features.sender_reputation < 0.5:
            score += 0.2

        # Very high gas price — graduated severity
        if features.gas_price_vs_median >= 100:
            score += 0.3   # Extreme: 100x+ the median gas price
        elif features.gas_price_vs_median > 10:
            score += 0.15

        # Large data payload (potential exploit or gas bomb)
        if features.data_size > 5000:
            score += 0.2

        # Contract interaction from untrusted sender
        if features.is_contract_interaction and features.sender_reputation < 0.5:
            score += 0.25
            # New account deploying/calling contracts is extra suspicious
            if features.sender_nonce < 3:
                score += 0.20

        return min(score, 1.0)

    def get_stats(self) -> dict:
        return {
            "trained": self.trained,
            "training_samples": self.training_samples,
            "mean_error": self.mean_error,
            "std_error": self.std_error,
            "neural_active": self._use_neural,
            "neural_errors": self._neural_errors,
        }
