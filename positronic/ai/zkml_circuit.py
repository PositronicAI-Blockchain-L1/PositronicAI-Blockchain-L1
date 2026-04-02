"""
Positronic - ZKML Circuit Definition (Phase 31)
Defines the arithmetic circuit for provable PoNC scoring.

The circuit operates on quantized (fixed-point) integers to ensure
deterministic computation across all platforms. It mirrors the scoring
logic from the AI validation pipeline but in a form that can be
committed to and proven via ZK proofs.

Architecture:
  Input (quantized features) → Linear layers → ReLU → Output (score)
  All operations use integer arithmetic with a fixed scaling factor.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from positronic.constants import ZKML_QUANTIZATION_BITS, ZKML_MAX_CIRCUIT_DEPTH


# Fixed-point scaling factor: 2^QUANTIZATION_BITS
SCALE = 1 << ZKML_QUANTIZATION_BITS  # 65536 for 16-bit


def quantize(value: float) -> int:
    """Convert float to fixed-point integer."""
    return int(round(value * SCALE))


def dequantize(value: int) -> float:
    """Convert fixed-point integer back to float."""
    return value / SCALE


def relu_q(x: int) -> int:
    """Quantized ReLU activation."""
    return max(0, x)


def clamp_q(x: int, lo: int, hi: int) -> int:
    """Clamp quantized value to range."""
    return max(lo, min(hi, x))


@dataclass
class CircuitLayer:
    """A single linear layer in the circuit.

    Computes: output[j] = ReLU(sum(weights[j][i] * input[i]) + bias[j])
    All values are quantized integers.
    """
    weights: List[List[int]]   # [out_dim][in_dim] quantized weights
    biases: List[int]          # [out_dim] quantized biases
    apply_relu: bool = True

    @property
    def in_dim(self) -> int:
        return len(self.weights[0]) if self.weights else 0

    @property
    def out_dim(self) -> int:
        return len(self.weights)

    def forward(self, inputs: List[int]) -> List[int]:
        """Run forward pass through this layer."""
        outputs = []
        for j in range(self.out_dim):
            # Dot product + bias
            acc = self.biases[j]
            for i in range(self.in_dim):
                acc += (self.weights[j][i] * inputs[i]) >> ZKML_QUANTIZATION_BITS
            if self.apply_relu:
                acc = relu_q(acc)
            outputs.append(acc)
        return outputs

    def commitment(self) -> bytes:
        """Compute hash commitment of this layer's parameters."""
        h = hashlib.sha512()
        for row in self.weights:
            for w in row:
                h.update(w.to_bytes(8, "big", signed=True))
        for b in self.biases:
            h.update(b.to_bytes(8, "big", signed=True))
        h.update(b"\x01" if self.apply_relu else b"\x00")
        return h.digest()


@dataclass
class ScoringCircuit:
    """Arithmetic circuit for provable AI scoring.

    A sequence of linear layers that takes quantized feature inputs
    and produces a single score output. The circuit can be committed
    to (via Merkle root of layer commitments) without revealing weights.
    """
    layers: List[CircuitLayer] = field(default_factory=list)
    version: int = 1

    def add_layer(
        self,
        weights: List[List[float]],
        biases: List[float],
        apply_relu: bool = True,
    ):
        """Add a layer with float weights (auto-quantized)."""
        if len(self.layers) >= ZKML_MAX_CIRCUIT_DEPTH:
            raise ValueError(f"Circuit depth exceeds max {ZKML_MAX_CIRCUIT_DEPTH}")
        q_weights = [[quantize(w) for w in row] for row in weights]
        q_biases = [quantize(b) for b in biases]
        self.layers.append(CircuitLayer(q_weights, q_biases, apply_relu))

    def forward(self, inputs: List[float]) -> int:
        """Run the full circuit on float inputs, return quantized score."""
        x = [quantize(v) for v in inputs]
        for layer in self.layers:
            x = layer.forward(x)
        # Final output: single score, clamped to [0, 10000] basis points
        if not x:
            return 0
        return clamp_q(x[0], 0, quantize(10000.0))

    def forward_quantized(self, inputs: List[int]) -> Tuple[int, List[List[int]]]:
        """Run circuit on pre-quantized inputs.

        Returns (score, intermediate_values) where intermediate_values
        contains the output of each layer (needed for proof generation).
        """
        x = list(inputs)
        intermediates = [list(x)]  # Include input as first intermediate
        for layer in self.layers:
            x = layer.forward(x)
            intermediates.append(list(x))
        score = clamp_q(x[0], 0, quantize(10000.0)) if x else 0
        return score, intermediates

    def model_commitment(self) -> bytes:
        """Compute Merkle root commitment of all layer parameters.

        This commitment can be published on-chain so verifiers know
        which model was used, without seeing the actual weights.
        """
        if not self.layers:
            return hashlib.sha512(b"empty_circuit").digest()

        # Leaf hashes = individual layer commitments
        leaves = [layer.commitment() for layer in self.layers]

        # Build Merkle tree
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])  # Duplicate last if odd
            next_level = []
            for i in range(0, len(leaves), 2):
                combined = hashlib.sha512(leaves[i] + leaves[i + 1]).digest()
                next_level.append(combined)
            leaves = next_level

        return leaves[0]

    def input_dim(self) -> int:
        """Expected input dimension."""
        return self.layers[0].in_dim if self.layers else 0

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "num_layers": len(self.layers),
            "input_dim": self.input_dim(),
            "depth": len(self.layers),
            "commitment": self.model_commitment().hex(),
        }


def build_scoring_circuit(
    input_dim: int = 8,
    hidden_dim: int = 16,
    seed: int = 42,
) -> ScoringCircuit:
    """Build a default scoring circuit for PoNC validation.

    Uses deterministic pseudo-random weights seeded from the given seed.
    This ensures all validators build the same circuit from the same seed.

    Args:
        input_dim: Number of input features (transaction features).
        hidden_dim: Hidden layer width.
        seed: Deterministic seed for weight generation.
    """
    import random
    rng = random.Random(seed)

    circuit = ScoringCircuit()

    # Layer 1: input_dim → hidden_dim (with ReLU)
    w1 = [[rng.gauss(0, 0.5) for _ in range(input_dim)] for _ in range(hidden_dim)]
    b1 = [rng.gauss(0, 0.1) for _ in range(hidden_dim)]
    circuit.add_layer(w1, b1, apply_relu=True)

    # Layer 2: hidden_dim → hidden_dim (with ReLU)
    w2 = [[rng.gauss(0, 0.5) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
    b2 = [rng.gauss(0, 0.1) for _ in range(hidden_dim)]
    circuit.add_layer(w2, b2, apply_relu=True)

    # Layer 3: hidden_dim → 1 (no ReLU, output is score)
    w3 = [[rng.gauss(0, 0.5) for _ in range(hidden_dim)]]
    b3 = [5000.0]  # Bias toward midrange score
    circuit.add_layer(w3, b3, apply_relu=False)

    return circuit
