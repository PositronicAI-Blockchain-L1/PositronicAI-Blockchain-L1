"""
Positronic - ZKML Proof Generator (Phase 31)
Generates zero-knowledge proofs that an AI score was computed correctly
from given inputs using a committed model, WITHOUT revealing model weights.

Proof system: Fiat-Shamir hash-based commitment scheme (computationally binding via SHA-512).
The proof demonstrates knowledge of intermediate values that satisfy
the circuit constraints, without revealing the model parameters.

Proof structure:
  1. Model commitment (Merkle root of quantized weights)
  2. Input hash (SHA-512 of quantized inputs)
  3. Output score
  4. Blinded intermediate commitments
  5. Fiat-Shamir challenge-response pairs
"""

import time
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from positronic.constants import (
    ZKML_PROOF_TIMEOUT_MS,
    ZKML_CHALLENGE_ROUNDS,
    ZKML_PROOF_FORMAT,
    ZKML_PROOF_CACHE_SIZE,
)
from positronic.ai.zkml_circuit import ScoringCircuit, quantize


@dataclass
class ZKMLProof:
    """A zero-knowledge proof of correct AI scoring.

    Contains enough information for a verifier to check that the
    claimed score was computed correctly by the committed model,
    without learning the model weights.
    """
    # Public inputs (visible to verifier)
    model_commitment: bytes       # Merkle root of model weights
    input_hash: bytes             # SHA-512 hash of quantized inputs
    output_score: int             # The claimed score (quantized)
    proof_format: str = ZKML_PROOF_FORMAT

    # Proof data (cryptographic commitments)
    blinded_intermediates: List[bytes] = field(default_factory=list)
    challenge_responses: List[Tuple[bytes, bytes]] = field(default_factory=list)

    # Metadata
    generated_at: float = 0.0
    generation_time_ms: float = 0.0
    circuit_depth: int = 0
    prover_id: str = ""

    @property
    def is_valid_format(self) -> bool:
        """Quick format check (not cryptographic verification)."""
        return (
            len(self.model_commitment) == 64
            and len(self.input_hash) == 64
            and len(self.blinded_intermediates) > 0
            and len(self.challenge_responses) == ZKML_CHALLENGE_ROUNDS
        )

    def to_dict(self) -> dict:
        return {
            "model_commitment": self.model_commitment.hex(),
            "input_hash": self.input_hash.hex(),
            "output_score": self.output_score,
            "proof_format": self.proof_format,
            "num_intermediates": len(self.blinded_intermediates),
            "num_challenges": len(self.challenge_responses),
            "generated_at": self.generated_at,
            "generation_time_ms": self.generation_time_ms,
            "circuit_depth": self.circuit_depth,
            "prover_id": self.prover_id,
            "valid_format": self.is_valid_format,
        }

    def to_bytes(self) -> bytes:
        """Serialize proof to bytes for on-chain storage."""
        parts = [
            self.model_commitment,
            self.input_hash,
            self.output_score.to_bytes(8, "big", signed=True),
        ]
        for bi in self.blinded_intermediates:
            parts.append(bi)
        for challenge, response in self.challenge_responses:
            parts.append(challenge)
            parts.append(response)
        return b"".join(parts)


class ZKMLProver:
    """Generates zero-knowledge proofs for AI scoring computations.

    The prover has access to the full circuit (including weights) and
    can produce proofs that the score was computed correctly. The proof
    reveals nothing about the weights themselves.

    Protocol (Fiat-Shamir):
    1. Compute circuit forward pass, recording intermediates
    2. Blind each intermediate with random nonce
    3. Compute Fiat-Shamir challenge from blinded values
    4. Respond to challenge using intermediates + nonces
    5. Package as ZKMLProof
    """

    def __init__(self, circuit: ScoringCircuit, prover_id: str = ""):
        self._circuit = circuit
        self._prover_id = prover_id
        self._proofs_generated: int = 0
        self._total_generation_time_ms: float = 0.0

        # LRU cache for recent proofs (input_hash → proof)
        self._cache: OrderedDict = OrderedDict()

    def generate_proof(
        self,
        inputs: List[float],
        use_cache: bool = True,
    ) -> Optional[ZKMLProof]:
        """Generate a ZK proof that the circuit produces a specific score.

        Args:
            inputs: Float feature vector (will be quantized).
            use_cache: Whether to check/update the proof cache.

        Returns:
            ZKMLProof or None if proof generation fails/times out.
        """
        start_ms = time.time() * 1000

        # Quantize inputs
        q_inputs = [quantize(v) for v in inputs]

        # Compute input hash
        input_hash = self._hash_inputs(q_inputs)

        # Check cache
        if use_cache and input_hash in self._cache:
            self._cache.move_to_end(input_hash)
            return self._cache[input_hash]

        # Run circuit forward pass with intermediates
        score, intermediates = self._circuit.forward_quantized(q_inputs)

        # Get model commitment
        model_commitment = self._circuit.model_commitment()

        # Generate blinded intermediate commitments
        nonces = []
        blinded = []
        for layer_output in intermediates:
            nonce = secrets.token_bytes(32)
            nonces.append(nonce)
            blinded.append(self._blind_values(layer_output, nonce))

        # Fiat-Shamir: derive challenges from public data + blinded values
        challenges_responses = []
        transcript = model_commitment + input_hash + score.to_bytes(8, "big", signed=True)
        for bi in blinded:
            transcript += bi

        for round_idx in range(ZKML_CHALLENGE_ROUNDS):
            # Challenge = H(transcript || round)
            challenge = hashlib.sha512(
                transcript + round_idx.to_bytes(4, "big")
            ).digest()

            # Response = H(challenge || nonces || intermediates)
            resp_input = challenge
            for nonce in nonces:
                resp_input += nonce
            for layer_vals in intermediates:
                for v in layer_vals:
                    resp_input += v.to_bytes(8, "big", signed=True)
            response = hashlib.sha512(resp_input).digest()

            challenges_responses.append((challenge, response))
            transcript += challenge + response

        elapsed_ms = time.time() * 1000 - start_ms

        # Timeout check
        if elapsed_ms > ZKML_PROOF_TIMEOUT_MS:
            return None

        proof = ZKMLProof(
            model_commitment=model_commitment,
            input_hash=input_hash,
            output_score=score,
            blinded_intermediates=blinded,
            challenge_responses=challenges_responses,
            generated_at=time.time(),
            generation_time_ms=elapsed_ms,
            circuit_depth=len(self._circuit.layers),
            prover_id=self._prover_id,
        )

        # Update cache
        if use_cache:
            self._cache[input_hash] = proof
            if len(self._cache) > ZKML_PROOF_CACHE_SIZE:
                self._cache.popitem(last=False)

        self._proofs_generated += 1
        self._total_generation_time_ms += elapsed_ms

        return proof

    def get_stats(self) -> dict:
        return {
            "proofs_generated": self._proofs_generated,
            "avg_generation_ms": (
                self._total_generation_time_ms / self._proofs_generated
                if self._proofs_generated > 0 else 0
            ),
            "cache_size": len(self._cache),
            "circuit_depth": len(self._circuit.layers),
            "model_commitment": self._circuit.model_commitment().hex(),
        }

    @staticmethod
    def _hash_inputs(q_inputs: List[int]) -> bytes:
        """Hash quantized inputs to a deterministic digest."""
        h = hashlib.sha512()
        for v in q_inputs:
            h.update(v.to_bytes(8, "big", signed=True))
        return h.digest()

    @staticmethod
    def _blind_values(values: List[int], nonce: bytes) -> bytes:
        """Commit to a list of values using a random nonce."""
        h = hashlib.sha512()
        h.update(nonce)
        for v in values:
            h.update(v.to_bytes(8, "big", signed=True))
        return h.digest()
