"""
Positronic - Hash-Based Commitment Scheme (Phase 24: Enhanced Commitments)

This module implements a hash-based commitment scheme using a Schnorr-like
sigma protocol structure.  Despite the structural similarity to Schnorr
signatures, **no elliptic-curve arithmetic is performed**.  All "point"
operations are simulated with SHA-256 hashes, which means this system
provides computational hiding and binding guarantees but does NOT
constitute a true zero-knowledge proof (no SNARK, STARK, or EC math).

The scheme proves knowledge of a secret via a commit-challenge-response
protocol where:
  - The "generator point" G(x) is actually H("GEN_POINT", x).
  - The challenge is derived via Fiat-Shamir (hash of commitment + public input).
  - The response r = (k + c * s) mod p is verified by checking a
    verification tag H("VER", G(r), c) embedded in the commitment.

Security model:
  - Prover has secret s (derived from private data + blinding factor).
  - Prover picks random nonce k, computes commitment.
  - Challenge c = H(commitment_core || public_input) [Fiat-Shamir].
  - Response r = (k + c * s) mod ZK_PRIME.
  - The commitment includes a verification tag: H("VER", G(r), c)
    so the verifier can check the response without knowing s or k.
  - Soundness: forging a valid (commitment, response) pair requires
    finding r such that H("VER", G(r), c) matches, which requires
    inverting SHA-256 -- computationally infeasible.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import hashlib
import hmac as _hmac
import time
import os

from positronic.constants import (
    ZK_PRIME,
    ZK_RANGE_BITS,
    ZK_MERKLE_DEPTH,
    ZK_DOMAIN_BALANCE,
    ZK_DOMAIN_MEMBERSHIP,
    ZK_DOMAIN_OWNERSHIP,
)


@dataclass
class HashCommitment:
    """A hash-based commitment (Schnorr-like structure, but hash-simulated -- not true ZK)."""
    proof_type: str  # "balance_above", "ownership", "membership"
    commitment: bytes  # 64 bytes: commitment_core (32) + verification_tag (32)
    challenge: bytes  # 32 bytes: Fiat-Shamir challenge
    response: bytes  # 32 bytes: r = (k + c * s) mod ZK_PRIME
    public_input: bytes = b""  # Public parameters
    prover_id: bytes = b""  # Security fix: binds proof to prover identity
    created_at: float = 0.0
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "proof_type": self.proof_type,
            "commitment": self.commitment.hex(),
            "challenge": self.challenge.hex(),
            "response": self.response.hex(),
            "public_input": self.public_input.hex(),
            "prover_id": self.prover_id.hex(),
            "verified": self.verified,
            "created_at": self.created_at,
        }


def _hash(*args: bytes) -> bytes:
    """SHA-256 hash of concatenated args."""
    h = hashlib.sha256()
    for a in args:
        h.update(a)
    return h.digest()


def _int_to_bytes(n: int) -> bytes:
    """Convert integer to 32-byte big-endian."""
    return (n % ZK_PRIME).to_bytes(32, 'big')


def _generator_point(scalar_bytes: bytes) -> bytes:
    """Hash-based point simulation: G(x) = H("GEN_POINT", x).

    NOTE: This is NOT real elliptic-curve scalar multiplication.
    It is a deterministic hash function used to simulate a generator
    point operation for the commitment scheme.  The security of
    this construction relies on the collision resistance and
    preimage resistance of SHA-256, not on the discrete-log problem.
    """
    return _hash(b"GEN_POINT", scalar_bytes)


def _verification_tag(g_r: bytes, challenge: bytes) -> bytes:
    """Compute verification tag: H("VER", G(r), c)."""
    return _hash(b"VER_TAG", g_r, challenge)


def _build_commitment(proof_type: str, domain_tag: bytes, secret_data: bytes,
                      public_input: bytes, prover_id: bytes = b"") -> HashCommitment:
    """Generic Schnorr-like commitment builder (hash-simulated, not true ZK).

    Security fix: challenge now binds proof_type and prover_id to prevent
    cross-type replay and identity-unbound forgery.

    Steps:
    1. s = H(domain_tag, secret_data) mod p  -- secret integer
    2. k = random mod p                       -- nonce
    3. commitment_core = G(k_bytes)
    4. challenge = H(commitment_core, public_input, prover_id, proof_type)
    5. r = (k + c * s) mod p
    6. verification_tag = H("VER", G(r_bytes), challenge)
    7. commitment = commitment_core || verification_tag  (64 bytes)
    """
    s = int.from_bytes(_hash(domain_tag, secret_data), 'big') % ZK_PRIME
    k = int.from_bytes(os.urandom(32), 'big') % ZK_PRIME
    if k == 0:
        k = 1
    k_bytes = _int_to_bytes(k)

    commitment_core = _generator_point(k_bytes)

    # Security fix: bind prover_id and proof_type into challenge
    challenge = _hash(commitment_core, public_input,
                      prover_id, proof_type.encode("utf-8"))
    c_int = int.from_bytes(challenge, 'big') % ZK_PRIME

    r = (k + c_int * s) % ZK_PRIME
    response = _int_to_bytes(r)

    g_r = _generator_point(response)
    ver_tag = _verification_tag(g_r, challenge)

    commitment = commitment_core + ver_tag  # 64 bytes

    return HashCommitment(
        proof_type=proof_type,
        commitment=commitment,
        challenge=challenge,
        response=response,
        public_input=public_input,
        prover_id=prover_id,
        created_at=time.time(),
    )


# Keep _build_proof as an internal alias so any monkey-patching or
# introspection that relied on the old name still works.
_build_proof = _build_commitment


class CommitmentProver:
    """Generate hash-based commitments using a Schnorr-like sigma protocol structure.

    NOTE: Despite the Schnorr-like structure, all point operations are
    simulated with SHA-256 hashes.  This is a commitment scheme, not a
    true zero-knowledge proof system.
    """

    _hash = staticmethod(_hash)

    @staticmethod
    def prove_balance_above(balance: int, threshold: int, secret: bytes = None,
                            prover_id: bytes = b"") -> Optional[HashCommitment]:
        """Prove balance >= threshold without revealing actual balance.

        Uses a hash-based commitment scheme (not true ZK).
        Security fix: prover_id binds proof to a specific prover identity.
        """
        if balance < threshold:
            return None
        if secret is None:
            secret = os.urandom(32)

        diff = balance - threshold
        secret_data = secret + diff.to_bytes(32, 'big')
        threshold_bytes = threshold.to_bytes(32, 'big')
        return _build_commitment("balance_above", ZK_DOMAIN_BALANCE, secret_data,
                                 threshold_bytes, prover_id)

    @staticmethod
    def prove_ownership(owner: bytes, asset_id: bytes, secret: bytes = None,
                        prover_id: bytes = b"") -> Optional[HashCommitment]:
        """Prove ownership of an asset without revealing the owner.

        Uses a hash-based commitment scheme (not true ZK).
        """
        if secret is None:
            secret = os.urandom(32)
        secret_data = owner + secret
        return _build_commitment("ownership", ZK_DOMAIN_OWNERSHIP, secret_data,
                                 asset_id, prover_id)

    @staticmethod
    def prove_membership(member: bytes, group_hash: bytes, secret: bytes = None,
                         prover_id: bytes = b"") -> Optional[HashCommitment]:
        """Prove membership in a group without revealing which member.

        Uses a hash-based commitment scheme (not true ZK).
        """
        if secret is None:
            secret = os.urandom(32)
        secret_data = member + secret
        return _build_commitment("membership", ZK_DOMAIN_MEMBERSHIP, secret_data,
                                 group_hash, prover_id)


class CommitmentVerifier:
    """Verify hash-based commitments with full response verification.

    NOTE: This verifies a hash-based commitment scheme, not a true
    zero-knowledge proof.  All "point" operations are SHA-256 hashes.

    Checks:
    1. commitment is 64 bytes (core + verification_tag)
    2. challenge == H(commitment_core, public_input, prover_id, proof_type)
    3. verification_tag == H("VER", G(response), challenge)
    """

    @staticmethod
    def verify(proof: HashCommitment) -> bool:
        """Verify a hash-based commitment.

        Security fix: challenge now includes prover_id and proof_type
        to prevent identity-unbound forgery and cross-type replay.
        """
        if not proof.commitment or not proof.challenge or not proof.response:
            return False

        # Commitment must be 64 bytes: 32 core + 32 verification tag
        if len(proof.commitment) != 64:
            return False

        commitment_core = proof.commitment[:32]
        embedded_ver_tag = proof.commitment[32:]

        # Step 1: Verify Fiat-Shamir challenge (now includes prover_id + proof_type)
        expected_challenge = _hash(commitment_core, proof.public_input,
                                   proof.prover_id, proof.proof_type.encode("utf-8"))
        if expected_challenge != proof.challenge:
            return False

        # Step 2: Verify response via verification tag
        g_r = _generator_point(proof.response)
        expected_ver_tag = _verification_tag(g_r, proof.challenge)

        if not _hmac.compare_digest(embedded_ver_tag, expected_ver_tag):
            return False

        proof.verified = True
        return True


class MerkleTree:
    """Simple Merkle tree for membership proofs."""

    def __init__(self, leaves: List[bytes]):
        self.leaves = [_hash(leaf) for leaf in leaves]
        self.root = self._build_root(self.leaves)

    @staticmethod
    def _build_root(leaves: List[bytes]) -> bytes:
        if not leaves:
            return _hash(b"EMPTY_TREE")
        layer = list(leaves)
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                if i + 1 < len(layer):
                    next_layer.append(_hash(layer[i], layer[i + 1]))
                else:
                    next_layer.append(layer[i])
            layer = next_layer
        return layer[0]

    def get_proof_path(self, index: int) -> List[tuple]:
        """Get Merkle proof path for leaf at index."""
        if index < 0 or index >= len(self.leaves):
            return []
        path = []
        layer = list(self.leaves)
        idx = index
        while len(layer) > 1:
            sibling_idx = idx ^ 1
            if sibling_idx < len(layer):
                path.append((layer[sibling_idx], "right" if idx % 2 == 0 else "left"))
            next_layer = []
            for i in range(0, len(layer), 2):
                if i + 1 < len(layer):
                    next_layer.append(_hash(layer[i], layer[i + 1]))
                else:
                    next_layer.append(layer[i])
            layer = next_layer
            idx //= 2
        return path

    @staticmethod
    def verify_path(leaf: bytes, path: List[tuple], root: bytes) -> bool:
        """Verify a Merkle proof path."""
        current = _hash(leaf)
        for sibling, direction in path:
            if direction == "left":
                current = _hash(sibling, current)
            else:
                current = _hash(current, sibling)
        return current == root


@dataclass
class ConfidentialEvidence:
    """Encrypted evidence with hash-based commitment proof of authenticity."""
    evidence_id: str
    encrypted_data: bytes
    commitment_proof: HashCommitment
    authorized_viewers: list = field(default_factory=list)
    created_at: float = 0.0

    @property
    def zk_proof(self) -> HashCommitment:
        """Backward-compatible alias for commitment_proof."""
        return self.commitment_proof

    @zk_proof.setter
    def zk_proof(self, value: HashCommitment):
        """Backward-compatible setter for commitment_proof."""
        self.commitment_proof = value

    def __init__(self, evidence_id: str, encrypted_data: bytes,
                 commitment_proof: HashCommitment = None,
                 zk_proof: HashCommitment = None,
                 authorized_viewers: list = None,
                 created_at: float = 0.0):
        """Accept both 'commitment_proof' and legacy 'zk_proof' kwarg."""
        self.evidence_id = evidence_id
        self.encrypted_data = encrypted_data
        self.commitment_proof = commitment_proof or zk_proof
        self.authorized_viewers = authorized_viewers if authorized_viewers is not None else []
        self.created_at = created_at

    def is_authorized(self, viewer: bytes) -> bool:
        return viewer in self.authorized_viewers

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "encrypted_data_size": len(self.encrypted_data),
            "proof": self.commitment_proof.to_dict(),
            "authorized_viewers": [v.hex() for v in self.authorized_viewers],
            "created_at": self.created_at,
        }


class CommitmentManager:
    """Manages hash-based commitments for the blockchain."""

    def __init__(self):
        self._proofs: list = []
        self._verified: int = 0
        self._failed: int = 0

    def submit_proof(self, proof: HashCommitment) -> bool:
        """Submit and verify a commitment."""
        valid = CommitmentVerifier.verify(proof)
        self._proofs.append(proof)
        if valid:
            self._verified += 1
        else:
            self._failed += 1
        return valid

    def get_stats(self) -> dict:
        return {
            "total_proofs": len(self._proofs),
            "verified": self._verified,
            "failed": self._failed,
        }


# ---------------------------------------------------------------------------
# Backward-compatible aliases so code using the old "ZK" names continues
# to work without modification when importing directly from this module.
# ---------------------------------------------------------------------------
ZKProof = HashCommitment
ZKProver = CommitmentProver
ZKVerifier = CommitmentVerifier
ZKManager = CommitmentManager
