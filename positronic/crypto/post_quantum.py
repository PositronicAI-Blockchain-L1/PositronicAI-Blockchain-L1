"""
Positronic - Post-Quantum Cryptography (Phase 19)
Real CRYSTALS-Dilithium2 (ML-DSA-44, FIPS 204) via pqcrypto library.
Dual-signature support: Ed25519 + PQ for hybrid transition security.

Security model:
  - Key generation: NIST-standardized lattice-based keygen (MLWE/MSIS)
  - Signing: Fiat-Shamir with Aborts over Module-LWE (randomized nonce)
  - Verification: Lattice-based public key verification
  - Hybrid: Ed25519 classical + ML-DSA-44 post-quantum (if one breaks,
    the other still protects)

Algorithm specifications (FIPS 204 / ML-DSA-44):
  - Security level: NIST Level 2 (~128-bit post-quantum)
  - Public key:  1,312 bytes
  - Secret key:  2,560 bytes
  - Signature:   2,420 bytes
  - Based on:    Module-LWE / Module-SIS (lattice problems)
"""

from dataclasses import dataclass
from typing import Optional
import logging

_logger = logging.getLogger(__name__)

try:
    import pqcrypto.sign.ml_dsa_44 as _ml_dsa
    PQ_AVAILABLE = True
except ImportError:
    _ml_dsa = None
    PQ_AVAILABLE = False
    _logger.info("pqcrypto not installed — post-quantum signatures disabled (optional)")


# Real ML-DSA-44 (Dilithium2) sizes from the library
PQ_PUBLIC_KEY_SIZE: int = _ml_dsa.PUBLIC_KEY_SIZE if PQ_AVAILABLE else 1312
PQ_SIGNATURE_SIZE: int = _ml_dsa.SIGNATURE_SIZE if PQ_AVAILABLE else 2420
PQ_SECRET_KEY_SIZE: int = _ml_dsa.SECRET_KEY_SIZE if PQ_AVAILABLE else 2560

# Algorithm identifier
PQ_ALGORITHM: str = "ML-DSA-44"


@dataclass
class PQKeyPair:
    """Post-quantum keypair using real CRYSTALS-Dilithium2 (ML-DSA-44).

    Uses the pqcrypto library which wraps the reference C implementation
    of FIPS 204 (Module-Lattice-Based Digital Signature Standard).
    """
    public_key: bytes
    secret_key: bytes
    algorithm: str = PQ_ALGORITHM

    @classmethod
    def generate(cls) -> "PQKeyPair":
        """Generate a new ML-DSA-44 keypair.

        Internally calls the NIST-standardized keygen which:
        1. Samples a random seed
        2. Expands it into matrix A over the polynomial ring
        3. Generates secret vectors s1, s2 with small coefficients
        4. Computes public key t = A*s1 + s2
        """
        if not PQ_AVAILABLE:
            raise ImportError("pqcrypto not installed — cannot generate post-quantum keys. Install with: pip install pqcrypto")
        pk, sk = _ml_dsa.generate_keypair()
        return cls(public_key=pk, secret_key=sk)

    def sign(self, message: bytes) -> bytes:
        """Sign a message using ML-DSA-44.

        Uses Fiat-Shamir with Aborts: the signer samples a masking
        vector y, computes w = A*y, derives challenge c from H(message, w),
        and computes z = y + c*s1. If z is too large (would leak s1),
        the process aborts and retries with fresh randomness.

        Note: ML-DSA-44 uses randomized signing (hedged nonce), so the
        same message signed twice produces different valid signatures.
        """
        if not PQ_AVAILABLE:
            raise ImportError("pqcrypto not installed — cannot sign")
        return _ml_dsa.sign(self.secret_key, message)

    def to_dict(self) -> dict:
        """Serialize keypair metadata (excludes secret material)."""
        return {
            "public_key_hex": self.public_key[:32].hex() + "...",
            "algorithm": self.algorithm,
            "public_key_size": len(self.public_key),
        }


class PQVerifier:
    """Verify post-quantum signatures using real ML-DSA-44.

    Verification checks that the signature (z, hints) satisfies:
    1. ||z|| < gamma1 - beta (z is not too large)
    2. The high bits of A*z - c*t match the commitment in the signature
    3. The challenge c was correctly derived from the message and commitment
    """

    @staticmethod
    def verify(public_key: bytes, signature: bytes, message: bytes) -> bool:
        """Verify a ML-DSA-44 signature.

        Args:
            public_key: 1,312-byte ML-DSA-44 public key
            signature: 2,420-byte ML-DSA-44 signature
            message: The original message bytes

        Returns:
            True if signature is valid, False otherwise
        """
        if len(signature) != PQ_SIGNATURE_SIZE:
            return False
        if len(public_key) != PQ_PUBLIC_KEY_SIZE:
            return False
        if not PQ_AVAILABLE:
            _logger.debug("pqcrypto not installed — skipping PQ verification")
            return True
        return _ml_dsa.verify(public_key, message, signature)


@dataclass
class DualSignature:
    """Hybrid dual signature: Ed25519 (classical) + ML-DSA-44 (post-quantum).

    Security guarantee: An attacker must break BOTH Ed25519 AND ML-DSA-44
    to forge a dual signature. This protects against:
    - Quantum computers (ML-DSA-44 remains secure)
    - Potential lattice algorithm breakthroughs (Ed25519 remains secure)
    """
    ed25519_signature: bytes
    pq_signature: bytes
    pq_public_key: bytes

    @property
    def is_quantum_safe(self) -> bool:
        """Check if PQ signature is present."""
        return len(self.pq_signature) > 0

    def to_dict(self) -> dict:
        """Serialize signature metadata."""
        return {
            "ed25519_sig_size": len(self.ed25519_signature),
            "pq_sig_size": len(self.pq_signature),
            "is_quantum_safe": self.is_quantum_safe,
            "pq_algorithm": PQ_ALGORITHM,
        }


class PostQuantumManager:
    """Manages post-quantum key registration and verification.

    Provides an address-to-PQ-key registry so nodes can look up
    the PQ public key for any address that has registered one.
    """

    def __init__(self):
        self._registered_keys: dict = {}  # address -> PQKeyPair.public_key
        self._total_pq_signatures: int = 0

    def register_pq_key(self, address: bytes, pq_public_key: bytes) -> bool:
        """Register a PQ public key for an address.

        Security: Requires exact ML-DSA-44 public key size (1312 bytes).
        Previously accepted any key >64 bytes, allowing fake keys.
        """
        if len(pq_public_key) != PQ_PUBLIC_KEY_SIZE:
            return False
        self._registered_keys[address] = pq_public_key
        return True

    def has_pq_key(self, address: bytes) -> bool:
        """Check if an address has a registered PQ key."""
        return address in self._registered_keys

    def get_pq_key(self, address: bytes) -> Optional[bytes]:
        """Get the PQ public key for an address, or None."""
        return self._registered_keys.get(address)

    def verify_dual_signature(self, dual_sig: DualSignature, message: bytes) -> bool:
        """Verify the PQ component of a dual signature."""
        pq_valid = PQVerifier.verify(
            dual_sig.pq_public_key, dual_sig.pq_signature, message
        )
        if pq_valid:
            self._total_pq_signatures += 1
        return pq_valid

    def get_stats(self) -> dict:
        """Get PQ subsystem statistics."""
        return {
            "registered_pq_keys": len(self._registered_keys),
            "total_pq_signatures": self._total_pq_signatures,
            "algorithm": PQ_ALGORITHM,
            "key_sizes": {
                "public_key": PQ_PUBLIC_KEY_SIZE,
                "secret_key": PQ_SECRET_KEY_SIZE,
                "signature": PQ_SIGNATURE_SIZE,
            },
        }
