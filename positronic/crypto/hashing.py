"""
Positronic - Hashing Utilities
Primary: SHA-512 (quantum-resistant, faster on 64-bit CPUs)
Secondary: Blake2b for performance-critical paths

Automatically uses native C extension (_native) when available
for 2-4x speedup. Falls back to pure Python hashlib otherwise.
"""

import hashlib
import struct
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── Try native C extension first ─────────────────────────────
_NATIVE_AVAILABLE = False
try:
    from positronic.crypto._native import (
        sha512 as _native_sha512,
        blake2b_160 as _native_blake2b_160,
        double_hash as _native_double_hash,
        hash_pair as _native_hash_pair,
        merkle_root as _native_merkle_root,
    )
    _NATIVE_AVAILABLE = True
    logger.debug("Native C crypto extension loaded — hot paths accelerated")
except ImportError:
    logger.debug("Native C crypto extension not available — using pure Python")

# ── Pure Python implementations (fallback) ───────────────────

def _py_sha512(data: bytes) -> bytes:
    """SHA-512 hash (64 bytes). Pure Python implementation."""
    return hashlib.sha512(data).digest()


def _py_blake2b_160(data: bytes) -> bytes:
    """Blake2b-160 hash (20 bytes). Pure Python implementation."""
    return hashlib.blake2b(data, digest_size=20).digest()


def _py_double_hash(data: bytes) -> bytes:
    """SHA-512(SHA-512(data)). Pure Python implementation."""
    return _py_sha512(_py_sha512(data))


def _py_hash_pair(left: bytes, right: bytes) -> bytes:
    """Hash two values together for Merkle tree nodes. Pure Python."""
    return _py_sha512(left + right)


def _py_merkle_root(items: List[bytes]) -> bytes:
    """Compute Merkle root. Pure Python implementation."""
    if not items:
        return b"\x00" * 64
    if len(items) == 1:
        return _py_sha512(items[0])
    level = list(items)
    if len(level) % 2 != 0:
        level.append(level[-1])
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            next_level.append(_py_hash_pair(level[i], level[i + 1]))
        level = next_level
        if len(level) > 1 and len(level) % 2 != 0:
            level.append(level[-1])
    return level[0]


# ── Public API (auto-selects native or Python) ───────────────

if _NATIVE_AVAILABLE:
    sha512 = _native_sha512
    blake2b_160 = _native_blake2b_160
    double_hash = _native_double_hash
    hash_pair = _native_hash_pair
    merkle_root = _native_merkle_root
else:
    sha512 = _py_sha512
    blake2b_160 = _py_blake2b_160
    double_hash = _py_double_hash
    hash_pair = _py_hash_pair
    merkle_root = _py_merkle_root


def sha512_hex(data: bytes) -> str:
    """SHA-512 hash as hex string."""
    return sha512(data).hex()


def sha256(data: bytes) -> bytes:
    """SHA-256 hash (32 bytes). Used for EVM compatibility where needed."""
    return hashlib.sha256(data).digest()


def blake2b_256(data: bytes) -> bytes:
    """Blake2b-256 hash (32 bytes). Fast hash for internal operations."""
    return hashlib.blake2b(data, digest_size=32).digest()


def keccak256(data: bytes) -> bytes:
    """
    Keccak-256 hash (32 bytes). For EVM/Solidity compatibility.

    Uses pysha3 or pycryptodome for true Keccak-256 (pre-NIST padding).
    Raises ImportError if neither library is available, because SHA3-256
    (NIST variant) uses different padding and would break EVM compatibility.
    """
    # Try pysha3 first (provides true Keccak-256)
    try:
        import sha3
        return sha3.keccak_256(data).digest()
    except ImportError:
        pass
    # Try pycryptodome (provides true Keccak-256)
    try:
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256, data=data)
        return k.digest()
    except ImportError:
        pass
    # CRITICAL: Do NOT fall back to SHA3-256 (NIST variant).
    # SHA3-256 uses different domain separation padding than Keccak-256,
    # producing different hashes. This would break EVM compatibility,
    # cross-chain bridges, and MetaMask address derivation.
    raise ImportError(
        "True Keccak-256 requires 'pysha3' or 'pycryptodome' library. "
        "Install one with: pip install pycryptodome"
    )


def hash_to_int(data: bytes) -> int:
    """
    Convert hash bytes to a large integer for random selection.

    Uses 32 bytes (256 bits) to produce values up to ~1.15 × 10⁷⁷,
    which safely exceeds typical total-stake values (denominated in
    10¹⁸ base units).  The previous 8-byte version (max ~1.8 × 10¹⁹)
    was smaller than even MIN_STAKE (32 × 10¹⁸ = 3.2 × 10¹⁹), which
    made stake-weighted proposer selection always pick the first
    validator in the cumulative list.
    """
    return int.from_bytes(data[:32], "big")


def target_hash(data: bytes, nonce: int) -> bytes:
    """Hash data with a nonce, used for various proof mechanisms."""
    return sha512(data + struct.pack(">Q", nonce))
