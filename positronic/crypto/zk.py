"""
Backward compatibility shim.

The actual implementation has been moved to commitments.py
to accurately reflect that this is a hash-based commitment scheme,
not a true zero-knowledge proof system.

All original names (ZKProof, ZKProver, ZKVerifier, ZKManager, etc.)
are re-exported here so existing imports continue to work.
"""
from positronic.crypto.commitments import *  # noqa: F401,F403

# Backward-compatible aliases (explicit re-exports for type checkers)
from positronic.crypto.commitments import HashCommitment as ZKProof  # noqa: F401
from positronic.crypto.commitments import CommitmentProver as ZKProver  # noqa: F401
from positronic.crypto.commitments import CommitmentVerifier as ZKVerifier  # noqa: F401
from positronic.crypto.commitments import CommitmentManager as ZKManager  # noqa: F401
from positronic.crypto.commitments import (  # noqa: F401
    MerkleTree,
    ConfidentialEvidence,
    _hash,
    _int_to_bytes,
    _generator_point,
    _verification_tag,
    _build_proof,
    _build_commitment,
)
