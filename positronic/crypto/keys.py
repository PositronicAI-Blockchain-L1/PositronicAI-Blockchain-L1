"""
Positronic - Ed25519 Key Management
Uses the cryptography library for Ed25519 signing and verification.
"""

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

from positronic.crypto.hashing import sha512, blake2b_160


class KeyPair:
    """
    Ed25519 key pair for signing and verification.
    Provides address derivation compatible with EVM-style 20-byte addresses.
    """

    def __init__(self, private_key: Ed25519PrivateKey = None):
        if private_key is None:
            self._private_key = Ed25519PrivateKey.generate()
        else:
            self._private_key = private_key
        self._public_key = self._private_key.public_key()

    @property
    def private_key(self) -> Ed25519PrivateKey:
        return self._private_key

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self._public_key

    @property
    def private_key_bytes(self) -> bytes:
        """Raw 32-byte private key."""
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_bytes(self) -> bytes:
        """Raw 32-byte public key."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def address(self) -> bytes:
        """
        20-byte address derived from public key.
        address = Blake2b-160(SHA-512(public_key))
        """
        return blake2b_160(sha512(self.public_key_bytes))

    @property
    def address_hex(self) -> str:
        """Address as 0x-prefixed hex string (EVM-compatible format)."""
        return "0x" + self.address.hex()

    def sign(self, message: bytes) -> bytes:
        """Sign a message with Ed25519. Returns 64-byte signature."""
        return self._private_key.sign(message)

    @staticmethod
    def verify(public_key_bytes: bytes, signature: bytes, message: bytes) -> bool:
        """Verify an Ed25519 signature against a public key."""
        try:
            pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            pub.verify(signature, message)
            return True
        except InvalidSignature:
            return False
        except (ValueError, TypeError):
            # Malformed key/signature bytes — not a valid crypto input
            return False

    def to_bytes(self) -> bytes:
        """Serialize the key pair (private key only, 32 bytes)."""
        return self.private_key_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "KeyPair":
        """Deserialize a key pair from 32-byte private key."""
        private_key = Ed25519PrivateKey.from_private_bytes(data)
        return cls(private_key)

    @classmethod
    def from_private_hex(cls, hex_str: str) -> "KeyPair":
        """Create key pair from hex-encoded private key."""
        data = bytes.fromhex(hex_str.removeprefix("0x"))
        return cls.from_bytes(data)

    @classmethod
    def from_seed(cls, seed: bytes) -> "KeyPair":
        """Create key pair from a seed (32+ bytes).

        Uses the first 32 bytes of the seed as the Ed25519 private key.
        If the seed is shorter, it is hashed with SHA-512 first.

        Phase 17: Used by HD wallet for deterministic key derivation.

        Args:
            seed: Seed bytes (at least 32 bytes recommended).

        Returns:
            KeyPair derived from the seed.
        """
        if len(seed) < 32:
            from positronic.crypto.hashing import sha512 as hash_fn
            seed = hash_fn(seed)
        key_bytes = seed[:32]
        return cls.from_bytes(key_bytes)

    def __repr__(self) -> str:
        return f"KeyPair(address={self.address_hex})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyPair):
            return False
        return self.public_key_bytes == other.public_key_bytes

    def __hash__(self) -> int:
        return hash(self.public_key_bytes)
