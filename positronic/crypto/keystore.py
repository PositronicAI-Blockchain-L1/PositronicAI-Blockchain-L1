"""
Positronic - Encrypted Keystore (Web3 Keystore V3 Compatible)
AES-256-GCM encryption with PBKDF2 key derivation.
"""

import json
import os
import hashlib

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from positronic.crypto.keys import KeyPair

KEYSTORE_VERSION = 3
KDF_ITERATIONS = 100_000
SALT_SIZE = 32
NONCE_SIZE = 12  # AES-GCM standard
KEY_SIZE = 32    # AES-256


class Keystore:
    """Encrypt/decrypt Ed25519 keypairs with password-based encryption."""

    @staticmethod
    def encrypt(keypair: KeyPair, password: str) -> dict:
        """Encrypt a KeyPair into a keystore dict."""
        salt = os.urandom(SALT_SIZE)
        derived_key = Keystore._derive_key(password, salt)

        nonce = os.urandom(NONCE_SIZE)
        aesgcm = AESGCM(derived_key)
        plaintext = keypair.private_key_bytes
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        mac = hashlib.sha256(derived_key + ciphertext).hexdigest()

        return {
            "version": KEYSTORE_VERSION,
            "address": keypair.address.hex(),
            "crypto": {
                "cipher": "aes-256-gcm",
                "ciphertext": ciphertext.hex(),
                "cipherparams": {"nonce": nonce.hex()},
                "kdf": "pbkdf2",
                "kdfparams": {
                    "iterations": KDF_ITERATIONS,
                    "salt": salt.hex(),
                    "dklen": KEY_SIZE,
                    "prf": "hmac-sha256",
                },
                "mac": mac,
            },
        }

    @staticmethod
    def decrypt(keystore: dict, password: str) -> KeyPair:
        """Decrypt a keystore dict back to a KeyPair."""
        crypto = keystore["crypto"]
        kdfparams = crypto["kdfparams"]

        salt = bytes.fromhex(kdfparams["salt"])
        iterations = kdfparams["iterations"]
        derived_key = Keystore._derive_key(password, salt, iterations)

        ciphertext = bytes.fromhex(crypto["ciphertext"])

        expected_mac = hashlib.sha256(derived_key + ciphertext).hexdigest()
        if expected_mac != crypto["mac"]:
            raise ValueError("Decryption failed: wrong password or corrupted keystore")

        nonce = bytes.fromhex(crypto["cipherparams"]["nonce"])
        aesgcm = AESGCM(derived_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        return KeyPair.from_bytes(plaintext)

    @staticmethod
    def save(keypair: KeyPair, password: str, path: str) -> dict:
        """Encrypt and save to a JSON file."""
        keystore = Keystore.encrypt(keypair, password)
        with open(path, "w") as f:
            json.dump(keystore, f, indent=2)
        return keystore

    @staticmethod
    def load(path: str, password: str) -> KeyPair:
        """Load and decrypt from a JSON file."""
        with open(path, "r") as f:
            keystore = json.load(f)
        return Keystore.decrypt(keystore, password)

    @staticmethod
    def _derive_key(password: str, salt: bytes, iterations: int = KDF_ITERATIONS) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(password.encode("utf-8"))
