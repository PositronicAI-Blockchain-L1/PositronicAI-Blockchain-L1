"""
Positronic - Encrypted Keystore
File-based encrypted key storage using AES-256-GCM.
"""

import logging
import os
import json
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from positronic.crypto.keys import KeyPair


class KeyStore:
    """
    Encrypted keystore for Positronic private keys.
    Uses PBKDF2 for key derivation and AES-256-GCM for encryption.
    """

    ITERATIONS = 100_000
    KEY_LENGTH = 32  # AES-256
    SALT_LENGTH = 32
    NONCE_LENGTH = 12

    @staticmethod
    def save_key(keypair: KeyPair, password: str, filepath: str):
        """Encrypt and save a key pair to a file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        salt = os.urandom(KeyStore.SALT_LENGTH)
        nonce = os.urandom(KeyStore.NONCE_LENGTH)

        # Derive encryption key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=KeyStore.KEY_LENGTH,
            salt=salt,
            iterations=KeyStore.ITERATIONS,
        )
        enc_key = kdf.derive(password.encode("utf-8"))

        # Encrypt private key
        aesgcm = AESGCM(enc_key)
        encrypted = aesgcm.encrypt(nonce, keypair.private_key_bytes, None)

        # Save to file
        keystore_data = {
            "version": 1,
            "address": keypair.address_hex,
            "crypto": {
                "cipher": "aes-256-gcm",
                "ciphertext": encrypted.hex(),
                "salt": salt.hex(),
                "nonce": nonce.hex(),
                "kdf": "pbkdf2-sha512",
                "iterations": KeyStore.ITERATIONS,
            },
        }

        with open(filepath, "w") as f:
            json.dump(keystore_data, f, indent=2)

    @staticmethod
    def load_key(filepath: str, password: str) -> KeyPair:
        """Load and decrypt a key pair from a file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        crypto = data["crypto"]
        salt = bytes.fromhex(crypto["salt"])
        nonce = bytes.fromhex(crypto["nonce"])
        ciphertext = bytes.fromhex(crypto["ciphertext"])
        iterations = crypto.get("iterations", KeyStore.ITERATIONS)

        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=KeyStore.KEY_LENGTH,
            salt=salt,
            iterations=iterations,
        )
        enc_key = kdf.derive(password.encode("utf-8"))

        # Decrypt
        aesgcm = AESGCM(enc_key)
        private_key_bytes = aesgcm.decrypt(nonce, ciphertext, None)

        return KeyPair.from_bytes(private_key_bytes)

    @staticmethod
    def list_keys(keystore_dir: str) -> list:
        """List all key files in the keystore directory."""
        if not os.path.exists(keystore_dir):
            return []

        keys = []
        for filename in os.listdir(keystore_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(keystore_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    keys.append({
                        "file": filename,
                        "address": data.get("address", "unknown"),
                    })
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        "Skipping keystore file %s: %s", filename, e
                    )
        return keys
