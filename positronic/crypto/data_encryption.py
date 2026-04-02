"""
Positronic - Data File Encryption

Encrypts/decrypts local data files (keypairs, settings, etc.) using
machine-specific AES-256-GCM encryption.  Reuses the same cryptographic
primitives as the database encryption layer in ``storage.database``.

Files are encrypted in-place with a ``POSITRONIC_ENC_V1`` magic header so
they can be detected as already-encrypted on subsequent reads.  Plaintext
files from older versions are transparently migrated on first access.
"""

import hashlib
import hmac
import logging
import os
import stat
import json
from typing import Optional

logger = logging.getLogger(__name__)

# ── Shared constants (same as database.py) ─────────────────────────
_ENC_MAGIC = b"POSITRONIC_ENC_V1"  # 17 bytes header
_ENC_SALT_LEN = 32
_ENC_NONCE_LEN = 12

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False


class DataEncryptor:
    """Encrypts/decrypts local data files using a machine-specific key.

    The key is derived from machine identity (hostname + username + data_dir)
    combined with a per-installation random salt, through PBKDF2 with 100K
    iterations.  Files encrypted on one machine cannot be decrypted on another.
    """

    ITERATIONS = 100_000
    KEY_LENGTH = 32  # AES-256

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._key: Optional[bytes] = None  # Lazy — derived on first use

    # ── Key derivation ─────────────────────────────────────────────

    def _get_key(self) -> bytes:
        """Lazily derive and cache the machine-specific encryption key."""
        if self._key is not None:
            return self._key

        # Load or create per-installation salt
        salt_path = os.path.join(self.data_dir, ".file_enc_salt")
        if os.path.exists(salt_path):
            with open(salt_path, "rb") as f:
                salt = f.read(32)
        else:
            salt = os.urandom(32)
            os.makedirs(self.data_dir, exist_ok=True)
            with open(salt_path, "wb") as f:
                f.write(salt)
            _restrict_permissions(salt_path)

        # Machine-specific identity
        import socket
        import getpass
        machine_id = (
            f"{socket.gethostname()}:{getpass.getuser()}"
            f":{os.path.abspath(self.data_dir)}"
        )

        self._key = hashlib.pbkdf2_hmac(
            "sha256", machine_id.encode(), b"positronic-file-v1" + salt,
            self.ITERATIONS,
        )
        return self._key

    # ── Low-level encrypt / decrypt ────────────────────────────────

    def encrypt_bytes(self, plaintext: bytes) -> bytes:
        """Encrypt raw bytes, returning ``MAGIC + salt + nonce + ciphertext``."""
        key = self._get_key()
        salt = os.urandom(_ENC_SALT_LEN)
        nonce = os.urandom(_ENC_NONCE_LEN)
        enc_key = hashlib.pbkdf2_hmac("sha256", key, salt, 1)[:self.KEY_LENGTH]

        if _HAS_CRYPTO:
            aesgcm = AESGCM(enc_key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, _ENC_MAGIC)
        else:
            ciphertext = _xor_encrypt(enc_key, nonce, plaintext, _ENC_MAGIC)

        return _ENC_MAGIC + salt + nonce + ciphertext

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Decrypt bytes previously encrypted by :meth:`encrypt_bytes`.

        If *data* does not start with the magic header it is returned as-is
        (backward compatibility with unencrypted files).
        """
        if data[:len(_ENC_MAGIC)] != _ENC_MAGIC:
            return data  # Not encrypted — pass through

        key = self._get_key()
        offset = len(_ENC_MAGIC)
        salt = data[offset:offset + _ENC_SALT_LEN]
        offset += _ENC_SALT_LEN
        nonce = data[offset:offset + _ENC_NONCE_LEN]
        offset += _ENC_NONCE_LEN
        ciphertext = data[offset:]

        enc_key = hashlib.pbkdf2_hmac("sha256", key, salt, 1)[:self.KEY_LENGTH]

        if _HAS_CRYPTO:
            aesgcm = AESGCM(enc_key)
            return aesgcm.decrypt(nonce, ciphertext, _ENC_MAGIC)
        else:
            return _xor_decrypt(enc_key, nonce, ciphertext, _ENC_MAGIC)

    # ── File-level helpers ─────────────────────────────────────────

    def encrypt_file(self, filepath: str) -> None:
        """Encrypt a file in-place.  Skips if already encrypted or missing."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            data = f.read()
        if not data or data[:len(_ENC_MAGIC)] == _ENC_MAGIC:
            return  # Empty or already encrypted
        encrypted = self.encrypt_bytes(data)
        tmp_path = filepath + ".enc.tmp"
        with open(tmp_path, "wb") as f:
            f.write(encrypted)
        os.replace(tmp_path, filepath)
        _restrict_permissions(filepath)
        logger.info("Encrypted file: %s", filepath)

    def decrypt_file(self, filepath: str) -> bytes:
        """Read and decrypt a file, returning the plaintext bytes.

        If the file is not encrypted, returns the raw contents (backward compat).
        If the file does not exist, returns ``b''``.
        """
        if not os.path.exists(filepath):
            return b""
        with open(filepath, "rb") as f:
            data = f.read()
        try:
            return self.decrypt_bytes(data)
        except Exception as exc:
            logger.warning("Failed to decrypt %s: %s — treating as new", filepath, exc)
            return b""

    def encrypt_and_write(self, filepath: str, plaintext: bytes) -> None:
        """Encrypt *plaintext* and write atomically to *filepath*."""
        encrypted = self.encrypt_bytes(plaintext)
        tmp_path = filepath + ".enc.tmp"
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(encrypted)
        os.replace(tmp_path, filepath)
        _restrict_permissions(filepath)

    # ── JSON settings helpers ──────────────────────────────────────

    def load_json(self, filepath: str) -> dict:
        """Load and decrypt a JSON settings file.

        Returns an empty dict if the file is missing, empty, or corrupt.
        """
        data = self.decrypt_file(filepath)
        if not data:
            return {}
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Corrupt settings file %s: %s — returning empty", filepath, exc)
            return {}

    def save_json(self, filepath: str, obj: dict) -> None:
        """Encrypt and save a JSON-serializable dict to *filepath*."""
        plaintext = json.dumps(obj, indent=2).encode("utf-8")
        self.encrypt_and_write(filepath, plaintext)


# ── File permissions helper ────────────────────────────────────────

def _restrict_permissions(filepath: str) -> None:
    """Set file permissions to owner-only read/write (0600) where supported."""
    try:
        if os.name == "nt":
            # Windows: use icacls to restrict to current user only
            import subprocess
            username = os.environ.get("USERNAME", "")
            if username:
                subprocess.run(
                    ["icacls", filepath, "/inheritance:r",
                     "/grant:r", f"{username}:(R,W)"],
                    capture_output=True, timeout=5,
                )
        else:
            os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as exc:
        logger.debug("Could not restrict permissions on %s: %s", filepath, exc)


# ── Fallback XOR cipher (when ``cryptography`` is not installed) ───

def _xor_keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """HMAC-SHA512 counter-mode keystream."""
    stream = b""
    counter = 0
    while len(stream) < length:
        block = hmac.new(
            key, nonce + counter.to_bytes(8, "big"), hashlib.sha512
        ).digest()
        stream += block
        counter += 1
    return stream[:length]


def _xor_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """XOR encrypt with HMAC-SHA256 authentication tag."""
    ks = _xor_keystream(key, nonce, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, ks))
    tag = hmac.new(key, aad + nonce + ciphertext, hashlib.sha256).digest()
    return ciphertext + tag


def _xor_decrypt(key: bytes, nonce: bytes, data: bytes, aad: bytes) -> bytes:
    """XOR decrypt with HMAC-SHA256 verification."""
    tag_len = 32
    ciphertext = data[:-tag_len]
    tag = data[-tag_len:]
    expected = hmac.new(key, aad + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Decryption failed: invalid key or corrupted file")
    ks = _xor_keystream(key, nonce, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, ks))
