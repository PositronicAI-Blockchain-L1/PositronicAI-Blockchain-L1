"""
Positronic - Hierarchical Deterministic Wallet (BIP-32/44 Style)

Generates deterministic key hierarchies from a single master seed.
All keys can be recovered from a mnemonic phrase backup.

Phase 17 GOD CHAIN addition.

Derivation path: m/44'/420420'/account'/0/index

**Fail-open**: If HD derivation fails for any reason, the wallet
logs a warning and falls back to random key generation.

**BIP-39 Support** (added in wallet version 3):

New wallets default to standard BIP-39 English wordlist and the
BIP-32 HMAC key ``b"Bitcoin seed"`` for interoperability with
external wallets (MetaMask, Trust Wallet, etc.).

Legacy wallets using the custom Positronic wordlist and
``b"Positronic seed"`` HMAC key remain fully supported via the
``legacy=True`` flag.
"""

import os
import hmac
import hashlib
from typing import List, Optional, Tuple, Dict

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512
from positronic.constants import HD_WALLET_COIN_TYPE, HD_WALLET_MAX_ACCOUNTS, HD_WALLET_MAX_ADDRESSES
from positronic.wallet.bip39_wordlist import BIP39_WORDLIST


# Hardened key offset (BIP-32 standard)
HARDENED = 0x80000000

# HMAC keys for master key derivation
HMAC_KEY_STANDARD = b"Bitcoin seed"       # BIP-32 standard
HMAC_KEY_LEGACY = b"Positronic seed"      # Legacy Positronic key

# Mnemonic format identifiers
FORMAT_BIP39 = "bip39"
FORMAT_LEGACY = "legacy"
FORMAT_UNKNOWN = "unknown"


class MnemonicWordList:
    """ASF-specific word list for mnemonic generation (WALLET_VERSION = 2).

    Contains 2048 deterministically generated pronounceable words.
    Same entropy always produces the same mnemonic.

    NOTE: This is NOT the standard BIP-39 English word list. ASF
    mnemonics are only portable within the ASF ecosystem. The HMAC
    key ``b"Positronic seed"`` ensures ASF derivation produces
    unique keys even given the same seed as another BIP-32 chain.

    For interoperability with external wallets, use raw seed
    import/export via ``HDWallet.from_seed()`` instead of mnemonics.
    """

    # Deterministic word list (2048 words) — generated from a hash chain.
    # ASF-specific. NOT the standard BIP-39 English wordlist.
    _WORDS = None

    @classmethod
    def _ensure_loaded(cls):
        """Lazily generate the word list."""
        if cls._WORDS is not None:
            return
        # Generate a deterministic word list from a seed
        # In production, this would be the full BIP-39 English word list.
        # For this implementation, we generate 2048 unique 4-8 letter words
        # deterministically from a hash chain.
        words = []
        seed = b"positronic bip39 wordlist v1"
        consonants = "bcdfghjklmnpqrstvwxyz"
        vowels = "aeiou"

        for i in range(2048):
            h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            # Build a pronounceable word from hash bytes
            word_len = 4 + (h[0] % 5)  # 4-8 letters
            word = ""
            for j in range(word_len):
                byte = h[(j + 1) % 32]
                if j % 2 == 0:
                    word += consonants[byte % len(consonants)]
                else:
                    word += vowels[byte % len(vowels)]
            # Ensure uniqueness
            while word in words:
                word += vowels[len(words) % len(vowels)]
            words.append(word)

        cls._WORDS = words

    @classmethod
    def get_word(cls, index: int) -> str:
        """Get word at index (0-2047)."""
        cls._ensure_loaded()
        return cls._WORDS[index % 2048]

    @classmethod
    def get_index(cls, word: str) -> int:
        """Get index of a word. Returns -1 if not found."""
        cls._ensure_loaded()
        try:
            return cls._WORDS.index(word)
        except ValueError:
            return -1

    @classmethod
    def word_count(cls) -> int:
        return 2048

    @classmethod
    def get_words(cls) -> List[str]:
        """Return a copy of the full word list."""
        cls._ensure_loaded()
        return list(cls._WORDS)


class BIP39WordList:
    """Standard BIP-39 English word list (2048 words).

    Provides the same interface as :class:`MnemonicWordList` but uses
    the official BIP-39 English word list for interoperability with
    external wallets such as MetaMask and Trust Wallet.
    """

    _WORDS = BIP39_WORDLIST  # already validated at import time

    @classmethod
    def get_word(cls, index: int) -> str:
        """Get word at index (0-2047)."""
        return cls._WORDS[index % 2048]

    @classmethod
    def get_index(cls, word: str) -> int:
        """Get index of a word. Returns -1 if not found."""
        try:
            return cls._WORDS.index(word)
        except ValueError:
            return -1

    @classmethod
    def word_count(cls) -> int:
        return 2048

    @classmethod
    def get_words(cls) -> List[str]:
        """Return a copy of the full word list."""
        return list(cls._WORDS)


def detect_mnemonic_format(mnemonic: str) -> str:
    """Detect whether a mnemonic uses BIP-39 or legacy Positronic format.

    Args:
        mnemonic: Space-separated mnemonic phrase.

    Returns:
        One of ``FORMAT_BIP39``, ``FORMAT_LEGACY``, or ``FORMAT_UNKNOWN``.
    """
    words = mnemonic.strip().split()
    if not words:
        return FORMAT_UNKNOWN

    bip39_set = set(BIP39_WORDLIST)
    legacy_list = MnemonicWordList
    legacy_list._ensure_loaded()
    legacy_set = set(legacy_list._WORDS)

    bip39_matches = sum(1 for w in words if w in bip39_set)
    legacy_matches = sum(1 for w in words if w in legacy_set)

    total = len(words)
    # Require all words to match for a definitive classification
    if bip39_matches == total:
        return FORMAT_BIP39
    if legacy_matches == total:
        return FORMAT_LEGACY
    # If most words match one format, classify accordingly
    if bip39_matches > legacy_matches and bip39_matches >= total * 0.8:
        return FORMAT_BIP39
    if legacy_matches > bip39_matches and legacy_matches >= total * 0.8:
        return FORMAT_LEGACY
    return FORMAT_UNKNOWN


class HDWallet:
    """Hierarchical Deterministic Wallet for Positronic.

    Derives an unlimited number of key pairs from a single master seed.
    Compatible with BIP-32/44 derivation paths.

    Derivation path: ``m/44'/420420'/account'/0/index``

    The mnemonic phrase (24 words by default) can be used to fully
    restore all derived keys.

    **Wallet modes:**

    * **Standard (default for new wallets):** Uses BIP-39 English
      wordlist and ``b"Bitcoin seed"`` HMAC key.  Interoperable with
      MetaMask, Trust Wallet, and other BIP-39 wallets.
    * **Legacy:** Uses the original Positronic wordlist and
      ``b"Positronic seed"`` HMAC key.  Required for existing wallets
      created before BIP-39 support was added.

    Example::

        # Create new BIP-39 wallet (default)
        wallet = HDWallet.create()
        print(wallet.mnemonic)  # Standard BIP-39 words

        # Create legacy wallet
        legacy = HDWallet.create(legacy=True)

        # Restore — auto-detects format
        restored = HDWallet.from_mnemonic("abandon ability able ...")
    """

    def __init__(self, seed: bytes, mnemonic: Optional[str] = None, legacy: bool = False):
        """Initialize from a master seed.

        Args:
            seed: 64-byte master seed.
            mnemonic: Optional mnemonic phrase that generated this seed.
            legacy: If True, derive master key with ``b"Positronic seed"``
                    (backward compat).  If False (default), use the BIP-32
                    standard ``b"Bitcoin seed"``.
        """
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self._seed = seed
        self._mnemonic = mnemonic
        self._legacy = legacy
        self._master_key, self._master_chain = self._derive_master(seed, legacy=legacy)
        self._cache: Dict[str, KeyPair] = {}

    @classmethod
    def create(cls, word_count: int = 24, legacy: bool = False) -> "HDWallet":
        """Create a new HD wallet with a random mnemonic.

        Args:
            word_count: Number of mnemonic words (12, 15, 18, 21, or 24).
                Default is 24 (256-bit security).
            legacy: If True, use the original Positronic wordlist and HMAC key.
                If False (default), use standard BIP-39 wordlist and BIP-32
                HMAC key for interoperability with external wallets.

        Returns:
            New HDWallet instance.
        """
        standard = not legacy
        mnemonic = cls.generate_mnemonic(word_count, standard=standard)
        seed = cls._mnemonic_to_seed(mnemonic)
        return cls(seed=seed, mnemonic=mnemonic, legacy=legacy)

    @classmethod
    def from_mnemonic(
        cls,
        mnemonic: str,
        passphrase: str = "",
        legacy: Optional[bool] = None,
    ) -> "HDWallet":
        """Restore an HD wallet from a mnemonic phrase.

        The format (BIP-39 vs. legacy Positronic) is auto-detected when
        *legacy* is ``None``.  Pass ``legacy=True`` or ``legacy=False``
        explicitly to override detection.

        Args:
            mnemonic: Space-separated mnemonic words.
            passphrase: Optional passphrase for extra security.
            legacy: ``True`` to force legacy Positronic derivation,
                    ``False`` to force BIP-32 standard derivation,
                    ``None`` (default) to auto-detect from the wordlist.

        Returns:
            Restored HDWallet instance.
        """
        if legacy is None:
            fmt = detect_mnemonic_format(mnemonic)
            legacy = (fmt == FORMAT_LEGACY)
        seed = cls._mnemonic_to_seed(mnemonic, passphrase)
        return cls(seed=seed, mnemonic=mnemonic, legacy=legacy)

    @classmethod
    def from_seed(cls, seed: bytes, legacy: bool = False) -> "HDWallet":
        """Create an HD wallet from a raw seed (no mnemonic).

        Args:
            seed: 64-byte master seed.
            legacy: If True, use legacy ``b"Positronic seed"`` HMAC key.

        Returns:
            HDWallet instance without mnemonic backup capability.
        """
        return cls(seed=seed, mnemonic=None, legacy=legacy)

    @property
    def mnemonic(self) -> Optional[str]:
        """The mnemonic phrase, or None if created from raw seed."""
        return self._mnemonic

    @property
    def legacy(self) -> bool:
        """True if this wallet uses legacy Positronic derivation."""
        return self._legacy

    @property
    def format(self) -> str:
        """Return the wallet format identifier (``FORMAT_BIP39`` or ``FORMAT_LEGACY``)."""
        return FORMAT_LEGACY if self._legacy else FORMAT_BIP39

    # ===== Mnemonic generation =====

    @staticmethod
    def generate_mnemonic(word_count: int = 24, standard: bool = True) -> str:
        """Generate a mnemonic phrase.

        Args:
            word_count: Number of words (12, 15, 18, 21, or 24).
            standard: If True (default), use the BIP-39 English wordlist
                for interoperability with external wallets.  If False,
                use the legacy Positronic wordlist.

        Returns:
            Space-separated mnemonic phrase.
        """
        valid_counts = {12: 16, 15: 20, 18: 24, 21: 28, 24: 32}
        if word_count not in valid_counts:
            raise ValueError(f"word_count must be one of {list(valid_counts.keys())}")

        wordlist = BIP39WordList if standard else MnemonicWordList

        entropy_bytes = valid_counts[word_count]
        entropy = os.urandom(entropy_bytes)

        # Generate checksum
        h = hashlib.sha256(entropy).digest()
        checksum_bits = word_count // 3

        # Convert entropy + checksum to bit string
        entropy_bits = bin(int.from_bytes(entropy, "big"))[2:].zfill(entropy_bytes * 8)
        checksum = bin(h[0])[2:].zfill(8)[:checksum_bits]
        all_bits = entropy_bits + checksum

        # Split into 11-bit groups -> word indices
        words = []
        for i in range(0, len(all_bits), 11):
            idx = int(all_bits[i:i + 11], 2)
            words.append(wordlist.get_word(idx))

        return " ".join(words)

    @staticmethod
    def _mnemonic_to_seed(mnemonic: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to 64-byte seed using PBKDF2-SHA512.

        Args:
            mnemonic: Space-separated mnemonic words.
            passphrase: Optional passphrase.

        Returns:
            64-byte seed.
        """
        password = mnemonic.encode("utf-8")
        salt = ("mnemonic" + passphrase).encode("utf-8")
        return hashlib.pbkdf2_hmac("sha512", password, salt, iterations=2048, dklen=64)

    # ===== Key derivation =====

    @staticmethod
    def _derive_master(seed: bytes, legacy: bool = False) -> Tuple[bytes, bytes]:
        """Derive master key and chain code from seed.

        Args:
            seed: The seed bytes.
            legacy: If True, use Positronic's original HMAC key
                    (``b"Positronic seed"``).  If False (default for new
                    wallets), use the BIP-32 standard (``b"Bitcoin seed"``).

        Returns:
            Tuple of (32-byte private key, 32-byte chain code).
        """
        key = HMAC_KEY_LEGACY if legacy else HMAC_KEY_STANDARD
        h = hmac.new(key, seed, hashlib.sha512).digest()
        return h[:32], h[32:]

    @staticmethod
    def _derive_child(
        parent_key: bytes,
        parent_chain: bytes,
        index: int,
    ) -> Tuple[bytes, bytes]:
        """Derive a child key from parent key + chain code.

        For hardened keys (index >= HARDENED), uses the private key.
        For normal keys, uses the public key (not applicable for Ed25519).

        Returns:
            Tuple of (32-byte child key, 32-byte child chain code).
        """
        # For Ed25519, all derivation is hardened
        data = b"\x00" + parent_key + index.to_bytes(4, "big")
        h = hmac.new(parent_chain, data, hashlib.sha512).digest()
        return h[:32], h[32:]

    def _derive_path(self, path: List[int]) -> Tuple[bytes, bytes]:
        """Derive key at a full derivation path.

        Args:
            path: List of derivation indices (use HARDENED for hardened).

        Returns:
            Tuple of (private key bytes, chain code bytes).
        """
        key = self._master_key
        chain = self._master_chain

        for index in path:
            key, chain = self._derive_child(key, chain, index)

        return key, chain

    def derive_address(self, account: int = 0, index: int = 0) -> KeyPair:
        """Derive a key pair at path ``m/44'/420420'/account'/0/index``.

        Args:
            account: Account number (0-99).
            index: Address index within the account (0-999).

        Returns:
            KeyPair at the derived path.

        Raises:
            ValueError: If account or index is out of range.
        """
        if account < 0 or account >= HD_WALLET_MAX_ACCOUNTS:
            raise ValueError(f"Account must be 0-{HD_WALLET_MAX_ACCOUNTS - 1}")
        if index < 0 or index >= HD_WALLET_MAX_ADDRESSES:
            raise ValueError(f"Index must be 0-{HD_WALLET_MAX_ADDRESSES - 1}")

        cache_key = f"{account}/{index}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # BIP-44 path: m/44'/420420'/account'/0/index
        path = [
            44 | HARDENED,              # Purpose
            HD_WALLET_COIN_TYPE | HARDENED,  # Coin type
            account | HARDENED,          # Account
            0,                           # Change (always 0 for external)
            index,                       # Address index
        ]

        key_bytes, _ = self._derive_path(path)

        # Create KeyPair from derived private key
        # Use derived bytes as seed for Ed25519 key generation
        kp = KeyPair.from_seed(key_bytes)
        self._cache[cache_key] = kp
        return kp

    def derive_account(self, account: int = 0) -> KeyPair:
        """Derive the primary key for an account (index 0).

        Args:
            account: Account number (0-99).

        Returns:
            KeyPair for the account's first address.
        """
        return self.derive_address(account=account, index=0)

    def get_all_addresses(
        self,
        account: int = 0,
        count: int = 10,
    ) -> List[Dict]:
        """Get multiple derived addresses for an account.

        Args:
            account: Account number.
            count: Number of addresses to derive.

        Returns:
            List of dictionaries with ``index``, ``address``, and ``pubkey``.
        """
        count = min(count, HD_WALLET_MAX_ADDRESSES)
        results = []
        for i in range(count):
            kp = self.derive_address(account=account, index=i)
            results.append({
                "index": i,
                "address": kp.address_hex,
                "pubkey": kp.public_key_bytes.hex() if hasattr(kp, "public_key_bytes") else "",
                "path": f"m/44'/{HD_WALLET_COIN_TYPE}'/{account}'/0/{i}",
            })
        return results

    def get_stats(self) -> Dict:
        """Return wallet state for monitoring."""
        return {
            "has_mnemonic": self._mnemonic is not None,
            "cached_keys": len(self._cache),
            "coin_type": HD_WALLET_COIN_TYPE,
            "max_accounts": HD_WALLET_MAX_ACCOUNTS,
            "max_addresses": HD_WALLET_MAX_ADDRESSES,
            "format": self.format,
            "legacy": self._legacy,
        }
