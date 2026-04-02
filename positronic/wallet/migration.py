"""
Wallet Migration Utility

Helps migrate legacy Positronic wallets to BIP-39 standard format.

Legacy wallets use:
  - A custom Positronic wordlist (hash-chain generated)
  - HMAC key ``b"Positronic seed"`` for master key derivation

BIP-39 standard wallets use:
  - The standard BIP-39 English wordlist (2048 words)
  - HMAC key ``b"Bitcoin seed"`` for master key derivation (BIP-32)

Migration does NOT attempt to convert a legacy mnemonic into an
equivalent BIP-39 mnemonic (this is impossible because the wordlists
differ).  Instead, migration creates a **new** BIP-39 wallet and
provides the user with both sets of keys so they can transfer funds.
"""

from typing import Dict, List, Optional

from positronic.wallet.hd_wallet import (
    HDWallet,
    BIP39WordList,
    MnemonicWordList,
    detect_mnemonic_format,
    FORMAT_BIP39,
    FORMAT_LEGACY,
    FORMAT_UNKNOWN,
)


class WalletMigrator:
    """Migrates legacy Positronic wallets to BIP-39 compatible format.

    Usage::

        migrator = WalletMigrator()

        # Check format
        fmt = migrator.detect_format("boji neko ...")
        assert fmt == "legacy"

        # Migrate
        result = migrator.migrate("boji neko ...", password="")
        print(result["new_mnemonic"])   # Save this!
        print(result["legacy_addresses"])
        print(result["new_addresses"])
    """

    def detect_format(self, mnemonic: str) -> str:
        """Detect if a mnemonic is BIP-39 or legacy Positronic format.

        Args:
            mnemonic: Space-separated mnemonic phrase.

        Returns:
            ``"bip39"``, ``"legacy"``, or ``"unknown"``.
        """
        return detect_mnemonic_format(mnemonic)

    def is_valid_mnemonic(self, mnemonic: str, expected_format: Optional[str] = None) -> bool:
        """Validate a mnemonic phrase against the appropriate wordlist.

        Args:
            mnemonic: Space-separated mnemonic phrase.
            expected_format: If provided, validate against this specific format
                (``"bip39"`` or ``"legacy"``).  Otherwise auto-detect.

        Returns:
            True if all words are in the expected wordlist and the word
            count is valid (12, 15, 18, 21, or 24).
        """
        words = mnemonic.strip().split()
        valid_counts = {12, 15, 18, 21, 24}
        if len(words) not in valid_counts:
            return False

        fmt = expected_format or self.detect_format(mnemonic)
        if fmt == FORMAT_BIP39:
            wordset = set(BIP39WordList.get_words())
        elif fmt == FORMAT_LEGACY:
            wordset = set(MnemonicWordList.get_words())
        else:
            return False

        return all(w in wordset for w in words)

    def can_migrate(self, legacy_mnemonic: str) -> bool:
        """Check if a legacy wallet can be migrated.

        A wallet can be migrated if:
        1. The mnemonic is detected as legacy format.
        2. All words are valid in the legacy wordlist.
        3. The word count is valid.

        Args:
            legacy_mnemonic: Space-separated legacy mnemonic phrase.

        Returns:
            True if migration is possible.
        """
        fmt = self.detect_format(legacy_mnemonic)
        if fmt != FORMAT_LEGACY:
            return False
        return self.is_valid_mnemonic(legacy_mnemonic, expected_format=FORMAT_LEGACY)

    def migrate(
        self,
        legacy_mnemonic: str,
        password: str = "",
        new_word_count: int = 24,
        num_addresses: int = 5,
    ) -> Dict:
        """Generate a new BIP-39 wallet and return migration info.

        This does NOT transfer funds.  It creates a new BIP-39 wallet
        and returns both the legacy and new addresses so the user (or
        an automated tool) can initiate the transfers.

        Args:
            legacy_mnemonic: The existing legacy mnemonic phrase.
            password: Passphrase used with the legacy mnemonic (if any).
            new_word_count: Word count for the new BIP-39 mnemonic (default 24).
            num_addresses: Number of addresses to derive from each wallet
                for the migration mapping.

        Returns:
            Dictionary containing:
                - ``legacy_format``: Detected format of the input mnemonic.
                - ``legacy_mnemonic``: The original legacy mnemonic (echo).
                - ``new_mnemonic``: The newly generated BIP-39 mnemonic.
                - ``new_format``: Always ``"bip39"``.
                - ``legacy_addresses``: List of legacy derived addresses.
                - ``new_addresses``: List of new BIP-39 derived addresses.
                - ``address_mapping``: Paired list of (legacy, new) addresses
                  for fund transfer.

        Raises:
            ValueError: If the mnemonic is not in legacy format or is invalid.
        """
        fmt = self.detect_format(legacy_mnemonic)
        if fmt == FORMAT_BIP39:
            raise ValueError(
                "This mnemonic is already in BIP-39 format. No migration needed."
            )
        if fmt == FORMAT_UNKNOWN:
            raise ValueError(
                "Cannot determine mnemonic format. Words do not match any known wordlist."
            )
        if not self.is_valid_mnemonic(legacy_mnemonic, expected_format=FORMAT_LEGACY):
            raise ValueError(
                "Invalid legacy mnemonic: not all words are in the Positronic wordlist "
                "or the word count is invalid."
            )

        # Restore the legacy wallet
        legacy_wallet = HDWallet.from_mnemonic(legacy_mnemonic, passphrase=password, legacy=True)
        legacy_addresses = legacy_wallet.get_all_addresses(account=0, count=num_addresses)

        # Create a new BIP-39 wallet
        new_wallet = HDWallet.create(word_count=new_word_count, legacy=False)
        new_addresses = new_wallet.get_all_addresses(account=0, count=num_addresses)

        # Build the address mapping
        mapping: List[Dict] = []
        for old, new in zip(legacy_addresses, new_addresses):
            mapping.append({
                "index": old["index"],
                "legacy_address": old["address"],
                "legacy_path": old["path"],
                "new_address": new["address"],
                "new_path": new["path"],
            })

        return {
            "legacy_format": FORMAT_LEGACY,
            "legacy_mnemonic": legacy_mnemonic,
            "new_mnemonic": new_wallet.mnemonic,
            "new_format": FORMAT_BIP39,
            "legacy_addresses": legacy_addresses,
            "new_addresses": new_addresses,
            "address_mapping": mapping,
        }

    def get_migration_summary(self, migration_result: Dict) -> str:
        """Return a human-readable summary of a migration result.

        Args:
            migration_result: The dict returned by :meth:`migrate`.

        Returns:
            Formatted string summarising the migration.
        """
        lines = [
            "=== Wallet Migration Summary ===",
            "",
            f"Legacy format : {migration_result['legacy_format']}",
            f"New format    : {migration_result['new_format']}",
            "",
            "NEW MNEMONIC (save this securely!):",
            f"  {migration_result['new_mnemonic']}",
            "",
            "Address Mapping (transfer funds from legacy -> new):",
        ]
        for entry in migration_result["address_mapping"]:
            lines.append(
                f"  [{entry['index']}] {entry['legacy_address'][:16]}... -> "
                f"{entry['new_address'][:16]}..."
            )
        lines.append("")
        lines.append("IMPORTANT: Transfer all funds before discarding the legacy mnemonic.")
        return "\n".join(lines)
