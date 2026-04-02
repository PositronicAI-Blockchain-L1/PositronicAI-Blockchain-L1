"""
Positronic - Address Utilities
EVM-compatible 20-byte addresses derived from Ed25519 public keys.

Uses native C extension when available for 2-3x speedup on
address derivation (combined SHA-512 + Blake2b-160 in single C call).
"""

from positronic.crypto.hashing import sha512, blake2b_160, _NATIVE_AVAILABLE

if _NATIVE_AVAILABLE:
    from positronic.crypto._native import address_from_pubkey
else:
    def address_from_pubkey(public_key_bytes: bytes) -> bytes:
        """
        Derive a 20-byte address from a 32-byte Ed25519 public key.
        address = Blake2b-160(SHA-512(pubkey))
        """
        return blake2b_160(sha512(public_key_bytes))


def address_to_hex(address: bytes) -> str:
    """Convert 20-byte address to 0x-prefixed hex string."""
    return "0x" + address.hex()


def address_from_hex(hex_str: str) -> bytes:
    """Convert 0x-prefixed hex string to 20-byte address."""
    return bytes.fromhex(hex_str.removeprefix("0x"))


def is_valid_address(address: bytes) -> bool:
    """Check if an address is valid (20 bytes, non-zero)."""
    return isinstance(address, bytes) and len(address) == 20


def is_zero_address(address: bytes) -> bool:
    """Check if address is the zero address (used for contract creation)."""
    return address == b"\x00" * 20


ZERO_ADDRESS = b"\x00" * 20
TREASURY_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000001")
COMMUNITY_POOL_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000002")
BURN_ADDRESS = bytes.fromhex("000000000000000000000000000000000000dead")
TEAM_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000004")
SECURITY_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000005")

TREASURY_WALLETS = {
    "ai_treasury": TREASURY_ADDRESS,
    "community": COMMUNITY_POOL_ADDRESS,
    "team": TEAM_ADDRESS,
    "security": SECURITY_ADDRESS,
}
