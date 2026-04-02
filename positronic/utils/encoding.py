"""
Positronic - Encoding Utilities
Hex, Base58, and integer encoding for blockchain data.
"""

import string

from positronic.constants import DENOMINATIONS, POSI

# Base58 alphabet (Bitcoin-style, no 0OIl)
BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to 0x-prefixed hex string."""
    return "0x" + data.hex()


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert 0x-prefixed hex string to bytes."""
    return bytes.fromhex(hex_str.removeprefix("0x"))


def int_to_bytes(value: int, length: int = 32) -> bytes:
    """Convert integer to big-endian bytes of given length."""
    return value.to_bytes(length, "big")


def bytes_to_int(data: bytes) -> int:
    """Convert big-endian bytes to integer."""
    return int.from_bytes(data, "big")


def base58_encode(data: bytes) -> str:
    """Encode bytes to Base58 string."""
    n = int.from_bytes(data, "big")
    result = []
    while n > 0:
        n, remainder = divmod(n, 58)
        result.append(BASE58_ALPHABET[remainder:remainder + 1])
    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result.append(BASE58_ALPHABET[0:1])
        else:
            break
    return b"".join(reversed(result)).decode("ascii")


def base58_decode(s: str) -> bytes:
    """Decode Base58 string to bytes. Raises ValueError for invalid characters."""
    n = 0
    for char in s.encode("ascii"):
        idx = BASE58_ALPHABET.find(char)
        if idx < 0:
            raise ValueError(f"Invalid Base58 character: {chr(char)}")
        n = n * 58 + idx
    # Count leading '1's (represent zero bytes)
    pad = 0
    for char in s:
        if char == "1":
            pad += 1
        else:
            break
    result = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * pad + result


def compact_size(n: int) -> bytes:
    """Encode an integer as a compact size (variable-length encoding)."""
    if n < 0xFD:
        return n.to_bytes(1, "little")
    elif n <= 0xFFFF:
        return b"\xfd" + n.to_bytes(2, "little")
    elif n <= 0xFFFFFFFF:
        return b"\xfe" + n.to_bytes(4, "little")
    else:
        return b"\xff" + n.to_bytes(8, "little")


def format_positronic(base_units: int) -> str:
    """Format base units as human-readable ASF amount."""
    whole = base_units // POSI
    frac = base_units % POSI
    if frac == 0:
        return f"{whole} ASF"
    frac_str = str(frac).rstrip("0").ljust(1, "0")
    return f"{whole}.{frac_str} ASF"


def format_denomination(value: int, denomination: str = "posi") -> str:
    """
    Format a base-unit value in any supported denomination.

    Args:
        value: Amount in base units (Quark).
        denomination: Target denomination name (case-insensitive).
            One of: quark, pulse, wave, flux, field, core, posi.

    Returns:
        Human-readable string like "1.5 CORE" or "1000 PULSE".

    Raises:
        ValueError: If the denomination name is not recognised.
    """
    denom_lower = denomination.lower()
    if denom_lower not in DENOMINATIONS:
        raise ValueError(
            f"Unknown denomination {denomination!r}. "
            f"Valid options: {list(DENOMINATIONS.keys())}"
        )
    divisor = DENOMINATIONS[denom_lower]
    whole = value // divisor
    frac = value % divisor
    label = denom_lower.upper()
    if frac == 0:
        return f"{whole} {label}"
    # Show fractional part, trimming trailing zeros
    frac_str = str(frac).zfill(len(str(divisor)) - 1).rstrip("0")
    return f"{whole}.{frac_str} {label}"
