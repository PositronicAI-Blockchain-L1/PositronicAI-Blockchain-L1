"""
Positronic - Serialization Utilities
JSON and binary serialization for blockchain objects.
"""

import json
import struct
import time
from typing import Any, Dict


def to_json(obj: Any) -> str:
    """Serialize object to JSON string with sorted keys, no spaces (JS-compatible)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), default=_json_default)


def from_json(data: str) -> Any:
    """Deserialize JSON string to object."""
    return json.loads(data)


def to_json_bytes(obj: Any) -> bytes:
    """Serialize object to JSON bytes (UTF-8)."""
    return to_json(obj).encode("utf-8")


def from_json_bytes(data: bytes) -> Any:
    """Deserialize JSON bytes to object."""
    return from_json(data.decode("utf-8"))


def _json_default(obj):
    """Custom JSON serializer for bytes and other types."""
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def pack_uint8(value: int) -> bytes:
    return struct.pack(">B", value)


def pack_uint16(value: int) -> bytes:
    return struct.pack(">H", value)


def pack_uint32(value: int) -> bytes:
    return struct.pack(">I", value)


def pack_uint64(value: int) -> bytes:
    return struct.pack(">Q", value)


def pack_int256(value: int) -> bytes:
    """Pack a 256-bit integer (32 bytes, big-endian, signed)."""
    if value < 0:
        value = (1 << 256) + value
    return value.to_bytes(32, "big")


def unpack_uint64(data: bytes) -> int:
    return struct.unpack(">Q", data)[0]


def pack_bytes(data: bytes) -> bytes:
    """Length-prefixed bytes: 4-byte length + data."""
    return struct.pack(">I", len(data)) + data


def unpack_bytes(data: bytes, offset: int = 0) -> tuple:
    """Unpack length-prefixed bytes. Returns (bytes, new_offset)."""
    length = struct.unpack(">I", data[offset:offset + 4])[0]
    offset += 4
    return data[offset:offset + length], offset + length


def current_timestamp() -> float:
    """Current time as Unix timestamp."""
    return time.time()


def serialize_dict(d: Dict) -> bytes:
    """Deterministic serialization of a dict for hashing."""
    return to_json_bytes(d)
