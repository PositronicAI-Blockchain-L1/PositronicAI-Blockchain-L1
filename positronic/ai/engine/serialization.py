"""
Positronic - Model Serialization

Save and load neural model weights to/from a compact binary format.
Designed for three primary use-cases:

    1. **Disk storage** -- persist trained model checkpoints.
    2. **Gossip protocol** -- compact payloads for peer-to-peer weight sharing.
    3. **Governance proposals** -- attach model updates to on-chain proposals.

Binary wire format
------------------
::

    [4 bytes : uint32]  number of parameters (N)
    For each of the N parameters:
        [4 bytes : uint32]  length of the name in bytes (L)
        [L bytes : utf-8 ]  parameter name
        [4 bytes : uint32]  number of dimensions (ndim)
        [ndim*4  : uint32]  shape (one uint32 per dimension)
        [1 byte  : uint8 ]  dtype code (see _DTYPE_TO_CODE)
        [variable: bytes ]  raw data in C-contiguous order
    [32 bytes : bytes]      SHA-256 checksum of everything above
"""

import hashlib
import struct
from typing import Dict

import numpy as np

# ---------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------

_DTYPE_TO_CODE: Dict[np.dtype, int] = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
    np.dtype(np.int32): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.float16): 4,
    np.dtype(np.uint8): 5,
    np.dtype(np.int8): 6,
    np.dtype(np.bool_): 7,
}

_CODE_TO_DTYPE: Dict[int, np.dtype] = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _dtype_to_code(dtype: np.dtype) -> int:
    """Convert a NumPy dtype to a single-byte code.

    Args:
        dtype: The NumPy dtype to encode.

    Returns:
        Integer code in the range [0, 255].

    Raises:
        ValueError: If the dtype is not supported.
    """
    dtype = np.dtype(dtype)
    if dtype not in _DTYPE_TO_CODE:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. "
            f"Supported: {list(_DTYPE_TO_CODE.keys())}"
        )
    return _DTYPE_TO_CODE[dtype]


def _code_to_dtype(code: int) -> np.dtype:
    """Convert a single-byte code back to a NumPy dtype.

    Args:
        code: Integer code produced by :func:`_dtype_to_code`.

    Returns:
        Corresponding NumPy dtype.

    Raises:
        ValueError: If the code is unrecognized.
    """
    if code not in _CODE_TO_DTYPE:
        raise ValueError(
            f"Unknown dtype code {code}. "
            f"Known codes: {list(_CODE_TO_DTYPE.keys())}"
        )
    return _CODE_TO_DTYPE[code]


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def serialize_state(state_dict: Dict[str, np.ndarray]) -> bytes:
    """
    Serialize a state dictionary to a self-contained byte string.

    The output includes a trailing SHA-256 checksum so that receivers
    can verify data integrity (important for gossip protocol transport).

    Args:
        state_dict: Mapping of parameter names to NumPy arrays.

    Returns:
        Byte string encoding every parameter plus a 32-byte checksum.

    Raises:
        ValueError: If any array has an unsupported dtype.
        TypeError: If any value is not a NumPy ndarray.
    """
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected dict, got {type(state_dict).__name__}")

    buf = bytearray()

    # Number of parameters
    buf.extend(struct.pack("<I", len(state_dict)))

    for name, arr in state_dict.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"Value for '{name}' must be np.ndarray, "
                f"got {type(arr).__name__}"
            )

        # Ensure C-contiguous layout
        arr = np.ascontiguousarray(arr)

        # Parameter name
        name_bytes = name.encode("utf-8")
        buf.extend(struct.pack("<I", len(name_bytes)))
        buf.extend(name_bytes)

        # Shape
        buf.extend(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            buf.extend(struct.pack("<I", dim))

        # Dtype
        buf.extend(struct.pack("<B", _dtype_to_code(arr.dtype)))

        # Raw data
        buf.extend(arr.tobytes())

    # Checksum over all preceding bytes
    checksum = hashlib.sha256(bytes(buf)).digest()
    buf.extend(checksum)

    return bytes(buf)


def deserialize_state(data: bytes) -> Dict[str, np.ndarray]:
    """
    Reconstruct a state dictionary from bytes produced by
    :func:`serialize_state`.

    Verifies the SHA-256 checksum before returning.

    Args:
        data: Byte string containing the serialized state.

    Returns:
        Dictionary mapping parameter names to NumPy arrays.

    Raises:
        ValueError: If the checksum does not match (corrupt data).
        struct.error: If the binary layout is malformed.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(data).__name__}")

    if len(data) < 36:
        raise ValueError("Data too short to contain a valid serialized state.")

    # Split payload and checksum
    payload = data[:-32]
    expected_checksum = data[-32:]
    actual_checksum = hashlib.sha256(payload).digest()

    if actual_checksum != expected_checksum:
        raise ValueError(
            "Checksum mismatch: data is corrupted or has been tampered with. "
            f"Expected {expected_checksum.hex()}, got {actual_checksum.hex()}"
        )

    offset = 0

    def _read(fmt: str) -> tuple:
        nonlocal offset
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, payload, offset)
        offset += size
        return values

    (num_params,) = _read("<I")
    state_dict: Dict[str, np.ndarray] = {}

    for _ in range(num_params):
        # Name
        (name_len,) = _read("<I")
        name = payload[offset : offset + name_len].decode("utf-8")
        offset += name_len

        # Shape
        (ndim,) = _read("<I")
        shape = tuple(_read(f"<{'I' * ndim}")) if ndim > 0 else ()

        # Dtype
        (dtype_code,) = _read("<B")
        dtype = _code_to_dtype(dtype_code)

        # Data
        num_elements = 1
        for s in shape:
            num_elements *= s
        data_size = num_elements * dtype.itemsize
        arr = np.frombuffer(payload, dtype=dtype, count=num_elements, offset=offset)
        arr = arr.reshape(shape).copy()  # own memory, correct shape
        offset += data_size

        state_dict[name] = arr

    return state_dict


def compute_checksum(state_dict: Dict[str, np.ndarray]) -> str:
    """
    Compute a deterministic SHA-256 hex digest over model weights.

    Useful for comparing two models or verifying that a governance
    proposal contains the expected weights.

    Args:
        state_dict: Mapping of parameter names to NumPy arrays.

    Returns:
        64-character lowercase hex string.
    """
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        arr = np.ascontiguousarray(state_dict[name])
        h.update(name.encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def state_dict_size(state_dict: Dict[str, np.ndarray]) -> int:
    """
    Compute the total in-memory size of model weights in bytes.

    This reports the raw weight data only (no serialization overhead).

    Args:
        state_dict: Mapping of parameter names to NumPy arrays.

    Returns:
        Total byte count across all arrays.
    """
    total = 0
    for arr in state_dict.values():
        total += arr.nbytes
    return total
