"""
Positronic - PositronicVM Precompiled Contracts

Precompiled contracts are built-in functions available at fixed addresses.
They provide efficient implementations of computationally expensive operations
that would be prohibitively costly to implement in bytecode.

Precompiled contracts:
    Address 0x01 - SHA512:       SHA-512 hash of input data
    Address 0x02 - BLAKE2B:      Blake2b-256 hash of input data
    Address 0x03 - ECVERIFY:     Ed25519 signature verification
    Address 0x04 - AI_QUERY:     Query the AI model for a risk score
    Address 0x05 - BASE64:       Base64 encode/decode (Phase 17)
    Address 0x06 - JSON_PARSE:   JSON field extraction (Phase 17)
    Address 0x07 - BATCH_VERIFY: Batch Ed25519 verification (Phase 17)

Each precompile is a function that takes raw input bytes and returns
(output_bytes, gas_consumed). Raises an exception on invalid input.
"""

import base64
import json
from typing import Callable, Dict, Tuple

from positronic.crypto.hashing import sha512, sha256, blake2b_256
from positronic.crypto.keys import KeyPair


# ===== Precompile addresses (20-byte, left-padded) =====

PRECOMPILE_SHA512 = b"\x00" * 19 + b"\x01"       # 0x0000...0001
PRECOMPILE_BLAKE2B = b"\x00" * 19 + b"\x02"      # 0x0000...0002
PRECOMPILE_ECVERIFY = b"\x00" * 19 + b"\x03"     # 0x0000...0003
PRECOMPILE_AI_QUERY = b"\x00" * 19 + b"\x04"     # 0x0000...0004
PRECOMPILE_BASE64 = b"\x00" * 19 + b"\x05"       # 0x0000...0005 (Phase 17)
PRECOMPILE_JSON_PARSE = b"\x00" * 19 + b"\x06"   # 0x0000...0006 (Phase 17)
PRECOMPILE_BATCH_VERIFY = b"\x00" * 19 + b"\x07" # 0x0000...0007 (Phase 17)

# Type alias for precompile functions
PrecompileFn = Callable[[bytes], Tuple[bytes, int]]


class PrecompileError(Exception):
    """Raised when a precompile encounters invalid input."""
    pass


# ===== SHA-512 Precompile (Address 0x01) =====

def precompile_sha512(input_data: bytes) -> Tuple[bytes, int]:
    """
    SHA-512 precompile.

    Computes the SHA-512 hash of the input data.

    Input:  Arbitrary-length bytes
    Output: 64-byte SHA-512 hash
    Gas:    60 base + 12 per 32-byte word of input

    Args:
        input_data: Raw input bytes to hash.

    Returns:
        Tuple of (64-byte hash, gas consumed).
    """
    base_gas = 60
    word_gas = 12
    words = (len(input_data) + 31) // 32
    total_gas = base_gas + word_gas * words

    result = sha512(input_data)
    return result, total_gas


# ===== Blake2b-256 Precompile (Address 0x02) =====

def precompile_blake2b(input_data: bytes) -> Tuple[bytes, int]:
    """
    Blake2b-256 precompile.

    Computes the Blake2b hash with a 256-bit (32-byte) digest.

    Input:  Arbitrary-length bytes
    Output: 32-byte Blake2b-256 hash
    Gas:    40 base + 8 per 32-byte word of input

    Args:
        input_data: Raw input bytes to hash.

    Returns:
        Tuple of (32-byte hash, gas consumed).
    """
    base_gas = 40
    word_gas = 8
    words = (len(input_data) + 31) // 32
    total_gas = base_gas + word_gas * words

    result = blake2b_256(input_data)
    return result, total_gas


# ===== Ed25519 Signature Verification Precompile (Address 0x03) =====

def precompile_ecverify(input_data: bytes) -> Tuple[bytes, int]:
    """
    Ed25519 signature verification precompile (ECVERIFY).

    Verifies an Ed25519 signature against a public key and message.

    Input format (fixed layout):
        bytes[0:32]   - Public key (32 bytes, Ed25519)
        bytes[32:96]  - Signature (64 bytes, Ed25519)
        bytes[96:]    - Message (variable length)

    Output: 32 bytes
        0x00...0001 if signature is valid
        0x00...0000 if signature is invalid

    Gas: 3000 (fixed)

    Args:
        input_data: Concatenated pubkey + signature + message.

    Returns:
        Tuple of (32-byte result, gas consumed).

    Raises:
        PrecompileError: If input is too short (< 96 bytes).
    """
    gas_cost = 3000

    if len(input_data) < 96:
        raise PrecompileError(
            f"ECVERIFY: input too short ({len(input_data)} bytes, need >= 96)"
        )

    pubkey = input_data[0:32]
    signature = input_data[32:96]
    message = input_data[96:]

    valid = KeyPair.verify(pubkey, signature, message)

    # Return 32-byte result: 1 for valid, 0 for invalid
    result = (1 if valid else 0).to_bytes(32, "big")
    return result, gas_cost


# ===== AI Query Precompile (Address 0x04) =====

def precompile_ai_query(input_data: bytes) -> Tuple[bytes, int]:
    """
    AI Query precompile.

    Queries the on-chain AI model for a risk assessment score.
    This precompile provides contracts with access to the Positronic
    AI validation layer.

    Input format:
        bytes[0:32]   - Query type (uint256):
                        0 = Transaction risk score
                        1 = Address reputation score
                        2 = Contract risk assessment
        bytes[32:]    - Query data (variable, depends on query type):
                        Type 0: 64-byte transaction hash
                        Type 1: 20-byte address (right-aligned in 32 bytes)
                        Type 2: Contract bytecode

    Output: 32 bytes
        uint256 score scaled to [0, 10000] representing [0.0, 1.0]
        0    = completely safe
        10000 = maximum risk

    Gas: 5000 base + 10 per 32-byte word of query data

    Note: In the current implementation, this returns a placeholder score.
    The actual AI model integration happens at the consensus layer.

    Args:
        input_data: Query type + query data.

    Returns:
        Tuple of (32-byte score, gas consumed).
    """
    base_gas = 5000
    word_gas = 10
    words = (max(len(input_data) - 32, 0) + 31) // 32
    total_gas = base_gas + word_gas * words

    if len(input_data) < 32:
        raise PrecompileError(
            f"AI_QUERY: input too short ({len(input_data)} bytes, need >= 32)"
        )

    # Parse query type
    query_type = int.from_bytes(input_data[0:32], "big")
    query_data = input_data[32:]

    score = _evaluate_ai_query(query_type, query_data)

    # Clamp score to [0, 10000]
    score = max(0, min(10000, score))

    result = score.to_bytes(32, "big")
    return result, total_gas


def _evaluate_ai_query(query_type: int, query_data: bytes) -> int:
    """
    Evaluate an AI query and return a risk score.

    This is a deterministic placeholder implementation. In production,
    this would interface with the trained AI model from positronic.ai.

    Args:
        query_type: Type of query (0=tx risk, 1=addr reputation, 2=contract risk).
        query_data: Raw query data.

    Returns:
        Risk score in range [0, 10000].
    """
    if query_type == 0:
        # Transaction risk score
        # Placeholder: hash the query data and derive a deterministic score
        if len(query_data) >= 64:
            tx_hash = query_data[:64]
            # Use first 2 bytes of hash to derive a score
            raw = int.from_bytes(sha256(tx_hash)[:2], "big")
            return raw % 10001
        return 0

    elif query_type == 1:
        # Address reputation score
        # Placeholder: return low risk (high reputation) by default
        if len(query_data) >= 20:
            addr = query_data[:20]
            # Deterministic score from address
            raw = int.from_bytes(sha256(addr)[:2], "big")
            # Most addresses are low risk
            return min(raw % 3000, 2000)
        return 0

    elif query_type == 2:
        # Contract risk assessment
        # Placeholder: analyze bytecode size and complexity heuristics
        if not query_data:
            return 0
        code = query_data
        # Simple heuristic: larger contracts have slightly higher base risk
        size_factor = min(len(code) * 2, 3000)
        # Check for potentially dangerous opcodes
        danger_score = 0
        for byte in code:
            if byte == 0xFF:  # SELFDESTRUCT
                danger_score += 500
            elif byte == 0xF1:  # CALL
                danger_score += 100
            elif byte == 0xF4:  # DELEGATECALL
                danger_score += 200
        danger_score = min(danger_score, 5000)
        return min(size_factor + danger_score, 10000)

    else:
        # Unknown query type
        return 5000  # Medium risk for unknown queries


# ===== Base64 Precompile (Address 0x05) — Phase 17 =====

def precompile_base64(input_data: bytes) -> Tuple[bytes, int]:
    """
    Base64 encode/decode precompile.

    Input format:
        bytes[0]  - Mode: 0 = encode, 1 = decode
        bytes[1:] - Data to encode or decode

    Output: Encoded/decoded bytes (variable length)
    Gas: 30 base + 5 per 32-byte word of input

    Args:
        input_data: Mode byte + data.

    Returns:
        Tuple of (result bytes, gas consumed).

    Raises:
        PrecompileError: If input is empty or decode fails.
    """
    base_gas = 30
    word_gas = 5
    words = (max(len(input_data) - 1, 0) + 31) // 32
    total_gas = base_gas + word_gas * words

    if len(input_data) < 1:
        raise PrecompileError("BASE64: empty input")

    mode = input_data[0]
    data = input_data[1:]

    if mode == 0:
        # Encode
        result = base64.b64encode(data)
    elif mode == 1:
        # Decode
        try:
            result = base64.b64decode(data, validate=True)
        except Exception as e:
            raise PrecompileError(f"BASE64: decode error: {e}")
    else:
        raise PrecompileError(f"BASE64: unknown mode {mode} (0=encode, 1=decode)")

    return result, total_gas


# ===== JSON Parse Precompile (Address 0x06) — Phase 17 =====

def precompile_json_parse(input_data: bytes) -> Tuple[bytes, int]:
    """
    JSON field extraction precompile.

    Parses a JSON object and extracts a named field. Returns the value as
    a 32-byte ABI-encoded result (uint256 for numbers, bytes32 for strings).

    Input format:
        bytes[0:32]  - Field name length (uint256, big-endian)
        bytes[32:32+len] - Field name (UTF-8)
        bytes[32+len:] - JSON data (UTF-8)

    Output: 32 bytes (ABI-encoded value)
    Gas: 100 base + 10 per 32-byte word of input

    Args:
        input_data: Field name length + field name + JSON data.

    Returns:
        Tuple of (32-byte result, gas consumed).

    Raises:
        PrecompileError: If input is malformed or JSON is invalid.
    """
    base_gas = 100
    word_gas = 10
    words = (len(input_data) + 31) // 32
    total_gas = base_gas + word_gas * words

    if len(input_data) < 32:
        raise PrecompileError("JSON_PARSE: input too short (need >= 32 bytes)")

    # Read field name length
    name_len = int.from_bytes(input_data[0:32], "big")
    if name_len > 256:
        raise PrecompileError(f"JSON_PARSE: field name too long ({name_len})")

    if len(input_data) < 32 + name_len:
        raise PrecompileError("JSON_PARSE: input too short for field name")

    field_name = input_data[32:32 + name_len].decode("utf-8", errors="replace")
    json_data = input_data[32 + name_len:]

    try:
        obj = json.loads(json_data.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise PrecompileError(f"JSON_PARSE: invalid JSON: {e}")

    if not isinstance(obj, dict):
        raise PrecompileError("JSON_PARSE: top-level value must be an object")

    if field_name not in obj:
        raise PrecompileError(f"JSON_PARSE: field '{field_name}' not found")

    value = obj[field_name]

    # Encode value as 32 bytes
    if isinstance(value, (int, float)):
        int_val = int(value)
        int_val = max(0, min(int_val, (1 << 256) - 1))
        result = int_val.to_bytes(32, "big")
    elif isinstance(value, bool):
        result = (1 if value else 0).to_bytes(32, "big")
    elif isinstance(value, str):
        str_bytes = value.encode("utf-8")[:32]
        result = str_bytes.ljust(32, b"\x00")
    else:
        # Arrays, objects, null -> return zero
        result = b"\x00" * 32

    return result, total_gas


# ===== Batch Ed25519 Verification Precompile (Address 0x07) — Phase 17 =====

def precompile_batch_verify(input_data: bytes) -> Tuple[bytes, int]:
    """
    Batch Ed25519 signature verification precompile.

    Verifies multiple Ed25519 signatures in a single call at a discounted
    gas rate (2000 per signature vs 3000 for individual ECVERIFY).

    Input format:
        bytes[0:32]   - Count of signatures (uint256, big-endian)
        Then for each signature:
            bytes[+0:+32]  - Public key (32 bytes)
            bytes[+32:+96] - Signature (64 bytes)
            bytes[+96:+128] - Message length (uint256, big-endian)
            bytes[+128:+128+msg_len] - Message data

    Output: 32 bytes
        0x00...0001 if ALL signatures are valid
        0x00...0000 if ANY signature is invalid

    Gas: 2000 per signature (discounted from 3000)

    Args:
        input_data: Count + repeated (pubkey + signature + msg_len + message).

    Returns:
        Tuple of (32-byte result, gas consumed).

    Raises:
        PrecompileError: If input format is invalid.
    """
    if len(input_data) < 32:
        raise PrecompileError("BATCH_VERIFY: input too short (need count)")

    count = int.from_bytes(input_data[0:32], "big")
    if count == 0:
        return (1).to_bytes(32, "big"), 100  # Base cost for empty batch
    if count > 256:
        raise PrecompileError(f"BATCH_VERIFY: too many signatures ({count}, max 256)")

    gas_per_sig = 2000
    total_gas = gas_per_sig * count

    offset = 32
    all_valid = True

    for i in range(count):
        # Read pubkey (32 bytes)
        if offset + 32 > len(input_data):
            raise PrecompileError(f"BATCH_VERIFY: truncated at sig {i} (pubkey)")
        pubkey = input_data[offset:offset + 32]
        offset += 32

        # Read signature (64 bytes)
        if offset + 64 > len(input_data):
            raise PrecompileError(f"BATCH_VERIFY: truncated at sig {i} (signature)")
        signature = input_data[offset:offset + 64]
        offset += 64

        # Read message length (32 bytes)
        if offset + 32 > len(input_data):
            raise PrecompileError(f"BATCH_VERIFY: truncated at sig {i} (msg_len)")
        msg_len = int.from_bytes(input_data[offset:offset + 32], "big")
        offset += 32

        # Read message
        if offset + msg_len > len(input_data):
            raise PrecompileError(f"BATCH_VERIFY: truncated at sig {i} (message)")
        message = input_data[offset:offset + msg_len]
        offset += msg_len

        # Verify
        if not KeyPair.verify(pubkey, signature, message):
            all_valid = False
            break  # Short-circuit on first failure

    result = (1 if all_valid else 0).to_bytes(32, "big")
    return result, total_gas


# ===== Precompile Registry =====

PRECOMPILED_CONTRACTS: Dict[bytes, PrecompileFn] = {
    PRECOMPILE_SHA512: precompile_sha512,
    PRECOMPILE_BLAKE2B: precompile_blake2b,
    PRECOMPILE_ECVERIFY: precompile_ecverify,
    PRECOMPILE_AI_QUERY: precompile_ai_query,
    PRECOMPILE_BASE64: precompile_base64,
    PRECOMPILE_JSON_PARSE: precompile_json_parse,
    PRECOMPILE_BATCH_VERIFY: precompile_batch_verify,
}


def is_precompile(address: bytes) -> bool:
    """
    Check if an address is a precompiled contract.

    Args:
        address: 20-byte address.

    Returns:
        True if the address hosts a precompiled contract.
    """
    return address in PRECOMPILED_CONTRACTS


def get_precompile(address: bytes) -> PrecompileFn:
    """
    Get the precompile function for an address.

    Args:
        address: 20-byte precompile address.

    Returns:
        The precompile function.

    Raises:
        KeyError: If address is not a precompile.
    """
    return PRECOMPILED_CONTRACTS[address]


def list_precompiles() -> Dict[str, bytes]:
    """
    List all precompiled contract addresses with human-readable names.

    Returns:
        Dictionary mapping name -> address.
    """
    return {
        "SHA512": PRECOMPILE_SHA512,
        "BLAKE2B": PRECOMPILE_BLAKE2B,
        "ECVERIFY": PRECOMPILE_ECVERIFY,
        "AI_QUERY": PRECOMPILE_AI_QUERY,
        "BASE64": PRECOMPILE_BASE64,
        "JSON_PARSE": PRECOMPILE_JSON_PARSE,
        "BATCH_VERIFY": PRECOMPILE_BATCH_VERIFY,
    }
