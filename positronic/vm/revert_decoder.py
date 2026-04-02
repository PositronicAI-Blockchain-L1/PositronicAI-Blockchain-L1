"""
Positronic - Revert Reason Decoder

Decodes human-readable error messages from EVM-style revert data.
Supports Error(string), Panic(uint256), and custom error selectors.

Phase 17 GOD CHAIN addition.
"""

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


class RevertDecoder:
    """Decode revert data into human-readable error messages.

    Supports three formats:
        - ``Error(string)``: Selector ``0x08c379a0`` followed by ABI-encoded string.
        - ``Panic(uint256)``: Selector ``0x4e487b71`` followed by uint256 error code.
        - Custom errors: Returns the 4-byte selector as hex for lookup.

    All methods are static and never raise -- they return a fallback string on
    any decoding failure.

    Example::

        reason = RevertDecoder.decode(result.return_data)
        print(f"Reverted: {reason}")
    """

    # Standard Error(string) function selector
    ERROR_SELECTOR = bytes.fromhex("08c379a0")

    # Panic(uint256) function selector
    PANIC_SELECTOR = bytes.fromhex("4e487b71")

    # Known Solidity panic codes
    PANIC_CODES = {
        0x00: "Generic compiler panic",
        0x01: "Assert condition failed",
        0x11: "Arithmetic overflow or underflow",
        0x12: "Division or modulo by zero",
        0x21: "Invalid enum conversion",
        0x22: "Storage encoding error",
        0x31: "Array pop on empty array",
        0x32: "Array index out of bounds",
        0x41: "Too much memory allocated",
        0x51: "Zero-initialized function pointer called",
    }

    @staticmethod
    def decode(revert_data: bytes) -> str:
        """Decode revert data into a human-readable string.

        Args:
            revert_data: Raw bytes returned from a reverted execution.

        Returns:
            Human-readable error message. Returns ``"Unknown error"`` for
            empty data and ``"0x<hex>"`` if the format is unrecognized.
        """
        if not revert_data:
            return "Unknown error"

        try:
            # Try Error(string) format
            if (len(revert_data) >= 4
                    and revert_data[:4] == RevertDecoder.ERROR_SELECTOR):
                return RevertDecoder._decode_error_string(revert_data[4:])

            # Try Panic(uint256) format
            if (len(revert_data) >= 4
                    and revert_data[:4] == RevertDecoder.PANIC_SELECTOR):
                return RevertDecoder._decode_panic(revert_data[4:])

            # Custom error -- show selector
            if len(revert_data) >= 4:
                selector = revert_data[:4].hex()
                return f"Custom error: 0x{selector}"

            # Too short for any known format
            return f"0x{revert_data.hex()}"

        except Exception as e:
            logger.debug("Revert data decode failed, returning raw hex: %s", e)
            return f"0x{revert_data.hex()}"

    @staticmethod
    def _decode_error_string(data: bytes) -> str:
        """Decode ABI-encoded string from Error(string) revert.

        ABI layout:
            bytes[0:32]  = offset (always 0x20)
            bytes[32:64] = string length
            bytes[64:]   = UTF-8 string data
        """
        if len(data) < 64:
            return f"Error (malformed): 0x{data.hex()}"

        # Read string length (bytes 32-63)
        str_len = int.from_bytes(data[32:64], "big")
        if str_len == 0:
            return "Error: (empty message)"

        # Read string data
        str_data = data[64:64 + str_len]
        try:
            return f"Error: {str_data.decode('utf-8')}"
        except UnicodeDecodeError:
            return f"Error: 0x{str_data.hex()}"

    @staticmethod
    def _decode_panic(data: bytes) -> str:
        """Decode Panic(uint256) revert code."""
        if len(data) < 32:
            return f"Panic (malformed): 0x{data.hex()}"

        code = int.from_bytes(data[:32], "big")
        description = RevertDecoder.PANIC_CODES.get(code, f"Unknown panic code")
        return f"Panic(0x{code:02x}): {description}"

    @staticmethod
    def encode_revert(message: str) -> bytes:
        """Encode a revert message in Error(string) ABI format.

        This is useful for contracts that want to emit human-readable
        revert reasons via the REVERT opcode.

        Args:
            message: The error message string.

        Returns:
            ABI-encoded revert data with Error(string) selector.
        """
        msg_bytes = message.encode("utf-8")
        # ABI encoding: selector + offset + length + data (padded to 32)
        offset = (32).to_bytes(32, "big")
        length = len(msg_bytes).to_bytes(32, "big")
        # Pad message to 32-byte boundary
        padded_len = ((len(msg_bytes) + 31) // 32) * 32
        padded_msg = msg_bytes.ljust(padded_len, b"\x00")
        return RevertDecoder.ERROR_SELECTOR + offset + length + padded_msg
