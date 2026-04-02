"""
Positronic - PositronicVM Assembler / Compiler

Simple assembler that converts text assembly source code into PositronicVM bytecode.

Syntax:
    - One instruction per line
    - Comments start with ';' (inline or full-line)
    - Labels are identifiers followed by ':' (e.g., loop:)
    - PUSH instructions accept hex data prefixed with 0x
    - Blank lines are ignored

Example:
    ; Simple addition contract
    PUSH1 0x03       ; push 3
    PUSH1 0x05       ; push 5
    ADD              ; 3 + 5 = 8
    PUSH1 0x00
    MSTORE           ; store result at memory[0]
    PUSH1 0x20
    PUSH1 0x00
    RETURN           ; return 32 bytes from memory[0]

Label example:
    PUSH1 0x00
    CALLDATALOAD
    ISZERO
    PUSH1 skip       ; resolves to label address
    JUMPI
    PUSH1 0x01
    PUSH1 0x00
    MSTORE
    skip:
    JUMPDEST
    STOP
"""

import re
from typing import Dict, List, Tuple, Optional

from positronic.vm.opcodes import Opcode, get_push_size


class CompilerError(Exception):
    """Raised on assembly syntax or semantic errors."""

    def __init__(self, message: str, line_number: int = 0, line_text: str = ""):
        self.line_number = line_number
        self.line_text = line_text
        super().__init__(
            f"Line {line_number}: {message}"
            + (f" -> '{line_text}'" if line_text else "")
        )


class PositronicCompiler:
    """
    Simple assembler for PositronicVM bytecode.

    Converts human-readable assembly text into raw bytecode that can
    be executed by the PositronicVM.

    Features:
        - All PositronicVM opcodes supported by name
        - PUSH1-PUSH32 with hex literal data
        - Labels for jump targets (automatically resolved)
        - Comments with ';'
        - Two-pass assembly: first pass collects labels, second emits bytecode
    """

    # Build opcode name lookup (uppercase name -> Opcode)
    _OPCODE_MAP: Dict[str, Opcode] = {op.name: op for op in Opcode}

    def __init__(self):
        self._labels: Dict[str, int] = {}
        self._source_lines: List[str] = []

    def compile(self, source: str) -> bytes:
        """
        Compile assembly source text to bytecode.

        Args:
            source: Assembly source code (multi-line string).

        Returns:
            Raw bytecode as bytes.

        Raises:
            CompilerError: On syntax errors, unknown opcodes, or unresolved labels.
        """
        self._labels.clear()
        lines = source.strip().split("\n")
        self._source_lines = lines

        # Tokenize: strip comments and whitespace, identify labels
        tokens = self._tokenize(lines)

        # First pass: calculate offsets and collect labels
        self._first_pass(tokens)

        # Second pass: emit bytecode with resolved labels
        bytecode = self._second_pass(tokens)

        return bytes(bytecode)

    def _tokenize(self, lines: List[str]) -> List[Tuple[int, str, Optional[str]]]:
        """
        Tokenize source lines into a list of (line_number, instruction, label_or_none).

        Strips comments, identifies labels, normalizes whitespace.

        Returns:
            List of (line_num, instruction_text, label_name_or_None).
            If a line is only a label, instruction_text is empty.
        """
        tokens = []
        for line_num, raw_line in enumerate(lines, 1):
            # Strip comments
            line = raw_line.split(";")[0].strip()
            if not line:
                continue

            # Check for label definition
            label = None
            if ":" in line:
                parts = line.split(":", 1)
                potential_label = parts[0].strip()
                # Labels must be alphanumeric + underscore, not a valid opcode
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", potential_label):
                    if potential_label.upper() not in self._OPCODE_MAP:
                        label = potential_label
                        line = parts[1].strip()
                    # else it might be an opcode followed by colon -- error
                    # but we allow it as a label if it's not an opcode name

            tokens.append((line_num, line, label))

        return tokens

    def _first_pass(
        self, tokens: List[Tuple[int, str, Optional[str]]]
    ) -> None:
        """
        First pass: walk through tokens to compute byte offsets for labels.

        Args:
            tokens: Tokenized source lines.
        """
        self._labels.clear()
        offset = 0

        for line_num, instruction, label in tokens:
            # Register label at current offset
            if label is not None:
                if label in self._labels:
                    raise CompilerError(
                        f"Duplicate label '{label}'", line_num
                    )
                self._labels[label] = offset

            if not instruction:
                continue

            # Parse the instruction to compute its size
            size = self._instruction_size(instruction, line_num)
            offset += size

    def _instruction_size(self, instruction: str, line_num: int) -> int:
        """
        Calculate the byte size of a single instruction.

        Args:
            instruction: The instruction text (e.g., "PUSH1 0x05", "ADD").
            line_num: Line number for error reporting.

        Returns:
            Number of bytes this instruction will occupy.
        """
        parts = instruction.split()
        if not parts:
            return 0

        mnemonic = parts[0].upper()

        if mnemonic not in self._OPCODE_MAP:
            raise CompilerError(f"Unknown opcode '{mnemonic}'", line_num, instruction)

        opcode = self._OPCODE_MAP[mnemonic]
        push_size = get_push_size(opcode)

        if push_size > 0:
            # PUSH instruction: 1 byte opcode + N bytes data
            return 1 + push_size
        else:
            # Regular instruction: 1 byte
            return 1

    def _second_pass(
        self, tokens: List[Tuple[int, str, Optional[str]]]
    ) -> bytearray:
        """
        Second pass: emit bytecode with resolved label references.

        Args:
            tokens: Tokenized source lines.

        Returns:
            Assembled bytecode as bytearray.
        """
        bytecode = bytearray()

        for line_num, instruction, label in tokens:
            if not instruction:
                continue

            parts = instruction.split()
            mnemonic = parts[0].upper()
            opcode = self._OPCODE_MAP[mnemonic]
            push_size = get_push_size(opcode)

            # Emit the opcode byte
            bytecode.append(int(opcode))

            if push_size > 0:
                # PUSH instruction: need to parse the data argument
                if len(parts) < 2:
                    raise CompilerError(
                        f"{mnemonic} requires a data argument",
                        line_num,
                        instruction,
                    )

                data_arg = parts[1]
                data_bytes = self._resolve_push_data(
                    data_arg, push_size, line_num, instruction
                )
                bytecode.extend(data_bytes)

        return bytecode

    def _resolve_push_data(
        self,
        data_arg: str,
        push_size: int,
        line_num: int,
        instruction: str,
    ) -> bytes:
        """
        Resolve the data argument for a PUSH instruction.

        Handles:
            - Hex literals: 0xFF, 0x1234
            - Decimal literals: 255, 1000
            - Label references: my_label (resolved to byte offset)

        Args:
            data_arg: The raw data argument string.
            push_size: Expected number of data bytes (1-32).
            line_num: Line number for error reporting.
            instruction: Full instruction text for error reporting.

        Returns:
            Data bytes, zero-padded or truncated to push_size.
        """
        value = None

        # Try hex literal
        if data_arg.startswith("0x") or data_arg.startswith("0X"):
            try:
                hex_str = data_arg[2:]
                # Pad to even length
                if len(hex_str) % 2 != 0:
                    hex_str = "0" + hex_str
                raw = bytes.fromhex(hex_str)
                value = int.from_bytes(raw, "big")
            except ValueError:
                raise CompilerError(
                    f"Invalid hex literal '{data_arg}'", line_num, instruction
                )

        # Try decimal literal
        elif data_arg.isdigit():
            value = int(data_arg)

        # Try label reference
        elif data_arg in self._labels:
            value = self._labels[data_arg]

        # Check for label-like identifier that was not found
        elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", data_arg):
            raise CompilerError(
                f"Unresolved label '{data_arg}'", line_num, instruction
            )

        else:
            raise CompilerError(
                f"Invalid PUSH argument '{data_arg}'", line_num, instruction
            )

        # Convert value to bytes of the correct size
        max_val = (1 << (push_size * 8)) - 1
        if value > max_val:
            raise CompilerError(
                f"Value 0x{value:x} too large for {instruction.split()[0]} "
                f"(max {push_size} bytes)",
                line_num,
                instruction,
            )

        return value.to_bytes(push_size, "big")

    def compile_to_hex(self, source: str) -> str:
        """
        Compile assembly source to a hex string.

        Args:
            source: Assembly source code.

        Returns:
            Hex-encoded bytecode string (no 0x prefix).
        """
        return self.compile(source).hex()

    def disassemble(self, bytecode: bytes) -> str:
        """
        Disassemble bytecode back to human-readable assembly.

        Args:
            bytecode: Raw bytecode.

        Returns:
            Multi-line assembly text.
        """
        lines = []
        i = 0
        opcode_names = {int(op): op.name for op in Opcode}

        while i < len(bytecode):
            op_byte = bytecode[i]
            name = opcode_names.get(op_byte, f"UNKNOWN(0x{op_byte:02x})")

            if 0x60 <= op_byte <= 0x7F:
                # PUSH instruction
                num_bytes = op_byte - 0x5F
                data = bytecode[i + 1: i + 1 + num_bytes]
                hex_data = "0x" + data.hex()
                lines.append(f"{i:04x}: {name} {hex_data}")
                i += 1 + num_bytes
            else:
                lines.append(f"{i:04x}: {name}")
                i += 1

        return "\n".join(lines)

    def get_labels(self) -> Dict[str, int]:
        """
        Get the label-to-offset mapping from the last compilation.

        Returns:
            Dictionary mapping label names to byte offsets.
        """
        return dict(self._labels)


# ===== Module-level convenience functions =====

def assemble(source: str) -> bytes:
    """
    Assemble PositronicVM assembly source code to bytecode.

    Args:
        source: Assembly source code (multi-line string).

    Returns:
        Raw bytecode.
    """
    compiler = PositronicCompiler()
    return compiler.compile(source)


def disassemble(bytecode: bytes) -> str:
    """
    Disassemble PositronicVM bytecode to human-readable assembly.

    Args:
        bytecode: Raw bytecode.

    Returns:
        Multi-line assembly text.
    """
    compiler = PositronicCompiler()
    return compiler.disassemble(bytecode)
