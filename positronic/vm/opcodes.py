"""
Positronic - PositronicVM Opcode Definitions

Complete instruction set for the PositronicVM smart contract virtual machine.
Each opcode is an IntEnum value (0x00-0xFF) with an associated gas cost.

Categories:
    - Stack operations:  PUSH, POP, DUP, SWAP
    - Arithmetic:        ADD, SUB, MUL, DIV, MOD, EXP
    - Comparison/Logic:  LT, GT, EQ, ISZERO, AND, OR, XOR, NOT
    - Memory:            MLOAD, MSTORE, MSIZE
    - Storage:           SLOAD, SSTORE
    - Flow control:      JUMP, JUMPI, JUMPDEST, PC, STOP
    - Context:           CALLER, CALLVALUE, CALLDATALOAD, etc.
    - Contract:          CALL, DELEGATECALL, CREATE, RETURN, REVERT, SELFDESTRUCT
    - Crypto:            SHA256, BLAKE2B, SHA512, VERIFY_SIG
    - Logging:           LOG0 through LOG4
    - AI-specific:       AI_SCORE, AI_QUARANTINE_STATUS
"""

from enum import IntEnum


class Opcode(IntEnum):
    """PositronicVM instruction set. Each value is a single-byte opcode."""

    # ===== Stop / Arithmetic (0x00 - 0x0F) =====
    STOP = 0x00          # Halt execution
    ADD = 0x01           # Addition
    SUB = 0x02           # Subtraction
    MUL = 0x03           # Multiplication
    DIV = 0x04           # Integer division
    MOD = 0x05           # Modulo remainder
    EXP = 0x06           # Exponentiation
    ADDMOD = 0x07        # (a + b) % N
    MULMOD = 0x08        # (a * b) % N
    SIGNEXTEND = 0x09    # Sign extend

    # ===== Comparison & Bitwise Logic (0x10 - 0x1F) =====
    LT = 0x10            # Less than
    GT = 0x11            # Greater than
    SLT = 0x12           # Signed less than
    SGT = 0x13           # Signed greater than
    EQ = 0x14            # Equality
    ISZERO = 0x15        # Is zero
    AND = 0x16           # Bitwise AND
    OR = 0x17            # Bitwise OR
    XOR = 0x18           # Bitwise XOR
    NOT = 0x19           # Bitwise NOT
    BYTE = 0x1A          # Retrieve single byte
    SHL = 0x1B           # Shift left
    SHR = 0x1C           # Shift right (logical)
    SAR = 0x1D           # Shift right (arithmetic)

    # ===== Cryptographic (0x20 - 0x2F) =====
    SHA256 = 0x20        # SHA-256 hash
    SHA512 = 0x21        # SHA-512 hash (primary Positronic hash)
    BLAKE2B = 0x22       # Blake2b-256 hash
    VERIFY_SIG = 0x23    # Ed25519 signature verification

    # ===== Context - Environment (0x30 - 0x3F) =====
    ADDRESS = 0x30       # Address of current contract
    BALANCE = 0x31       # Balance of given account
    ORIGIN = 0x32        # Transaction origin (original sender)
    CALLER = 0x33        # Direct caller address
    CALLVALUE = 0x34     # Value sent with call
    CALLDATALOAD = 0x35  # Load 32 bytes of call data
    CALLDATASIZE = 0x36  # Size of call data
    CALLDATACOPY = 0x37  # Copy call data to memory
    CODESIZE = 0x38      # Size of current contract code
    CODECOPY = 0x39      # Copy contract code to memory
    GASPRICE = 0x3A      # Gas price of transaction
    EXTCODESIZE = 0x3B   # Size of external contract code
    EXTCODECOPY = 0x3C   # Copy external contract code to memory

    # ===== Block Context (0x40 - 0x4F) =====
    BLOCKHASH = 0x40     # Hash of a recent block
    BLOCKHEIGHT = 0x41   # Current block number / height
    TIMESTAMP = 0x42     # Current block timestamp
    GASLIMIT = 0x43      # Block gas limit
    CHAINID = 0x44       # Chain ID (420420 for Positronic)

    # ===== Stack / Memory / Storage (0x50 - 0x5F) =====
    POP = 0x50           # Remove top stack item
    MLOAD = 0x51         # Load word from memory
    MSTORE = 0x52        # Store word to memory
    MSTORE8 = 0x53       # Store single byte to memory
    SLOAD = 0x54         # Load word from storage
    SSTORE = 0x55        # Store word to storage
    JUMP = 0x56          # Unconditional jump
    JUMPI = 0x57         # Conditional jump
    PC = 0x58            # Program counter
    MSIZE = 0x59         # Size of active memory
    GAS = 0x5A           # Remaining gas
    JUMPDEST = 0x5B      # Jump destination marker

    # ===== Push Operations (0x60 - 0x7F) =====
    PUSH1 = 0x60         # Push 1 byte
    PUSH2 = 0x61         # Push 2 bytes
    PUSH3 = 0x62         # Push 3 bytes
    PUSH4 = 0x63         # Push 4 bytes
    PUSH5 = 0x64         # Push 5 bytes
    PUSH6 = 0x65         # Push 6 bytes
    PUSH7 = 0x66         # Push 7 bytes
    PUSH8 = 0x67         # Push 8 bytes
    PUSH9 = 0x68         # Push 9 bytes
    PUSH10 = 0x69        # Push 10 bytes
    PUSH11 = 0x6A        # Push 11 bytes
    PUSH12 = 0x6B        # Push 12 bytes
    PUSH13 = 0x6C        # Push 13 bytes
    PUSH14 = 0x6D        # Push 14 bytes
    PUSH15 = 0x6E        # Push 15 bytes
    PUSH16 = 0x6F        # Push 16 bytes
    PUSH17 = 0x70        # Push 17 bytes
    PUSH18 = 0x71        # Push 18 bytes
    PUSH19 = 0x72        # Push 19 bytes
    PUSH20 = 0x73        # Push 20 bytes
    PUSH21 = 0x74        # Push 21 bytes
    PUSH22 = 0x75        # Push 22 bytes
    PUSH23 = 0x76        # Push 23 bytes
    PUSH24 = 0x77        # Push 24 bytes
    PUSH25 = 0x78        # Push 25 bytes
    PUSH26 = 0x79        # Push 26 bytes
    PUSH27 = 0x7A        # Push 27 bytes
    PUSH28 = 0x7B        # Push 28 bytes
    PUSH29 = 0x7C        # Push 29 bytes
    PUSH30 = 0x7D        # Push 30 bytes
    PUSH31 = 0x7E        # Push 31 bytes
    PUSH32 = 0x7F        # Push 32 bytes (full word)

    # ===== Dup Operations (0x80 - 0x8F) =====
    DUP1 = 0x80          # Duplicate 1st stack item
    DUP2 = 0x81          # Duplicate 2nd stack item
    DUP3 = 0x82          # Duplicate 3rd stack item
    DUP4 = 0x83          # Duplicate 4th stack item
    DUP5 = 0x84          # Duplicate 5th stack item
    DUP6 = 0x85          # Duplicate 6th stack item
    DUP7 = 0x86          # Duplicate 7th stack item
    DUP8 = 0x87          # Duplicate 8th stack item
    DUP9 = 0x88          # Duplicate 9th stack item
    DUP10 = 0x89         # Duplicate 10th stack item
    DUP11 = 0x8A         # Duplicate 11th stack item
    DUP12 = 0x8B         # Duplicate 12th stack item
    DUP13 = 0x8C         # Duplicate 13th stack item
    DUP14 = 0x8D         # Duplicate 14th stack item
    DUP15 = 0x8E         # Duplicate 15th stack item
    DUP16 = 0x8F         # Duplicate 16th stack item

    # ===== Swap Operations (0x90 - 0x9F) =====
    SWAP1 = 0x90         # Swap 1st and 2nd stack items
    SWAP2 = 0x91         # Swap 1st and 3rd stack items
    SWAP3 = 0x92         # Swap 1st and 4th stack items
    SWAP4 = 0x93         # Swap 1st and 5th stack items
    SWAP5 = 0x94         # Swap 1st and 6th stack items
    SWAP6 = 0x95         # Swap 1st and 7th stack items
    SWAP7 = 0x96         # Swap 1st and 8th stack items
    SWAP8 = 0x97         # Swap 1st and 9th stack items
    SWAP9 = 0x98         # Swap 1st and 10th stack items
    SWAP10 = 0x99        # Swap 1st and 11th stack items
    SWAP11 = 0x9A        # Swap 1st and 12th stack items
    SWAP12 = 0x9B        # Swap 1st and 13th stack items
    SWAP13 = 0x9C        # Swap 1st and 14th stack items
    SWAP14 = 0x9D        # Swap 1st and 15th stack items
    SWAP15 = 0x9E        # Swap 1st and 16th stack items
    SWAP16 = 0x9F        # Swap 1st and 17th stack items

    # ===== Logging (0xA0 - 0xA4) =====
    LOG0 = 0xA0          # Log with 0 topics
    LOG1 = 0xA1          # Log with 1 topic
    LOG2 = 0xA2          # Log with 2 topics
    LOG3 = 0xA3          # Log with 3 topics
    LOG4 = 0xA4          # Log with 4 topics

    # ===== AI-Specific Opcodes (0xB0 - 0xBF) =====
    AI_SCORE = 0xB0              # Get AI risk score of current transaction
    AI_QUARANTINE_STATUS = 0xB1  # Get quarantine status of a transaction

    # ===== Contract Operations (0xF0 - 0xFF) =====
    CREATE = 0xF0        # Create a new contract
    CALL = 0xF1          # Call another contract
    CALLCODE = 0xF2      # Call with current storage context
    RETURN = 0xF3        # Return output data
    DELEGATECALL = 0xF4  # Delegate call (preserves caller/value)
    CREATE2 = 0xF5       # Create with deterministic address
    STATICCALL = 0xF6    # Static call (read-only, no state changes)
    REVERT = 0xFD        # Revert execution with return data
    INVALID = 0xFE       # Invalid instruction (consumes all gas)
    SELFDESTRUCT = 0xFF  # Destroy contract and send balance


# ===== Gas Cost Table =====
# Maps each opcode to its base gas cost.

GAS_COSTS: dict[Opcode, int] = {
    # Stop / Arithmetic
    Opcode.STOP: 0,
    Opcode.ADD: 3,
    Opcode.SUB: 3,
    Opcode.MUL: 5,
    Opcode.DIV: 5,
    Opcode.MOD: 5,
    Opcode.EXP: 10,
    Opcode.ADDMOD: 8,
    Opcode.MULMOD: 8,
    Opcode.SIGNEXTEND: 5,

    # Comparison & Bitwise Logic
    Opcode.LT: 3,
    Opcode.GT: 3,
    Opcode.SLT: 3,
    Opcode.SGT: 3,
    Opcode.EQ: 3,
    Opcode.ISZERO: 3,
    Opcode.AND: 3,
    Opcode.OR: 3,
    Opcode.XOR: 3,
    Opcode.NOT: 3,
    Opcode.BYTE: 3,
    Opcode.SHL: 3,
    Opcode.SHR: 3,
    Opcode.SAR: 3,

    # Cryptographic
    Opcode.SHA256: 60,
    Opcode.SHA512: 80,
    Opcode.BLAKE2B: 40,
    Opcode.VERIFY_SIG: 3000,

    # Context - Environment
    Opcode.ADDRESS: 2,
    Opcode.BALANCE: 700,
    Opcode.ORIGIN: 2,
    Opcode.CALLER: 2,
    Opcode.CALLVALUE: 2,
    Opcode.CALLDATALOAD: 3,
    Opcode.CALLDATASIZE: 2,
    Opcode.CALLDATACOPY: 3,
    Opcode.CODESIZE: 2,
    Opcode.CODECOPY: 3,
    Opcode.GASPRICE: 2,
    Opcode.EXTCODESIZE: 700,
    Opcode.EXTCODECOPY: 700,

    # Block Context
    Opcode.BLOCKHASH: 20,
    Opcode.BLOCKHEIGHT: 2,
    Opcode.TIMESTAMP: 2,
    Opcode.GASLIMIT: 2,
    Opcode.CHAINID: 2,

    # Stack / Memory / Storage
    Opcode.POP: 2,
    Opcode.MLOAD: 3,
    Opcode.MSTORE: 3,
    Opcode.MSTORE8: 3,
    Opcode.SLOAD: 800,
    Opcode.SSTORE: 5000,       # 5000 for set, 20000 for first write (dynamic)
    Opcode.JUMP: 8,
    Opcode.JUMPI: 10,
    Opcode.PC: 2,
    Opcode.MSIZE: 2,
    Opcode.GAS: 2,
    Opcode.JUMPDEST: 1,

    # Push (all PUSH variants cost 3 gas)
    **{Opcode(0x60 + i): 3 for i in range(32)},

    # Dup (all DUP variants cost 3 gas)
    **{Opcode(0x80 + i): 3 for i in range(16)},

    # Swap (all SWAP variants cost 3 gas)
    **{Opcode(0x90 + i): 3 for i in range(16)},

    # Logging
    Opcode.LOG0: 375,
    Opcode.LOG1: 750,
    Opcode.LOG2: 1125,
    Opcode.LOG3: 1500,
    Opcode.LOG4: 1875,

    # AI-Specific
    Opcode.AI_SCORE: 100,
    Opcode.AI_QUARANTINE_STATUS: 100,

    # Contract Operations
    Opcode.CREATE: 32000,
    Opcode.CALL: 700,
    Opcode.CALLCODE: 700,
    Opcode.RETURN: 0,
    Opcode.DELEGATECALL: 700,
    Opcode.CREATE2: 32000,
    Opcode.STATICCALL: 700,
    Opcode.REVERT: 0,
    Opcode.INVALID: 0,          # Consumes ALL remaining gas
    Opcode.SELFDESTRUCT: 5000,
}

# Additional dynamic gas costs (applied on top of base costs)
SSTORE_SET_GAS = 20000       # First write to a storage slot
SSTORE_RESET_GAS = 5000      # Overwrite an existing slot
SSTORE_REFUND_GAS = 15000    # Refund for clearing a slot to zero

LOG_DATA_GAS = 8             # Per byte of log data
LOG_TOPIC_GAS = 375          # Per topic

MEMORY_GAS_WORD = 3          # Per 32-byte word of memory expansion
MEMORY_GAS_QUADRATIC = 1     # Quadratic coefficient for memory cost

CALL_VALUE_TRANSFER_GAS = 9000    # Extra gas for sending value with CALL
CALL_NEW_ACCOUNT_GAS = 25000      # Extra gas for calling non-existent account

EXP_BYTE_GAS = 50            # Per byte of exponent in EXP

COPY_GAS_PER_WORD = 3        # Per 32-byte word for copy operations

CREATE_DATA_GAS = 200        # Per byte of deployed contract code


def get_push_size(opcode: Opcode) -> int:
    """
    Get the number of immediate bytes for a PUSH opcode.
    Returns 0 if the opcode is not a PUSH instruction.
    """
    val = int(opcode)
    if 0x60 <= val <= 0x7F:
        return val - 0x5F  # PUSH1=1, PUSH2=2, ..., PUSH32=32
    return 0


def get_dup_depth(opcode: Opcode) -> int:
    """
    Get the stack depth for a DUP opcode.
    Returns 0 if the opcode is not a DUP instruction.
    DUP1 -> 1, DUP2 -> 2, ..., DUP16 -> 16
    """
    val = int(opcode)
    if 0x80 <= val <= 0x8F:
        return val - 0x7F
    return 0


def get_swap_depth(opcode: Opcode) -> int:
    """
    Get the stack depth for a SWAP opcode.
    Returns 0 if the opcode is not a SWAP instruction.
    SWAP1 -> 1, SWAP2 -> 2, ..., SWAP16 -> 16
    """
    val = int(opcode)
    if 0x90 <= val <= 0x9F:
        return val - 0x8F
    return 0


def get_log_topic_count(opcode: Opcode) -> int:
    """
    Get the number of topics for a LOG opcode.
    Returns -1 if the opcode is not a LOG instruction.
    """
    val = int(opcode)
    if 0xA0 <= val <= 0xA4:
        return val - 0xA0
    return -1


def is_push(opcode_val: int) -> bool:
    """Check if a raw byte value is a PUSH opcode."""
    return 0x60 <= opcode_val <= 0x7F


def is_dup(opcode_val: int) -> bool:
    """Check if a raw byte value is a DUP opcode."""
    return 0x80 <= opcode_val <= 0x8F


def is_swap(opcode_val: int) -> bool:
    """Check if a raw byte value is a SWAP opcode."""
    return 0x90 <= opcode_val <= 0x9F


def is_log(opcode_val: int) -> bool:
    """Check if a raw byte value is a LOG opcode."""
    return 0xA0 <= opcode_val <= 0xA4
