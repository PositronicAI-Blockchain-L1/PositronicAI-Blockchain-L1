"""
Positronic - Smart Contract Virtual Machine (PositronicVM)

A stack-based virtual machine for executing smart contracts on the Positronic
blockchain. Inspired by the EVM architecture but extended with AI-specific
opcodes and quantum-resistant cryptographic primitives.

Key features:
    - Stack-based execution with 1024-depth stack
    - Byte-addressable expandable memory (up to 1MB)
    - Persistent contract storage bridged to world state
    - Gas metering for resource accounting
    - Sub-context support for CALL/CREATE/DELEGATECALL
    - Built-in AI opcodes (AI_SCORE, AI_QUARANTINE_STATUS)
    - Precompiled contracts for SHA-512, BLAKE2B, ECVERIFY, AI_QUERY
    - Simple assembler for text-to-bytecode compilation

VM Resource Limits (defined in constants.py):
    MAX_CALL_DEPTH   = 256      Max nested contract calls (prevents stack overflow)
    MAX_STACK_DEPTH  = 1024     Max items on the execution stack
    MAX_MEMORY       = 1 MB     Max expandable memory per execution context
    MAX_CODE_SIZE    = 64 KB    Max contract bytecode size at deployment
    TX_BASE_GAS      = 21,000   Base gas cost for simple transfer
    CREATE_GAS       = 53,000   Gas cost for contract creation
    BLOCK_GAS_LIMIT  = 30M      Default block gas limit (dynamically adjusted)

    These limits protect the network from resource exhaustion attacks
    while allowing rich smart contract functionality.

Modules:
    opcodes     - Opcode definitions (IntEnum) and gas cost table
    stack       - VMStack with push/pop/peek/dup/swap operations
    memory      - Expandable byte-addressable memory
    storage     - Contract storage interface bridging to StateManager
    gas         - GasMeter for gas consumption and refund tracking
    context     - ExecutionContext carrying msg/block/tx metadata
    vm          - Main PositronicVM execution engine with dispatch loop
    compiler    - Simple assembler: text assembly to bytecode
    precompiles - Precompiled contracts (SHA512, BLAKE2B, ECVERIFY, AI_QUERY)
"""

from positronic import __version__
