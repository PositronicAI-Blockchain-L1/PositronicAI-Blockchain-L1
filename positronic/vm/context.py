"""
Positronic - PositronicVM Execution Context

Carries all environmental data needed during contract execution:
message context (sender, value, data), block context (height, timestamp),
and transaction metadata.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from positronic.constants import CHAIN_ID, BLOCK_GAS_LIMIT


@dataclass
class LogEntry:
    """A log entry emitted during execution (LOG0-LOG4)."""
    address: bytes          # Contract that emitted the log
    topics: List[bytes]     # 0-4 indexed topics (each 32 bytes)
    data: bytes             # Unindexed log data


@dataclass
class MessageContext:
    """
    Message-level context for the current call frame.

    Corresponds to Solidity's msg.sender, msg.value, msg.data, etc.
    """
    sender: bytes           # msg.sender - immediate caller (20-byte address)
    value: int = 0          # msg.value - amount of ASF sent with this call
    data: bytes = b""       # msg.data - calldata (function selector + arguments)
    gas: int = 0            # Gas available for this call frame
    is_static: bool = False # True if this is a STATICCALL (no state changes)


@dataclass
class BlockContext:
    """
    Block-level context. Set once per block during execution.

    Corresponds to Solidity's block.number, block.timestamp, etc.
    """
    height: int = 0                 # block.number / block.height
    timestamp: int = 0              # block.timestamp (Unix epoch seconds)
    gas_limit: int = BLOCK_GAS_LIMIT  # block.gaslimit
    coinbase: bytes = b"\x00" * 20  # block.coinbase (validator address)
    chain_id: int = CHAIN_ID        # block.chainid
    difficulty: int = 0             # block.difficulty (unused in DPoS, kept for compat)

    # Recent block hashes (up to 256 most recent)
    _block_hashes: dict = field(default_factory=dict)

    def set_block_hash(self, block_number: int, block_hash: bytes) -> None:
        """
        Store a block hash for a given block number.

        Args:
            block_number: The block number.
            block_hash: The 64-byte SHA-512 hash of the block.
        """
        self._block_hashes[block_number] = block_hash

    def get_block_hash(self, block_number: int) -> bytes:
        """
        Get the hash of a recent block.
        Returns zero hash if block number is out of range (> 256 blocks old).

        Args:
            block_number: The block number to look up.

        Returns:
            64-byte block hash, or zero bytes if unavailable.
        """
        if block_number < 0 or block_number >= self.height:
            return b"\x00" * 64
        if self.height - block_number > 256:
            return b"\x00" * 64
        return self._block_hashes.get(block_number, b"\x00" * 64)


@dataclass
class TransactionContext:
    """
    Transaction-level context. Constant throughout the entire transaction.

    Corresponds to Solidity's tx.origin, tx.gasprice.
    """
    origin: bytes = b"\x00" * 20   # tx.origin - original external sender
    gas_price: int = 0             # tx.gasprice - gas price in base units
    tx_hash: bytes = b"\x00" * 64  # Hash of the transaction


class ExecutionContext:
    """
    Complete execution context for a PositronicVM call frame.

    Combines message, block, and transaction context. Supports
    nested call frames through parent linking. Tracks logs,
    return data, and execution state.

    Attributes:
        msg: Message context (sender, value, data)
        block: Block context (height, timestamp)
        tx: Transaction context (origin, gas price)
        contract_address: Address of the contract being executed
        code: Bytecode being executed
        logs: Log entries emitted during execution
        return_data: Return data from the last sub-call
        call_depth: Current nesting depth of calls
        parent: Parent execution context (for nested calls)
    """

    def __init__(
        self,
        msg: MessageContext,
        block: BlockContext,
        tx: TransactionContext,
        contract_address: bytes,
        code: bytes = b"",
        call_depth: int = 0,
        parent: Optional["ExecutionContext"] = None,
    ):
        self.msg = msg
        self.block = block
        self.tx = tx
        self.contract_address = contract_address
        self.code = code
        self.call_depth = call_depth
        self.parent = parent

        # Execution outputs
        self.logs: List[LogEntry] = []
        self.return_data: bytes = b""
        self.last_return_data: bytes = b""  # From most recent sub-call

        # Execution state
        self.reverted: bool = False
        self.stopped: bool = False
        self.selfdestruct_set: set = set()  # Addresses to destroy

    @property
    def caller(self) -> bytes:
        """Shortcut for msg.sender."""
        return self.msg.sender

    @property
    def callvalue(self) -> int:
        """Shortcut for msg.value."""
        return self.msg.value

    @property
    def calldata(self) -> bytes:
        """Shortcut for msg.data."""
        return self.msg.data

    @property
    def origin(self) -> bytes:
        """Shortcut for tx.origin."""
        return self.tx.origin

    @property
    def gas_price(self) -> int:
        """Shortcut for tx.gas_price."""
        return self.tx.gas_price

    @property
    def block_height(self) -> int:
        """Shortcut for block.height."""
        return self.block.height

    @property
    def timestamp(self) -> int:
        """Shortcut for block.timestamp."""
        return self.block.timestamp

    @property
    def is_static(self) -> bool:
        """Whether this is a static (read-only) call."""
        return self.msg.is_static

    def calldataload(self, offset: int) -> int:
        """
        Load 32 bytes of calldata starting at offset as a uint256.
        Pads with zeros if reading past the end of calldata.

        Args:
            offset: Byte offset into calldata.

        Returns:
            256-bit unsigned integer.
        """
        data = self.msg.data
        # Extract 32 bytes, zero-pad if needed
        chunk = data[offset:offset + 32]
        if len(chunk) < 32:
            chunk = chunk + b"\x00" * (32 - len(chunk))
        return int.from_bytes(chunk, "big")

    def add_log(self, topics: List[bytes], data: bytes) -> None:
        """
        Emit a log entry.

        Args:
            topics: List of 32-byte topics (0 to 4).
            data: Arbitrary-length log data.

        Raises:
            RuntimeError: If called in a static context.
        """
        if self.msg.is_static:
            raise RuntimeError("Cannot emit logs in static call")
        self.logs.append(LogEntry(
            address=self.contract_address,
            topics=topics,
            data=data,
        ))

    def create_sub_context(
        self,
        sender: bytes,
        to: bytes,
        value: int,
        data: bytes,
        gas: int,
        code: bytes,
        is_static: bool = False,
        is_delegatecall: bool = False,
    ) -> "ExecutionContext":
        """
        Create a sub-context for a CALL, DELEGATECALL, or CREATE.

        Args:
            sender: The caller address for the sub-context.
            to: The target contract address.
            value: Value to send.
            data: Calldata for the sub-call.
            gas: Gas allocated to the sub-call.
            code: Bytecode to execute.
            is_static: Whether this is a STATICCALL.
            is_delegatecall: Whether this is a DELEGATECALL.

        Returns:
            New ExecutionContext for the sub-call.
        """
        from positronic.constants import MAX_CALL_DEPTH

        new_depth = self.call_depth + 1
        if new_depth > MAX_CALL_DEPTH:
            raise RuntimeError(
                f"Call depth limit exceeded: {new_depth} > {MAX_CALL_DEPTH}"
            )

        msg = MessageContext(
            sender=sender,
            value=value,
            data=data,
            gas=gas,
            is_static=is_static or self.msg.is_static,  # Static propagates
        )

        # For DELEGATECALL, the contract address stays as the current one
        contract_addr = self.contract_address if is_delegatecall else to

        return ExecutionContext(
            msg=msg,
            block=self.block,
            tx=self.tx,
            contract_address=contract_addr,
            code=code,
            call_depth=new_depth,
            parent=self,
        )

    def collect_logs(self) -> List[LogEntry]:
        """
        Collect all log entries from this context and all child contexts.

        Returns:
            Combined list of log entries.
        """
        return list(self.logs)

    def __repr__(self) -> str:
        addr_hex = "0x" + self.contract_address.hex()[:10] + "..."
        caller_hex = "0x" + self.msg.sender.hex()[:10] + "..."
        return (
            f"ExecutionContext("
            f"contract={addr_hex}, "
            f"caller={caller_hex}, "
            f"value={self.msg.value}, "
            f"depth={self.call_depth}, "
            f"data_size={len(self.msg.data)}, "
            f"code_size={len(self.code)})"
        )
