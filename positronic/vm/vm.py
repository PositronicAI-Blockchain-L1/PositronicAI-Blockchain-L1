"""
Positronic - PositronicVM Execution Engine

Main virtual machine that executes smart contract bytecode opcode-by-opcode.
Manages the dispatch loop, gas accounting, sub-context creation for CALL/CREATE,
and integrates all VM components (stack, memory, storage, gas meter, context).
"""

import struct
from typing import Optional, Tuple, List

from positronic.utils.logging import get_logger
from positronic.constants import (
    MAX_CALL_DEPTH,
    MAX_CODE_SIZE,
    CHAIN_ID,
    BLOCK_GAS_LIMIT,
    CREATE_GAS,
)
from positronic.core.state import StateManager
from positronic.crypto.hashing import sha512, sha256, blake2b_256
from positronic.crypto.keys import KeyPair

from positronic.vm.opcodes import (
    Opcode,
    GAS_COSTS,
    get_push_size,
    get_dup_depth,
    get_swap_depth,
    get_log_topic_count,
    is_push,
    is_dup,
    is_swap,
    is_log,
    SSTORE_SET_GAS,
    SSTORE_RESET_GAS,
    SSTORE_REFUND_GAS,
    LOG_DATA_GAS,
    LOG_TOPIC_GAS,
    MEMORY_GAS_WORD,
    CALL_VALUE_TRANSFER_GAS,
    CALL_NEW_ACCOUNT_GAS,
    EXP_BYTE_GAS,
    COPY_GAS_PER_WORD,
    CREATE_DATA_GAS,
)
from positronic.vm.stack import VMStack, StackOverflowError, StackUnderflowError
from positronic.vm.memory import VMMemory, MemoryLimitError
from positronic.vm.storage import ContractStorage
from positronic.vm.gas import GasMeter, OutOfGasError
from positronic.vm.context import (
    ExecutionContext,
    MessageContext,
    BlockContext,
    TransactionContext,
    LogEntry,
)
from positronic.vm.precompiles import PRECOMPILED_CONTRACTS, is_precompile

logger = get_logger(__name__)


# ===== Result types =====

class ExecutionResult:
    """Result of a PositronicVM execution."""

    def __init__(
        self,
        success: bool,
        return_data: bytes = b"",
        gas_used: int = 0,
        gas_refund: int = 0,
        logs: Optional[List[LogEntry]] = None,
        error: str = "",
        reverted: bool = False,
    ):
        self.success = success
        self.return_data = return_data
        self.gas_used = gas_used
        self.gas_refund = gas_refund
        self.logs = logs or []
        self.error = error
        self.reverted = reverted

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else ("REVERT" if self.reverted else "FAIL")
        return (
            f"ExecutionResult({status}, gas_used={self.gas_used}, "
            f"return_size={len(self.return_data)}, logs={len(self.logs)})"
        )


class VMError(Exception):
    """Base class for VM execution errors."""
    pass


class InvalidJumpError(VMError):
    """Jump to a non-JUMPDEST location."""
    pass


class WriteInStaticCallError(VMError):
    """Attempt to modify state in a STATICCALL."""
    pass


# ===== Helper: 256-bit signed/unsigned conversion =====

UINT256_MAX = (1 << 256) - 1
UINT256_CEIL = 1 << 256
INT256_MAX = (1 << 255) - 1
INT256_MIN = -(1 << 255)


def to_signed(value: int) -> int:
    """Convert uint256 to signed int256 (two's complement)."""
    if value > INT256_MAX:
        return value - UINT256_CEIL
    return value


def to_unsigned(value: int) -> int:
    """Convert signed int256 to uint256 (two's complement)."""
    if value < 0:
        return value + UINT256_CEIL
    return value & UINT256_MAX


# ===== Main VM =====

class PositronicVM:
    """
    PositronicVM - Smart Contract Execution Engine for Positronic.

    Executes EVM-style bytecode with extensions for AI opcodes and
    Positronic-specific cryptographic primitives.

    Usage:
        state = StateManager()
        vm = PositronicVM(state)
        result = vm.execute(context)
    """

    def __init__(self, state: StateManager):
        """
        Initialize the VM with a state manager.

        Args:
            state: The global StateManager for reading/writing account state.
        """
        self.state = state

    def execute(self, ctx: ExecutionContext) -> ExecutionResult:
        """
        Execute bytecode in the given execution context.

        This is the main entry point. It sets up the stack, memory, storage,
        and gas meter, then runs the dispatch loop.

        Args:
            ctx: The execution context with bytecode, calldata, and environment.

        Returns:
            ExecutionResult with success/failure, return data, gas used, and logs.
        """
        stack = VMStack()
        memory = VMMemory()
        storage = ContractStorage(self.state, ctx.contract_address)
        gas = GasMeter(ctx.msg.gas)

        # Pre-scan for valid JUMPDEST positions
        valid_jumpdests = self._scan_jumpdests(ctx.code)

        pc = 0  # Program counter
        code = ctx.code
        code_len = len(code)

        try:
            while pc < code_len:
                if ctx.stopped or ctx.reverted:
                    break

                # Fetch opcode
                op_byte = code[pc]

                try:
                    op = Opcode(op_byte)
                except ValueError:
                    # Unknown opcode -> treat as INVALID
                    gas.consume_all()
                    raise VMError(f"Invalid opcode 0x{op_byte:02x} at PC={pc}")

                # Consume base gas
                base_gas = GAS_COSTS.get(op, 0)
                if op == Opcode.INVALID:
                    gas.consume_all()
                    raise VMError(f"INVALID opcode at PC={pc}")

                gas.consume(base_gas)

                # Dispatch
                pc = self._dispatch(
                    op, pc, stack, memory, storage, gas, ctx, valid_jumpdests
                )

            # Execution completed normally
            storage.commit()
            refund = storage.calculate_refund() + gas.gas_refund

            return ExecutionResult(
                success=not ctx.reverted,
                return_data=ctx.return_data,
                gas_used=gas.effective_gas_used,
                gas_refund=refund,
                logs=ctx.logs if not ctx.reverted else [],
                reverted=ctx.reverted,
            )

        except OutOfGasError as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Out of gas: {e}",
            )
        except (StackOverflowError, StackUnderflowError) as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Stack error: {e}",
            )
        except MemoryLimitError as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Memory error: {e}",
            )
        except WriteInStaticCallError as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Static call violation: {e}",
            )
        except InvalidJumpError as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Invalid jump: {e}",
            )
        except VMError as e:
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=str(e),
            )
        except Exception as e:
            logger.error("internal_vm_error: %s", e)
            storage.revert()
            return ExecutionResult(
                success=False,
                gas_used=gas.gas_limit,
                error=f"Internal VM error: {e}",
            )

    def _scan_jumpdests(self, code: bytes) -> set:
        """
        Pre-scan bytecode to find all valid JUMPDEST positions.
        Skips over PUSH immediate data.

        Args:
            code: Contract bytecode.

        Returns:
            Set of valid jump destination PCs.
        """
        jumpdests = set()
        i = 0
        while i < len(code):
            op = code[i]
            if op == Opcode.JUMPDEST:
                jumpdests.add(i)
            if is_push(op):
                # Skip over push data bytes
                push_size = op - 0x5F
                i += push_size
            i += 1
        return jumpdests

    def _dispatch(
        self,
        op: Opcode,
        pc: int,
        stack: VMStack,
        memory: VMMemory,
        storage: ContractStorage,
        gas: GasMeter,
        ctx: ExecutionContext,
        valid_jumpdests: set,
    ) -> int:
        """
        Dispatch a single opcode. Returns the next PC value.

        Args:
            op: The opcode to execute.
            pc: Current program counter.
            stack: VM stack.
            memory: VM memory.
            storage: Contract storage.
            gas: Gas meter.
            ctx: Execution context.
            valid_jumpdests: Set of valid JUMPDEST positions.

        Returns:
            Next program counter value.
        """
        code = ctx.code

        # ===== STOP =====
        if op == Opcode.STOP:
            ctx.stopped = True
            return pc + 1

        # ===== ARITHMETIC =====
        elif op == Opcode.ADD:
            a, b = stack.pop(), stack.pop()
            stack.push((a + b) & UINT256_MAX)

        elif op == Opcode.SUB:
            a, b = stack.pop(), stack.pop()
            stack.push((a - b) & UINT256_MAX)

        elif op == Opcode.MUL:
            a, b = stack.pop(), stack.pop()
            stack.push((a * b) & UINT256_MAX)

        elif op == Opcode.DIV:
            a, b = stack.pop(), stack.pop()
            stack.push(a // b if b != 0 else 0)

        elif op == Opcode.MOD:
            a, b = stack.pop(), stack.pop()
            stack.push(a % b if b != 0 else 0)

        elif op == Opcode.EXP:
            base, exponent = stack.pop(), stack.pop()
            # Dynamic gas: 50 per byte of exponent
            if exponent > 0:
                exp_bytes = (exponent.bit_length() + 7) // 8
                gas.consume(EXP_BYTE_GAS * exp_bytes)
            stack.push(pow(base, exponent, UINT256_CEIL) & UINT256_MAX)

        elif op == Opcode.ADDMOD:
            a, b, n = stack.pop(), stack.pop(), stack.pop()
            stack.push((a + b) % n if n != 0 else 0)

        elif op == Opcode.MULMOD:
            a, b, n = stack.pop(), stack.pop(), stack.pop()
            stack.push((a * b) % n if n != 0 else 0)

        elif op == Opcode.SIGNEXTEND:
            b, x = stack.pop(), stack.pop()
            if b < 31:
                bit_index = b * 8 + 7
                mask = (1 << bit_index) - 1
                if x & (1 << bit_index):
                    stack.push(x | (UINT256_MAX - mask))
                else:
                    stack.push(x & mask)
            else:
                stack.push(x)

        # ===== COMPARISON & BITWISE LOGIC =====
        elif op == Opcode.LT:
            a, b = stack.pop(), stack.pop()
            stack.push(1 if a < b else 0)

        elif op == Opcode.GT:
            a, b = stack.pop(), stack.pop()
            stack.push(1 if a > b else 0)

        elif op == Opcode.SLT:
            a, b = stack.pop(), stack.pop()
            stack.push(1 if to_signed(a) < to_signed(b) else 0)

        elif op == Opcode.SGT:
            a, b = stack.pop(), stack.pop()
            stack.push(1 if to_signed(a) > to_signed(b) else 0)

        elif op == Opcode.EQ:
            a, b = stack.pop(), stack.pop()
            stack.push(1 if a == b else 0)

        elif op == Opcode.ISZERO:
            a = stack.pop()
            stack.push(1 if a == 0 else 0)

        elif op == Opcode.AND:
            a, b = stack.pop(), stack.pop()
            stack.push(a & b)

        elif op == Opcode.OR:
            a, b = stack.pop(), stack.pop()
            stack.push(a | b)

        elif op == Opcode.XOR:
            a, b = stack.pop(), stack.pop()
            stack.push(a ^ b)

        elif op == Opcode.NOT:
            a = stack.pop()
            stack.push(UINT256_MAX ^ a)

        elif op == Opcode.BYTE:
            i, x = stack.pop(), stack.pop()
            if i < 32:
                stack.push((x >> (248 - i * 8)) & 0xFF)
            else:
                stack.push(0)

        elif op == Opcode.SHL:
            shift, value = stack.pop(), stack.pop()
            if shift >= 256:
                stack.push(0)
            else:
                stack.push((value << shift) & UINT256_MAX)

        elif op == Opcode.SHR:
            shift, value = stack.pop(), stack.pop()
            if shift >= 256:
                stack.push(0)
            else:
                stack.push(value >> shift)

        elif op == Opcode.SAR:
            shift, value = stack.pop(), stack.pop()
            signed_val = to_signed(value)
            if shift >= 256:
                stack.push(UINT256_MAX if signed_val < 0 else 0)
            else:
                result = signed_val >> shift
                stack.push(to_unsigned(result))

        # ===== CRYPTOGRAPHIC =====
        elif op == Opcode.SHA256:
            offset, length = stack.pop(), stack.pop()
            mem_cost = memory.expansion_cost(offset, length)
            gas.consume(mem_cost)
            data = memory.load_bytes(offset, length)
            hash_result = sha256(data)
            stack.push(int.from_bytes(hash_result, "big"))

        elif op == Opcode.SHA512:
            offset, length = stack.pop(), stack.pop()
            mem_cost = memory.expansion_cost(offset, length)
            gas.consume(mem_cost)
            data = memory.load_bytes(offset, length)
            hash_result = sha512(data)
            # Push first 32 bytes (256 bits) of the 64-byte hash
            stack.push(int.from_bytes(hash_result[:32], "big"))

        elif op == Opcode.BLAKE2B:
            offset, length = stack.pop(), stack.pop()
            mem_cost = memory.expansion_cost(offset, length)
            gas.consume(mem_cost)
            data = memory.load_bytes(offset, length)
            hash_result = blake2b_256(data)
            stack.push(int.from_bytes(hash_result, "big"))

        elif op == Opcode.VERIFY_SIG:
            # Stack: pubkey_offset, sig_offset, msg_offset, msg_length
            pk_off = stack.pop()
            sig_off = stack.pop()
            msg_off = stack.pop()
            msg_len = stack.pop()
            # Memory costs
            gas.consume(memory.expansion_cost(pk_off, 32))
            gas.consume(memory.expansion_cost(sig_off, 64))
            gas.consume(memory.expansion_cost(msg_off, msg_len))
            pubkey = memory.load_bytes(pk_off, 32)
            signature = memory.load_bytes(sig_off, 64)
            message = memory.load_bytes(msg_off, msg_len)
            valid = KeyPair.verify(pubkey, signature, message)
            stack.push(1 if valid else 0)

        # ===== CONTEXT - ENVIRONMENT =====
        elif op == Opcode.ADDRESS:
            addr_int = int.from_bytes(ctx.contract_address.ljust(32, b"\x00"), "big")
            stack.push(addr_int)

        elif op == Opcode.BALANCE:
            addr_val = stack.pop()
            addr_bytes = (addr_val & ((1 << 160) - 1)).to_bytes(20, "big")
            balance = self.state.get_balance(addr_bytes)
            stack.push(balance)

        elif op == Opcode.ORIGIN:
            origin_int = int.from_bytes(ctx.tx.origin.ljust(32, b"\x00"), "big")
            stack.push(origin_int)

        elif op == Opcode.CALLER:
            caller_int = int.from_bytes(ctx.msg.sender.ljust(32, b"\x00"), "big")
            stack.push(caller_int)

        elif op == Opcode.CALLVALUE:
            stack.push(ctx.msg.value)

        elif op == Opcode.CALLDATALOAD:
            offset = stack.pop()
            stack.push(ctx.calldataload(offset))

        elif op == Opcode.CALLDATASIZE:
            stack.push(len(ctx.msg.data))

        elif op == Opcode.CALLDATACOPY:
            mem_offset = stack.pop()
            data_offset = stack.pop()
            length = stack.pop()
            gas.consume(memory.expansion_cost(mem_offset, length))
            words = (length + 31) // 32
            gas.consume(COPY_GAS_PER_WORD * words)
            # Copy calldata to memory, zero-pad if needed
            calldata = ctx.msg.data
            src = calldata[data_offset:data_offset + length]
            if len(src) < length:
                src = src + b"\x00" * (length - len(src))
            memory.store_bytes(mem_offset, src)

        elif op == Opcode.CODESIZE:
            stack.push(len(ctx.code))

        elif op == Opcode.CODECOPY:
            mem_offset = stack.pop()
            code_offset = stack.pop()
            length = stack.pop()
            gas.consume(memory.expansion_cost(mem_offset, length))
            words = (length + 31) // 32
            gas.consume(COPY_GAS_PER_WORD * words)
            src = ctx.code[code_offset:code_offset + length]
            if len(src) < length:
                src = src + b"\x00" * (length - len(src))
            memory.store_bytes(mem_offset, src)

        elif op == Opcode.GASPRICE:
            stack.push(ctx.tx.gas_price)

        elif op == Opcode.EXTCODESIZE:
            addr_val = stack.pop()
            addr_bytes = (addr_val & ((1 << 160) - 1)).to_bytes(20, "big")
            ext_code = self.state.get_code(addr_bytes)
            stack.push(len(ext_code))

        elif op == Opcode.EXTCODECOPY:
            addr_val = stack.pop()
            mem_offset = stack.pop()
            code_offset = stack.pop()
            length = stack.pop()
            gas.consume(memory.expansion_cost(mem_offset, length))
            words = (length + 31) // 32
            gas.consume(COPY_GAS_PER_WORD * words)
            addr_bytes = (addr_val & ((1 << 160) - 1)).to_bytes(20, "big")
            ext_code = self.state.get_code(addr_bytes)
            src = ext_code[code_offset:code_offset + length]
            if len(src) < length:
                src = src + b"\x00" * (length - len(src))
            memory.store_bytes(mem_offset, src)

        # ===== BLOCK CONTEXT =====
        elif op == Opcode.BLOCKHASH:
            block_num = stack.pop()
            bh = ctx.block.get_block_hash(block_num)
            stack.push(int.from_bytes(bh[:32], "big"))

        elif op == Opcode.BLOCKHEIGHT:
            stack.push(ctx.block.height)

        elif op == Opcode.TIMESTAMP:
            stack.push(ctx.block.timestamp)

        elif op == Opcode.GASLIMIT:
            stack.push(ctx.block.gas_limit)

        elif op == Opcode.CHAINID:
            stack.push(ctx.block.chain_id)

        # ===== STACK / MEMORY / STORAGE =====
        elif op == Opcode.POP:
            stack.pop()

        elif op == Opcode.MLOAD:
            offset = stack.pop()
            gas.consume(memory.expansion_cost(offset, 32))
            memory._expand_to(offset, 32)
            stack.push(memory.load(offset))

        elif op == Opcode.MSTORE:
            offset = stack.pop()
            value = stack.pop()
            gas.consume(memory.expansion_cost(offset, 32))
            memory.store(offset, value)

        elif op == Opcode.MSTORE8:
            offset = stack.pop()
            value = stack.pop()
            gas.consume(memory.expansion_cost(offset, 1))
            memory.store8(offset, value)

        elif op == Opcode.SLOAD:
            key = stack.pop()
            value = storage.load(key)
            stack.push(value)

        elif op == Opcode.SSTORE:
            if ctx.is_static:
                raise WriteInStaticCallError("SSTORE in static call")
            key = stack.pop()
            value = stack.pop()
            # Dynamic gas: check original value for set vs reset pricing
            original = storage.get_original_value(key)
            current = storage.load(key)
            zero = 0
            if current == value:
                # No-op, minimal gas (already charged base)
                pass
            elif current == zero:
                # Setting a fresh slot
                gas.consume(SSTORE_SET_GAS - GAS_COSTS[Opcode.SSTORE])
            else:
                if value == zero:
                    # Clearing a slot - schedule refund
                    gas.add_refund(SSTORE_REFUND_GAS)
                # Reset cost already covered by base gas
            storage.store(key, value)

        elif op == Opcode.JUMP:
            dest = stack.pop()
            if dest not in valid_jumpdests:
                raise InvalidJumpError(f"Invalid jump destination: {dest}")
            return dest  # Jump directly to destination

        elif op == Opcode.JUMPI:
            dest = stack.pop()
            condition = stack.pop()
            if condition != 0:
                if dest not in valid_jumpdests:
                    raise InvalidJumpError(f"Invalid jump destination: {dest}")
                return dest
            # Condition false, fall through

        elif op == Opcode.PC:
            stack.push(pc)

        elif op == Opcode.MSIZE:
            stack.push(memory.size)

        elif op == Opcode.GAS:
            stack.push(gas.gas_remaining)

        elif op == Opcode.JUMPDEST:
            pass  # No-op marker

        # ===== PUSH =====
        elif is_push(int(op)):
            num_bytes = get_push_size(op)
            # Read immediate data from bytecode
            start = pc + 1
            end = start + num_bytes
            if end > len(code):
                # Pad with zeros if bytecode is too short
                push_data = code[start:] + b"\x00" * (end - len(code))
            else:
                push_data = code[start:end]
            value = int.from_bytes(push_data, "big")
            stack.push(value)
            return end  # Skip over push data

        # ===== DUP =====
        elif is_dup(int(op)):
            depth = get_dup_depth(op)
            stack.dup(depth)

        # ===== SWAP =====
        elif is_swap(int(op)):
            depth = get_swap_depth(op)
            stack.swap(depth)

        # ===== LOGGING =====
        elif is_log(int(op)):
            if ctx.is_static:
                raise WriteInStaticCallError("LOG in static call")
            topic_count = get_log_topic_count(op)
            offset = stack.pop()
            length = stack.pop()
            topics = [stack.pop().to_bytes(32, "big") for _ in range(topic_count)]
            # Dynamic gas
            gas.consume(memory.expansion_cost(offset, length))
            gas.consume(LOG_DATA_GAS * length)
            gas.consume(LOG_TOPIC_GAS * topic_count)
            data = memory.load_bytes(offset, length)
            ctx.add_log(topics, data)

        # ===== AI-SPECIFIC =====
        elif op == Opcode.AI_SCORE:
            # Push the AI risk score of the current transaction (scaled to uint256)
            # Score is float 0.0-1.0, scale to 0-10000 for integer representation
            score_scaled = int(getattr(ctx, '_ai_score', 0.0) * 10000)
            stack.push(score_scaled)

        elif op == Opcode.AI_QUARANTINE_STATUS:
            # Stack input: tx_hash (as uint256)
            tx_hash_int = stack.pop()
            # Return quarantine status: 0=not quarantined, 1=quarantined, 2=released, 3=expired
            status = getattr(ctx, '_quarantine_status_fn', lambda _: 0)(tx_hash_int)
            stack.push(status)

        # ===== CONTRACT OPERATIONS =====
        elif op == Opcode.CREATE:
            if ctx.is_static:
                raise WriteInStaticCallError("CREATE in static call")
            result_addr = self._handle_create(stack, memory, gas, ctx, storage)
            stack.push(result_addr)

        elif op == Opcode.CREATE2:
            if ctx.is_static:
                raise WriteInStaticCallError("CREATE2 in static call")
            result_addr = self._handle_create2(stack, memory, gas, ctx, storage)
            stack.push(result_addr)

        elif op == Opcode.CALL:
            success = self._handle_call(
                stack, memory, gas, ctx, storage, is_delegate=False, is_static=False
            )
            stack.push(1 if success else 0)

        elif op == Opcode.CALLCODE:
            success = self._handle_call(
                stack, memory, gas, ctx, storage, is_delegate=True, is_static=False
            )
            stack.push(1 if success else 0)

        elif op == Opcode.DELEGATECALL:
            success = self._handle_delegatecall(stack, memory, gas, ctx, storage)
            stack.push(1 if success else 0)

        elif op == Opcode.STATICCALL:
            success = self._handle_call(
                stack, memory, gas, ctx, storage, is_delegate=False, is_static=True
            )
            stack.push(1 if success else 0)

        elif op == Opcode.RETURN:
            offset = stack.pop()
            length = stack.pop()
            gas.consume(memory.expansion_cost(offset, length))
            ctx.return_data = memory.load_bytes(offset, length)
            ctx.stopped = True

        elif op == Opcode.REVERT:
            offset = stack.pop()
            length = stack.pop()
            gas.consume(memory.expansion_cost(offset, length))
            ctx.return_data = memory.load_bytes(offset, length)
            ctx.reverted = True
            ctx.stopped = True

        elif op == Opcode.SELFDESTRUCT:
            if ctx.is_static:
                raise WriteInStaticCallError("SELFDESTRUCT in static call")
            beneficiary_val = stack.pop()
            beneficiary = (beneficiary_val & ((1 << 160) - 1)).to_bytes(20, "big")
            # Transfer balance to beneficiary
            balance = self.state.get_balance(ctx.contract_address)
            if balance > 0:
                if not self.state.account_exists(beneficiary):
                    gas.consume(CALL_NEW_ACCOUNT_GAS)
                self.state.add_balance(beneficiary, balance)
                self.state.sub_balance(ctx.contract_address, balance)
            ctx.selfdestruct_set.add(ctx.contract_address)
            ctx.stopped = True

        else:
            raise VMError(f"Unhandled opcode: {op.name} (0x{int(op):02x}) at PC={pc}")

        return pc + 1

    # ===== CALL/CREATE handlers =====

    def _handle_call(
        self,
        stack: VMStack,
        memory: VMMemory,
        gas: GasMeter,
        ctx: ExecutionContext,
        storage: ContractStorage,
        is_delegate: bool,
        is_static: bool,
    ) -> bool:
        """
        Handle CALL, CALLCODE, and STATICCALL opcodes.

        Args:
            stack: VM stack.
            memory: VM memory.
            gas: Gas meter.
            ctx: Current execution context.
            storage: Current contract storage.
            is_delegate: True for CALLCODE (execute code in current storage context).
            is_static: True for STATICCALL (no state modifications).

        Returns:
            True on success, False on failure.
        """
        call_gas = stack.pop()
        to_val = stack.pop()
        value = stack.pop()
        args_offset = stack.pop()
        args_length = stack.pop()
        ret_offset = stack.pop()
        ret_length = stack.pop()

        to_addr = (to_val & ((1 << 160) - 1)).to_bytes(20, "big")

        if is_static and value > 0:
            raise WriteInStaticCallError("Value transfer in static call")

        # Memory expansion costs for args and return data
        gas.consume(memory.expansion_cost(args_offset, args_length))
        gas.consume(memory.expansion_cost(ret_offset, ret_length))

        # Extra gas for value transfer
        extra_gas = 0
        if value > 0:
            extra_gas += CALL_VALUE_TRANSFER_GAS
            if not self.state.account_exists(to_addr):
                extra_gas += CALL_NEW_ACCOUNT_GAS
        gas.consume(extra_gas)

        # Calculate gas to forward (63/64 rule)
        forwarded_gas = gas.gas_for_call(call_gas)
        gas.consume(forwarded_gas)

        # Read calldata from memory
        calldata = memory.load_bytes(args_offset, args_length)

        # Check for precompiled contracts
        if is_precompile(to_addr):
            precompile_fn = PRECOMPILED_CONTRACTS[to_addr]
            try:
                result_data, precompile_gas = precompile_fn(calldata)
                if precompile_gas > forwarded_gas:
                    gas.return_gas(forwarded_gas)
                    ctx.last_return_data = b""
                    return False
                gas.return_gas(forwarded_gas - precompile_gas)
                # Copy return data to memory
                actual_ret = result_data[:ret_length]
                if actual_ret:
                    memory.store_bytes(ret_offset, actual_ret)
                ctx.last_return_data = result_data
                return True
            except Exception as e:
                logger.debug("Precompile execution failed: %s", e)
                gas.return_gas(forwarded_gas)
                ctx.last_return_data = b""
                return False

        # Transfer value
        if value > 0:
            if not self.state.transfer(ctx.contract_address, to_addr, value):
                gas.return_gas(forwarded_gas)
                ctx.last_return_data = b""
                return False

        # Get callee code
        callee_code = self.state.get_code(to_addr)
        if not callee_code:
            # No code at target -- call succeeds with no execution
            gas.return_gas(forwarded_gas)
            ctx.last_return_data = b""
            return True

        # Determine addresses for sub-context
        if is_delegate:
            # CALLCODE: execute callee code but in current contract's storage context
            sender = ctx.msg.sender
            contract_addr = ctx.contract_address
        else:
            sender = ctx.contract_address
            contract_addr = to_addr

        # Create sub-context
        try:
            sub_ctx = ctx.create_sub_context(
                sender=sender,
                to=contract_addr,
                value=value,
                data=calldata,
                gas=forwarded_gas,
                code=callee_code,
                is_static=is_static,
            )
            sub_ctx.contract_address = contract_addr
        except RuntimeError:
            # Call depth exceeded
            gas.return_gas(forwarded_gas)
            ctx.last_return_data = b""
            return False

        # Execute sub-context
        result = self.execute(sub_ctx)

        # Return unused gas
        gas.return_gas(forwarded_gas - result.gas_used)

        # Store return data
        ctx.last_return_data = result.return_data

        # Copy return data to memory
        actual_ret = result.return_data[:ret_length]
        if actual_ret:
            memory.store_bytes(ret_offset, actual_ret)

        # Merge logs on success
        if result.success:
            ctx.logs.extend(result.logs)

        return result.success

    def _handle_delegatecall(
        self,
        stack: VMStack,
        memory: VMMemory,
        gas: GasMeter,
        ctx: ExecutionContext,
        storage: ContractStorage,
    ) -> bool:
        """
        Handle DELEGATECALL opcode.
        Like CALL but preserves msg.sender and msg.value from the current context.

        Returns:
            True on success, False on failure.
        """
        call_gas = stack.pop()
        to_val = stack.pop()
        args_offset = stack.pop()
        args_length = stack.pop()
        ret_offset = stack.pop()
        ret_length = stack.pop()

        to_addr = (to_val & ((1 << 160) - 1)).to_bytes(20, "big")

        # Memory expansion costs
        gas.consume(memory.expansion_cost(args_offset, args_length))
        gas.consume(memory.expansion_cost(ret_offset, ret_length))

        # Calculate gas to forward (63/64 rule)
        forwarded_gas = gas.gas_for_call(call_gas)
        gas.consume(forwarded_gas)

        # Read calldata from memory
        calldata = memory.load_bytes(args_offset, args_length)

        # Get code from target address (but execute in current context)
        callee_code = self.state.get_code(to_addr)
        if not callee_code:
            gas.return_gas(forwarded_gas)
            ctx.last_return_data = b""
            return True

        # Create sub-context preserving sender and value
        try:
            sub_ctx = ctx.create_sub_context(
                sender=ctx.msg.sender,      # Preserve original sender
                to=ctx.contract_address,    # Execute in current contract's context
                value=ctx.msg.value,         # Preserve original value
                data=calldata,
                gas=forwarded_gas,
                code=callee_code,
                is_delegatecall=True,
            )
        except RuntimeError:
            gas.return_gas(forwarded_gas)
            ctx.last_return_data = b""
            return False

        # Execute
        result = self.execute(sub_ctx)

        # Return unused gas
        gas.return_gas(forwarded_gas - result.gas_used)

        ctx.last_return_data = result.return_data

        # Copy return data to memory
        actual_ret = result.return_data[:ret_length]
        if actual_ret:
            memory.store_bytes(ret_offset, actual_ret)

        if result.success:
            ctx.logs.extend(result.logs)

        return result.success

    def _handle_create(
        self,
        stack: VMStack,
        memory: VMMemory,
        gas: GasMeter,
        ctx: ExecutionContext,
        storage: ContractStorage,
    ) -> int:
        """
        Handle CREATE opcode: deploy a new contract.

        Stack inputs: value, offset, length
        Returns: address of created contract as uint256 (0 on failure).
        """
        value = stack.pop()
        offset = stack.pop()
        length = stack.pop()

        gas.consume(memory.expansion_cost(offset, length))

        # Read init code from memory
        init_code = memory.load_bytes(offset, length)

        if len(init_code) > MAX_CODE_SIZE:
            return 0

        # Compute new contract address: hash(sender, nonce)
        sender_nonce = self.state.get_nonce(ctx.contract_address)
        addr_data = ctx.contract_address + sender_nonce.to_bytes(8, "big")
        new_addr = sha512(addr_data)[:20]

        # Increment nonce
        self.state.increment_nonce(ctx.contract_address)

        # Transfer value
        if value > 0:
            if not self.state.transfer(ctx.contract_address, new_addr, value):
                return 0

        # Calculate gas for sub-execution
        forwarded_gas = gas.gas_for_call(gas.gas_remaining)
        gas.consume(forwarded_gas)

        # Create sub-context to run init code
        try:
            sub_ctx = ctx.create_sub_context(
                sender=ctx.contract_address,
                to=new_addr,
                value=value,
                data=b"",
                gas=forwarded_gas,
                code=init_code,
            )
            sub_ctx.contract_address = new_addr
        except RuntimeError:
            gas.return_gas(forwarded_gas)
            return 0

        # Execute init code
        result = self.execute(sub_ctx)

        gas.return_gas(forwarded_gas - result.gas_used)

        if not result.success:
            return 0

        # Deploy the returned bytecode
        deployed_code = result.return_data
        if len(deployed_code) > MAX_CODE_SIZE:
            return 0

        # Charge per-byte deployment gas
        deploy_gas = CREATE_DATA_GAS * len(deployed_code)
        try:
            gas.consume(deploy_gas)
        except OutOfGasError:
            return 0

        self.state.deploy_contract(new_addr, deployed_code)

        # Merge logs
        ctx.logs.extend(result.logs)

        return int.from_bytes(new_addr.ljust(32, b"\x00"), "big")

    def _handle_create2(
        self,
        stack: VMStack,
        memory: VMMemory,
        gas: GasMeter,
        ctx: ExecutionContext,
        storage: ContractStorage,
    ) -> int:
        """
        Handle CREATE2 opcode: deploy with deterministic address.

        Stack inputs: value, offset, length, salt
        Address = hash(0xff + sender + salt + hash(init_code))[:20]
        Returns: address of created contract as uint256 (0 on failure).
        """
        value = stack.pop()
        offset = stack.pop()
        length = stack.pop()
        salt = stack.pop()

        gas.consume(memory.expansion_cost(offset, length))

        # Read init code from memory
        init_code = memory.load_bytes(offset, length)

        if len(init_code) > MAX_CODE_SIZE:
            return 0

        # Hash init code for address computation
        words = (length + 31) // 32
        gas.consume(COPY_GAS_PER_WORD * words)

        # Compute CREATE2 address
        salt_bytes = salt.to_bytes(32, "big")
        init_code_hash = sha512(init_code)
        addr_data = b"\xff" + ctx.contract_address + salt_bytes + init_code_hash
        new_addr = sha512(addr_data)[:20]

        # Check if address already has code
        if self.state.get_code(new_addr):
            return 0

        # Increment nonce
        self.state.increment_nonce(ctx.contract_address)

        # Transfer value
        if value > 0:
            if not self.state.transfer(ctx.contract_address, new_addr, value):
                return 0

        # Calculate gas for sub-execution
        forwarded_gas = gas.gas_for_call(gas.gas_remaining)
        gas.consume(forwarded_gas)

        # Create sub-context
        try:
            sub_ctx = ctx.create_sub_context(
                sender=ctx.contract_address,
                to=new_addr,
                value=value,
                data=b"",
                gas=forwarded_gas,
                code=init_code,
            )
            sub_ctx.contract_address = new_addr
        except RuntimeError:
            gas.return_gas(forwarded_gas)
            return 0

        # Execute init code
        result = self.execute(sub_ctx)

        gas.return_gas(forwarded_gas - result.gas_used)

        if not result.success:
            return 0

        deployed_code = result.return_data
        if len(deployed_code) > MAX_CODE_SIZE:
            return 0

        deploy_gas = CREATE_DATA_GAS * len(deployed_code)
        try:
            gas.consume(deploy_gas)
        except OutOfGasError:
            return 0

        self.state.deploy_contract(new_addr, deployed_code)
        ctx.logs.extend(result.logs)

        return int.from_bytes(new_addr.ljust(32, b"\x00"), "big")

    # ===== Phase 17: Simulation =====

    def simulate(
        self,
        ctx: ExecutionContext,
    ) -> ExecutionResult:
        """
        Simulate execution without modifying persistent state.

        Takes a state snapshot before execution and always reverts after,
        regardless of success or failure. Used for ``eth_call`` and
        ``eth_estimateGas`` to provide accurate gas estimation and
        read-only contract queries.

        Args:
            ctx: The execution context (same as ``execute``).

        Returns:
            ExecutionResult capturing success/failure, return data, and
            gas used — but the state is always reverted.
        """
        snapshot = self.state.snapshot()
        try:
            result = self.execute(ctx)
            return result
        except Exception as e:
            logger.debug("vm_simulation_error: %s", e)
            return ExecutionResult(
                success=False,
                gas_used=ctx.msg.gas,
                error=f"Simulation error: {e}",
            )
        finally:
            self.state.revert(snapshot)

    # ===== Convenience execution helpers =====

    def execute_code(
        self,
        code: bytes,
        sender: bytes = b"\x00" * 20,
        contract_address: bytes = b"\x00" * 20,
        value: int = 0,
        data: bytes = b"",
        gas_limit: int = 1_000_000,
        block_height: int = 0,
        block_timestamp: int = 0,
        gas_price: int = 1,
    ) -> ExecutionResult:
        """
        Execute bytecode with minimal setup. Convenience method for testing.

        Args:
            code: Bytecode to execute.
            sender: Caller address.
            contract_address: Address of the contract.
            value: Value sent with the call.
            data: Calldata.
            gas_limit: Gas limit.
            block_height: Current block height.
            block_timestamp: Current block timestamp.
            gas_price: Gas price.

        Returns:
            ExecutionResult.
        """
        msg = MessageContext(
            sender=sender,
            value=value,
            data=data,
            gas=gas_limit,
        )
        block = BlockContext(
            height=block_height,
            timestamp=block_timestamp,
        )
        tx = TransactionContext(
            origin=sender,
            gas_price=gas_price,
        )
        ctx = ExecutionContext(
            msg=msg,
            block=block,
            tx=tx,
            contract_address=contract_address,
            code=code,
        )
        return self.execute(ctx)
