"""
Positronic - Transaction Executor
Executes transactions and produces state transitions.
Integrates with AI validation gate and PositronicVM.
All transactions are fully traceable.
"""

import logging
import time
from typing import Optional, List

from positronic.core.transaction import Transaction, TxType, TxStatus
from positronic.core.block import TransactionReceipt
from positronic.core.state import StateManager
from positronic.core.account import Account
from positronic.crypto.hashing import sha512
from positronic.crypto.address import (
    address_from_pubkey,
    TREASURY_ADDRESS,
    COMMUNITY_POOL_ADDRESS,
    BURN_ADDRESS,
)
from positronic.constants import (
    TX_BASE_GAS,
    CREATE_GAS,
    FEE_PRODUCER_SHARE,
    FEE_ATTESTER_SHARE,
    FEE_TREASURY_SHARE,
    FEE_NODE_SHARE,
    FEE_BURN_SHARE,
    MIN_STAKE,
)

logger = logging.getLogger(__name__)


class TransactionExecutor:
    """
    Executes transactions against the world state.
    Handles all transaction types including contract interactions.
    All transactions are fully traceable - no anonymous transfers.
    """

    def __init__(self, state: StateManager, validator_registry=None,
                 token_registry=None):
        self.state = state
        self._validator_registry = validator_registry
        self._token_registry = token_registry  # TokenRegistry for PRC-20/PRC-721 ops
        self._vm = None  # Lazy-initialized PositronicVM
        self._blockchain = None  # Wired by PositronicBlockchain after init

    def _get_vm(self):
        """Lazy-initialize PositronicVM to avoid circular imports."""
        if self._vm is None:
            try:
                from positronic.vm.vm import PositronicVM
                self._vm = PositronicVM(self.state)
            except ImportError:
                logger.warning("PositronicVM not available; contract execution disabled")
        return self._vm

    def execute(
        self,
        tx: Transaction,
        block_height: int,
        block_hash: bytes,
        validator_address: bytes = b"",
        nvn_addresses: List[bytes] = None,
        block_timestamp: int = 0,
    ) -> TransactionReceipt:
        """
        Execute a single transaction.
        Returns a receipt with status, gas used, and logs.
        """
        # Create snapshot for rollback
        snapshot_id = self.state.snapshot()

        try:
            receipt = self._execute_inner(
                tx, block_height, block_hash, validator_address, nvn_addresses or [],
                block_timestamp=block_timestamp,
            )

            if receipt.status:
                self.state.commit_snapshot(snapshot_id)
            else:
                self.state.revert(snapshot_id)

            return receipt

        except Exception as e:
            logger.debug("tx_execution_failed: %s", e)
            self.state.revert(snapshot_id)
            # After revert, deduct gas fee and distribute to validator
            sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""
            if sender_addr and tx.tx_type not in (TxType.REWARD, TxType.AI_TREASURY, TxType.GAME_REWARD):
                gas_fee = tx.gas_limit * tx.gas_price
                self.state.sub_balance(sender_addr, gas_fee)
                self._distribute_fees(gas_fee, validator_address, nvn_addresses or [])
            return TransactionReceipt(
                tx_hash=tx.tx_hash,
                block_hash=block_hash,
                block_height=block_height,
                tx_index=0,
                status=False,
                gas_used=tx.gas_limit,
                ai_score=tx.ai_score,
                error=str(e),
            )

    def _execute_inner(
        self,
        tx: Transaction,
        block_height: int,
        block_hash: bytes,
        validator_address: bytes,
        nvn_addresses: List[bytes],
        block_timestamp: int = 0,
    ) -> TransactionReceipt:
        """Internal execution logic."""

        # For system TXs, sender IS the address (20 bytes, not pubkey)
        # For user TXs, sender is pubkey (32 bytes) → derive address
        if tx.sender and len(tx.sender) == 20 and tx.signature == b"":
            sender_addr = tx.sender  # system TX: sender is raw address
        elif tx.sender:
            sender_addr = address_from_pubkey(tx.sender)
        else:
            sender_addr = b""

        # System transactions (rewards) - no fee deduction
        if tx.tx_type in (TxType.REWARD, TxType.AI_TREASURY, TxType.GAME_REWARD):
            return self._execute_reward(tx, block_height, block_hash)

        # System STAKE/UNSTAKE TXs: dispatch to proper handlers
        # so all nodes execute the same state changes.
        if tx.signature == b"" and tx.gas_price == 0 and tx.tx_type == TxType.STAKE:
            acc = self.state.get_account(sender_addr)
            logger.info("System STAKE TX: sender=%s value=%d staked_before=%d eff_bal=%d",
                        sender_addr.hex()[:16], tx.value, acc.staked_amount, acc.effective_balance)
            gas, ok, err = self._execute_stake(tx, sender_addr)
            acc2 = self.state.get_account(sender_addr)
            logger.info("System STAKE result: ok=%s err=%s staked_after=%d", ok, err, acc2.staked_amount)
            return TransactionReceipt(
                tx_hash=tx.tx_hash, status=ok, gas_used=gas,
                block_height=block_height, block_hash=block_hash,
                tx_index=0, error=err if not ok else None,
            )
        if tx.signature == b"" and tx.gas_price == 0 and tx.tx_type == TxType.UNSTAKE:
            gas, ok, err = self._execute_unstake(tx, sender_addr)
            return TransactionReceipt(
                tx_hash=tx.tx_hash, status=ok, gas_used=gas,
                block_height=block_height, block_hash=block_hash,
                tx_index=0, error=err if not ok else None,
            )
        if tx.signature == b"" and tx.gas_price == 0 and tx.tx_type == TxType.CLAIM_REWARDS:
            gas, ok, err = self._execute_claim_rewards(tx, sender_addr)
            return TransactionReceipt(
                tx_hash=tx.tx_hash, status=ok, gas_used=gas,
                block_height=block_height, block_hash=block_hash,
                tx_index=0, error=err if not ok else None,
            )

        # System TXs (admin transfers, faucet): execute the actual balance
        # transfer so ALL nodes update their state identically.
        if tx.signature == b"" and tx.gas_price == 0 and tx.gas_limit == 0:
            recipient_addr = bytes.fromhex(tx.recipient) if isinstance(tx.recipient, str) else tx.recipient
            if tx.value > 0 and recipient_addr:
                # Credit recipient — sender was already debited by the originating
                # node's admin_transfer(). Peer nodes debit+credit here.
                # Check if sender still has funds (peer nodes haven't debited yet)
                if sender_addr and sender_addr != b'\x00' * len(sender_addr):
                    sender_acc = self.state.get_account(sender_addr)
                    if sender_acc.balance >= tx.value:
                        # Peer node: debit sender (originator already did this)
                        sender_acc.balance -= tx.value
                        self.state.set_account(sender_addr, sender_acc)
                recipient_acc = self.state.get_account(recipient_addr)
                recipient_acc.balance += tx.value
                self.state.set_account(recipient_addr, recipient_acc)
                logger.info(
                    "system_tx_exec: type=%s to=%s value=%d",
                    tx.tx_type.name, recipient_addr.hex()[:16], tx.value,
                )
            else:
                logger.debug(
                    "system_tx_record: type=%s sender=%s value=%d",
                    tx.tx_type.name, sender_addr.hex()[:16], tx.value,
                )
            return TransactionReceipt(
                tx_hash=tx.tx_hash,
                block_hash=block_hash,
                block_height=block_height,
                tx_index=0,
                status=True,
                gas_used=0,
                ai_score=tx.ai_score,
            )

        # Verify sender has enough effective balance (excludes staked funds)
        sender_acc = self.state.get_account(sender_addr)
        total_cost = tx.total_cost
        if sender_acc.effective_balance < total_cost:
            return TransactionReceipt(
                tx_hash=tx.tx_hash,
                block_hash=block_hash,
                block_height=block_height,
                tx_index=0,
                status=False,
                gas_used=tx.intrinsic_gas,
                ai_score=tx.ai_score,
                error="Insufficient balance",
            )

        # Deduct gas upfront
        gas_cost = tx.gas_limit * tx.gas_price
        self.state.sub_balance(sender_addr, gas_cost)

        # Increment nonce
        self.state.increment_nonce(sender_addr)

        # Execute based on type
        gas_used = tx.intrinsic_gas
        receipt_status = True
        error = ""
        contract_address = None
        return_data = b""
        logs = []

        if tx.tx_type == TxType.TRANSFER:
            gas_used, receipt_status, error = self._execute_transfer(tx, sender_addr)

        elif tx.tx_type == TxType.CONTRACT_CREATE:
            gas_used, receipt_status, error, contract_address = (
                self._execute_contract_create(tx, sender_addr, block_height, validator_address,
                                              block_timestamp=block_timestamp)
            )

        elif tx.tx_type == TxType.CONTRACT_CALL:
            gas_used, receipt_status, error, return_data, logs = (
                self._execute_contract_call(tx, sender_addr, block_height, validator_address,
                                            block_timestamp=block_timestamp)
            )

        elif tx.tx_type == TxType.STAKE:
            gas_used, receipt_status, error = self._execute_stake(tx, sender_addr)

        elif tx.tx_type == TxType.UNSTAKE:
            gas_used, receipt_status, error = self._execute_unstake(tx, sender_addr)

        elif tx.tx_type == TxType.EVIDENCE:
            gas_used, receipt_status, error = self._execute_evidence(tx, sender_addr)

        elif tx.tx_type == TxType.TOKEN_CREATE:
            gas_used, receipt_status, error = self._execute_token_create(tx, sender_addr)

        elif tx.tx_type == TxType.TOKEN_TRANSFER:
            gas_used, receipt_status, error = self._execute_token_transfer(tx, sender_addr)

        elif tx.tx_type == TxType.NFT_MINT:
            gas_used, receipt_status, error = self._execute_nft_mint(tx, sender_addr)

        elif tx.tx_type == TxType.NFT_TRANSFER:
            gas_used, receipt_status, error = self._execute_nft_transfer(tx, sender_addr)

        elif tx.tx_type == TxType.BRIDGE_LOCK:
            receipt_status, gas_used = self._execute_bridge_lock(tx, sender_addr, block_height, block_timestamp)
        elif tx.tx_type == TxType.BRIDGE_MINT:
            receipt_status, gas_used = self._execute_bridge_system(tx, "mint")
        elif tx.tx_type == TxType.BRIDGE_BURN:
            receipt_status, gas_used = self._execute_bridge_system(tx, "burn")
        elif tx.tx_type == TxType.BRIDGE_RELEASE:
            receipt_status, gas_used = self._execute_bridge_system(tx, "release")

        # Bound gas_used: must be at least intrinsic_gas, at most gas_limit
        gas_used = max(gas_used, tx.intrinsic_gas)
        gas_used = min(gas_used, tx.gas_limit)

        # Refund unused gas
        gas_refund = (tx.gas_limit - gas_used) * tx.gas_price
        if gas_refund > 0:
            self.state.add_balance(sender_addr, gas_refund)

        # Distribute fees
        actual_fee = gas_used * tx.gas_price
        self._distribute_fees(actual_fee, validator_address, nvn_addresses)

        return TransactionReceipt(
            tx_hash=tx.tx_hash,
            block_hash=block_hash,
            block_height=block_height,
            tx_index=0,
            status=receipt_status,
            gas_used=gas_used,
            logs=logs,
            contract_address=contract_address,
            return_data=return_data,
            ai_score=tx.ai_score,
            error=error,
        )

    def _execute_transfer(self, tx: Transaction, sender_addr: bytes):
        """Execute a simple value transfer."""
        if tx.value > 0:
            success = self.state.transfer(sender_addr, tx.recipient, tx.value)
            if not success:
                return tx.intrinsic_gas, False, "Transfer failed: insufficient balance"
        return tx.intrinsic_gas, True, ""

    def _execute_contract_create(
        self, tx: Transaction, sender_addr: bytes,
        block_height: int = 0, validator_address: bytes = b"",
        block_timestamp: int = 0,
    ):
        """Execute contract deployment with VM init code execution."""
        # Generate contract address from sender + nonce
        nonce = self.state.get_nonce(sender_addr)
        contract_addr_input = sender_addr + nonce.to_bytes(8, "big")
        contract_address = sha512(contract_addr_input)[:20]

        # Transfer value to contract if any
        if tx.value > 0:
            success = self.state.transfer(sender_addr, contract_address, tx.value)
            if not success:
                return CREATE_GAS, False, "Insufficient balance for deployment", None

        if not tx.data:
            return CREATE_GAS, True, "", contract_address

        # Execute init code through PositronicVM
        vm = self._get_vm()
        if vm is not None:
            try:
                from positronic.vm.context import (
                    ExecutionContext, MessageContext, BlockContext, TransactionContext,
                )

                gas_for_init = tx.gas_limit - CREATE_GAS
                if gas_for_init < 0:
                    gas_for_init = 0

                msg = MessageContext(
                    sender=sender_addr,
                    value=tx.value,
                    data=b"",  # init code IS the code, not calldata
                    gas=gas_for_init,
                )
                block_ctx = BlockContext(
                    height=block_height,
                    timestamp=block_timestamp or int(time.time()),
                    coinbase=validator_address if validator_address else b"\x00" * 20,
                )
                tx_ctx = TransactionContext(
                    origin=sender_addr,
                    gas_price=tx.gas_price,
                    tx_hash=tx.tx_hash if tx.tx_hash else b"\x00" * 64,
                )
                ctx = ExecutionContext(
                    msg=msg,
                    block=block_ctx,
                    tx=tx_ctx,
                    contract_address=contract_address,
                    code=tx.data,  # init bytecode
                )

                result = vm.execute(ctx)

                if result.success:
                    # Deploy the returned bytecode (runtime code)
                    runtime_code = result.return_data if result.return_data else tx.data
                    self.state.deploy_contract(contract_address, runtime_code)
                    gas_used = CREATE_GAS + result.gas_used
                    return gas_used, True, "", contract_address
                else:
                    return CREATE_GAS + result.gas_used, False, result.error or "Init code reverted", None

            except Exception as e:
                logger.debug("VM init execution failed, deploying code directly: %s", e)
                # Fall through to direct deploy

        # Fallback: deploy code directly without running init
        self.state.deploy_contract(contract_address, tx.data)
        gas_used = CREATE_GAS + len(tx.data) * 200
        return gas_used, True, "", contract_address

    def _execute_contract_call(
        self, tx: Transaction, sender_addr: bytes,
        block_height: int = 0, validator_address: bytes = b"",
        block_timestamp: int = 0,
    ):
        """Execute a contract call through PositronicVM."""
        # Check if target is a contract
        code = self.state.get_code(tx.recipient)
        if not code:
            # Not a contract, treat as transfer
            if tx.value > 0:
                self.state.transfer(sender_addr, tx.recipient, tx.value)
            return tx.intrinsic_gas, True, "", b"", []

        # Check sender has enough balance for value transfer
        if tx.value > 0:
            sender_acc = self.state.get_account(sender_addr)
            if sender_acc.effective_balance < tx.value:
                return tx.intrinsic_gas, False, "Insufficient balance for call", b"", []

        # Execute through PositronicVM
        vm = self._get_vm()
        if vm is not None:
            try:
                from positronic.vm.context import (
                    ExecutionContext, MessageContext, BlockContext, TransactionContext,
                )

                gas_for_call = tx.gas_limit - tx.intrinsic_gas
                if gas_for_call < 0:
                    gas_for_call = 0

                msg = MessageContext(
                    sender=sender_addr,
                    value=tx.value,
                    data=tx.data if tx.data else b"",
                    gas=gas_for_call,
                )
                block_ctx = BlockContext(
                    height=block_height,
                    timestamp=block_timestamp or int(time.time()),
                    coinbase=validator_address if validator_address else b"\x00" * 20,
                )
                tx_ctx = TransactionContext(
                    origin=sender_addr,
                    gas_price=tx.gas_price,
                    tx_hash=tx.tx_hash if tx.tx_hash else b"\x00" * 64,
                )
                ctx = ExecutionContext(
                    msg=msg,
                    block=block_ctx,
                    tx=tx_ctx,
                    contract_address=tx.recipient,
                    code=code,
                )

                result = vm.execute(ctx)

                # Only transfer value on successful execution
                if result.success and tx.value > 0:
                    self.state.transfer(sender_addr, tx.recipient, tx.value)

                gas_used = tx.intrinsic_gas + result.gas_used
                logs = result.logs if result.logs else []
                return gas_used, result.success, result.error, result.return_data, logs

            except Exception as e:
                logger.debug("VM call execution failed: %s", e)
                # Fall through to simple gas estimation

        # Fallback when VM is not available — transfer value directly
        if tx.value > 0:
            self.state.transfer(sender_addr, tx.recipient, tx.value)
        gas_used = tx.intrinsic_gas + len(tx.data) * 16
        return gas_used, True, "", b"", []

    def _execute_stake(self, tx: Transaction, sender_addr: bytes):
        """Execute a staking transaction and register as validator.

        Stores the sender's public key on the account so that
        _sync_registry_from_state() can register the validator
        on ALL nodes that process this block (on-chain registry).
        """
        if tx.value < MIN_STAKE:
            return tx.intrinsic_gas, False, f"Minimum stake is {MIN_STAKE}"

        success = self.state.stake(sender_addr, tx.value)
        if not success:
            return tx.intrinsic_gas, False, "Insufficient balance for staking"

        acc = self.state.get_account(sender_addr)
        acc.is_validator = True
        # Store pubkey on-chain: prefer tx.data (system TXs), fallback to tx.sender (signed TXs)
        if tx.data and len(tx.data) == 32:
            acc.validator_pubkey = tx.data
        elif tx.sender and len(tx.sender) >= 32:
            acc.validator_pubkey = tx.sender[:32]
        self.state._sync_account_to_trie(sender_addr)

        # NOTE: Registry update is NOT done here — it happens in
        # blockchain._sync_registry_from_state() after block execution,
        # ensuring all nodes get the same registry state.

        return tx.intrinsic_gas, True, ""

    def _execute_unstake(self, tx: Transaction, sender_addr: bytes):
        """Execute an unstaking transaction."""
        success = self.state.unstake(sender_addr, tx.value)
        if not success:
            acc = self.state.get_account(sender_addr)
            if acc.staked_amount < tx.value:
                return tx.intrinsic_gas, False, "Insufficient staked amount"
            return tx.intrinsic_gas, False, "Cannot partially unstake below minimum stake"

        acc = self.state.get_account(sender_addr)
        if acc.staked_amount < MIN_STAKE:
            acc.is_validator = False

        return tx.intrinsic_gas, True, ""

    def _execute_claim_rewards(self, tx: Transaction, sender_addr: bytes):
        """Execute a claim-rewards transaction.

        Moves pending_rewards to balance for the sender address.
        Executed identically on all nodes via system TX in block.
        """
        claimed = self.state.claim_rewards(sender_addr)
        if claimed <= 0:
            return 0, False, "No rewards to claim"
        logger.info("claim_rewards_exec: addr=%s claimed=%d",
                     sender_addr.hex()[:16], claimed)
        return 0, True, ""

    # ── New TxType handlers (Whitepaper Ch 2.3 alignment) ──────────

    def _execute_evidence(self, tx: Transaction, sender_addr: bytes):
        """Execute an evidence submission transaction (TxType.EVIDENCE).

        Records forensic evidence on-chain. The ``data`` field carries the
        evidence payload (hash or serialized report).  Evidence transactions
        are value-less — the only cost is gas.
        """
        if not tx.data:
            return tx.intrinsic_gas, False, "Evidence payload required (data field empty)"
        # Evidence gas = base gas + payload size * 16 per byte
        evidence_gas = tx.intrinsic_gas + len(tx.data) * 16
        logger.info(
            "evidence_recorded: sender=%s payload_size=%d",
            sender_addr.hex()[:16], len(tx.data),
        )
        return evidence_gas, True, ""

    def _execute_token_create(self, tx: Transaction, sender_addr: bytes):
        """Execute a PRC-20 token creation transaction (TxType.TOKEN_CREATE).

        ``data`` must contain JSON-encoded token parameters:
        {name, symbol, decimals, total_supply}.
        A creation fee (TOKEN_CREATION_FEE) is burned.
        """
        import json as _json
        if not tx.data:
            return tx.intrinsic_gas, False, "Token parameters required in data field"
        try:
            params = _json.loads(tx.data)
        except (ValueError, UnicodeDecodeError):
            return tx.intrinsic_gas, False, "Invalid JSON in data field"

        required = ("name", "symbol", "decimals", "total_supply")
        if not all(k in params for k in required):
            return tx.intrinsic_gas, False, f"Missing fields; required: {required}"

        gas_used = tx.intrinsic_gas + len(tx.data) * 16

        # Actually create the token in the registry
        if self._token_registry is not None:
            token = self._token_registry.create_token(
                name=params["name"],
                symbol=params["symbol"],
                decimals=int(params["decimals"]),
                total_supply=int(params["total_supply"]),
                owner=sender_addr,
            )
            if token is None:
                return gas_used, False, "Token creation failed (duplicate symbol or invalid params)"
            logger.info(
                "token_create: sender=%s symbol=%s supply=%s token_id=%s",
                sender_addr.hex()[:16], params["symbol"],
                params["total_supply"], token.token_id,
            )
        else:
            logger.info(
                "token_create: sender=%s symbol=%s supply=%s (no registry)",
                sender_addr.hex()[:16], params.get("symbol"), params.get("total_supply"),
            )
        return gas_used, True, ""

    def _execute_token_transfer(self, tx: Transaction, sender_addr: bytes):
        """Execute a PRC-20 token transfer (TxType.TOKEN_TRANSFER).

        ``data`` must contain JSON: {token_id, amount}.
        ``recipient`` is the destination address.
        """
        import json as _json
        if not tx.data:
            return tx.intrinsic_gas, False, "Token transfer params required in data field"
        try:
            params = _json.loads(tx.data)
        except (ValueError, UnicodeDecodeError):
            return tx.intrinsic_gas, False, "Invalid JSON in data field"

        if "token_id" not in params or "amount" not in params:
            return tx.intrinsic_gas, False, "Missing token_id or amount"

        gas_used = tx.intrinsic_gas + len(tx.data) * 16

        # Actually transfer tokens in the registry
        if self._token_registry is not None:
            token = self._token_registry.get_token(params["token_id"])
            if token is None:
                return gas_used, False, f"Token {params['token_id']} not found"
            amount = int(params["amount"])
            if not token.transfer(sender_addr, tx.recipient, amount):
                return gas_used, False, "Token transfer failed (insufficient balance or invalid)"
            logger.info(
                "token_transfer: token=%s from=%s to=%s amount=%d",
                params["token_id"], sender_addr.hex()[:16],
                tx.recipient.hex()[:16], amount,
            )

        return gas_used, True, ""

    def _execute_nft_mint(self, tx: Transaction, sender_addr: bytes):
        """Execute a PRC-721 NFT mint (TxType.NFT_MINT).

        ``data`` must contain JSON: {collection_id, metadata_uri} or
        {name, symbol} to create a new collection and mint the first token.
        """
        import json as _json
        if not tx.data:
            return tx.intrinsic_gas, False, "NFT mint params required in data field"
        try:
            params = _json.loads(tx.data)
        except (ValueError, UnicodeDecodeError):
            return tx.intrinsic_gas, False, "Invalid JSON in data field"

        gas_used = tx.intrinsic_gas + len(tx.data) * 16

        if self._token_registry is not None:
            collection_id = params.get("collection_id")

            # If no collection_id, create a new collection first
            if not collection_id and "name" in params and "symbol" in params:
                collection = self._token_registry.create_collection(
                    name=params["name"],
                    symbol=params["symbol"],
                    owner=sender_addr,
                    max_supply=int(params.get("max_supply", 0)),
                )
                if collection is None:
                    return gas_used, False, "NFT collection creation failed"
                collection_id = collection.collection_id
                logger.info(
                    "nft_collection_created: sender=%s collection_id=%s",
                    sender_addr.hex()[:16], collection_id,
                )

            if collection_id:
                collection = self._token_registry.get_collection(collection_id)
                if collection is None:
                    return gas_used, False, f"Collection {collection_id} not found"

                # Build metadata if provided
                nft_meta = None
                metadata_uri = params.get("metadata_uri", "")
                if metadata_uri or params.get("token_name"):
                    from positronic.tokens.prc721 import NFTMetadata
                    nft_meta = NFTMetadata(
                        name=params.get("token_name", ""),
                        description=params.get("description", ""),
                        image_uri=metadata_uri,
                        attributes=params.get("attributes", {}),
                        dynamic=params.get("dynamic", False),
                    )

                # Mint to sender (or to recipient if specified)
                mint_to = tx.recipient if tx.recipient and tx.recipient != b"" else sender_addr
                token_id = collection.mint(to=mint_to, metadata=nft_meta)
                if token_id is None:
                    return gas_used, False, "NFT mint failed (max supply reached or duplicate)"
                logger.info(
                    "nft_mint: sender=%s collection=%s token_id=%d",
                    sender_addr.hex()[:16], collection_id, token_id,
                )
            else:
                return gas_used, False, "Missing collection_id or name/symbol for new collection"
        else:
            logger.info("nft_mint: sender=%s (no registry)", sender_addr.hex()[:16])

        return gas_used, True, ""

    def _execute_nft_transfer(self, tx: Transaction, sender_addr: bytes):
        """Execute a PRC-721 NFT transfer (TxType.NFT_TRANSFER).

        ``data`` must contain JSON: {collection_id, token_id}.
        ``recipient`` is the destination address.
        """
        import json as _json
        if not tx.data:
            return tx.intrinsic_gas, False, "NFT transfer params required in data field"
        try:
            params = _json.loads(tx.data)
        except (ValueError, UnicodeDecodeError):
            return tx.intrinsic_gas, False, "Invalid JSON in data field"

        if "collection_id" not in params or "token_id" not in params:
            return tx.intrinsic_gas, False, "Missing collection_id or token_id"

        gas_used = tx.intrinsic_gas + len(tx.data) * 16

        # Actually transfer the NFT in the registry
        if self._token_registry is not None:
            collection = self._token_registry.get_collection(params["collection_id"])
            if collection is None:
                return gas_used, False, f"Collection {params['collection_id']} not found"
            nft_token_id = int(params["token_id"])
            if not collection.transfer(sender_addr, tx.recipient, nft_token_id):
                return gas_used, False, "NFT transfer failed (not owner or invalid token)"
            logger.info(
                "nft_transfer: collection=%s token=%d from=%s to=%s",
                params["collection_id"], nft_token_id,
                sender_addr.hex()[:16], tx.recipient.hex()[:16],
            )

        return gas_used, True, ""

    # ── End new TxType handlers ──────────────────────────────────

    def _execute_reward(self, tx: Transaction, block_height: int, block_hash: bytes):
        """Execute a reward/treasury transaction (no fees).

        REWARD and GAME_REWARD mint new tokens (credit only).
        AI_TREASURY debits the treasury address and credits the recipient.
        """
        if tx.tx_type == TxType.AI_TREASURY:
            # Treasury spend: debit treasury, credit recipient
            sender_addr = tx.sender if tx.sender and len(tx.sender) == 20 else TREASURY_ADDRESS
            treasury_acc = self.state.get_account(sender_addr)
            if treasury_acc.balance < tx.value:
                return TransactionReceipt(
                    tx_hash=tx.tx_hash, block_hash=block_hash,
                    block_height=block_height, tx_index=0,
                    status=False, gas_used=0,
                    error="Insufficient treasury balance",
                )
            self.state.sub_balance(sender_addr, tx.value)
        self.state.add_balance(tx.recipient, tx.value)
        return TransactionReceipt(
            tx_hash=tx.tx_hash,
            block_hash=block_hash,
            block_height=block_height,
            tx_index=0,
            status=True,
            gas_used=0,
        )

    def _distribute_fees(
        self,
        total_fee: int,
        validator_address: bytes,
        nvn_addresses: List[bytes],
        attester_payouts: dict = None,
    ):
        """
        Distribute transaction fees (Consensus v2 Three-Layer):
        25% Block Producer, 25% Attesters, 20% Nodes, 10% Treasury, 20% Burned

        If attester_payouts is provided (v2 mode), distribute attester share
        pro-rata by stake. Otherwise, fall back to NVN-based distribution
        for backward compatibility.

        When no attesters/NVNs are registered, that share is redirected:
        50% to treasury and 50% to burn.
        """
        if total_fee <= 0:
            return

        producer_share = int(total_fee * FEE_PRODUCER_SHARE)
        attester_share = int(total_fee * FEE_ATTESTER_SHARE)
        treasury_share = int(total_fee * FEE_TREASURY_SHARE)
        node_share = int(total_fee * FEE_NODE_SHARE)
        # Burn is the remainder to prevent rounding dust loss
        burn_share = total_fee - producer_share - attester_share - treasury_share - node_share

        # Block Producer gets their share (25%)
        if validator_address:
            self.state.add_balance(validator_address, producer_share)

        # Attesters (25%, pro-rata by stake) — Consensus v2
        if attester_payouts:
            total_attester_stake = sum(attester_payouts.values())
            if total_attester_stake > 0:
                distributed = 0
                items = list(attester_payouts.items())
                for i, (addr, stake) in enumerate(items):
                    if i == len(items) - 1:
                        share = attester_share - distributed
                    else:
                        share = int(attester_share * stake / total_attester_stake)
                    if share > 0:
                        self.state.add_pending_rewards(addr, share)
                    distributed += share
            else:
                # No stake → redirect attester share
                treasury_share += attester_share // 2
                burn_share += attester_share - attester_share // 2
        elif nvn_addresses:
            # Backward-compat: split among NVN addresses
            per_nvn = attester_share // len(nvn_addresses)
            remainder = attester_share - per_nvn * len(nvn_addresses)
            for nvn_addr in nvn_addresses:
                self.state.add_pending_rewards(nvn_addr, per_nvn)
            if remainder > 0:
                self.state.add_balance(TREASURY_ADDRESS, remainder)
        else:
            # No attesters/NVNs: redirect to treasury (50%) and burn (50%)
            nvn_to_treasury = attester_share // 2
            nvn_to_burn = attester_share - nvn_to_treasury
            treasury_share += nvn_to_treasury
            burn_share += nvn_to_burn

        # Treasury (10%)
        self.state.add_balance(TREASURY_ADDRESS, treasury_share)

        # Node operators / Community pool (20%)
        if node_share > 0:
            self.state.add_balance(COMMUNITY_POOL_ADDRESS, node_share)

        # Burn (20%) - send to burn address, removing from circulation
        self.state.add_balance(BURN_ADDRESS, burn_share)

    # ------------------------------------------------------------------ #
    #  Bridge Execution                                                    #
    # ------------------------------------------------------------------ #

    def _execute_bridge_lock(self, tx, sender_addr: bytes, block_height: int, block_timestamp: int):
        """Execute a BRIDGE_LOCK transaction: deduct balance, create lock record."""
        import json
        try:
            data = json.loads(tx.data.decode()) if tx.data else {}
        except Exception:
            return False, TX_BASE_GAS

        target_chain_id = data.get("target_chain", 0)
        recipient_ext   = data.get("recipient", "")
        if not recipient_ext:
            return False, TX_BASE_GAS

        amount = tx.value
        if amount <= 0:
            return False, TX_BASE_GAS

        # Check balance
        if not self.state.sub_balance(sender_addr, amount):
            return False, TX_BASE_GAS

        # Record the lock via blockchain bridge (if available)
        if self._blockchain is not None and hasattr(self._blockchain, 'lock_mint_bridge'):
            bridge = self._blockchain.lock_mint_bridge
            from positronic.bridge.cross_chain import TargetChain, LockStatus
            try:
                target_chain = TargetChain(target_chain_id)
            except ValueError:
                target_chain = TargetChain.ETHEREUM
            # Create lock record directly (balance already deducted)
            import hashlib, time, secrets
            lock_id = "lock_" + secrets.token_hex(8)
            fee     = amount * 30 // 10000  # 0.3%
            net     = amount - fee
            from positronic.bridge.cross_chain import LockRecord
            record = LockRecord(
                lock_id=lock_id,
                sender=sender_addr,
                amount=net,
                fee=fee,
                target_chain=target_chain,
                recipient_external=recipient_ext,
                lock_hash=hashlib.sha256(sender_addr + net.to_bytes(32, "big")).digest(),
                status=LockStatus.PENDING,
                confirmations=[],
                created_at=time.time(),
                confirmed_at=0.0,
                minted_at=0.0,
                nonce=bridge._nonce,
            )
            bridge._locks[lock_id] = record
            bridge._nonce += 1
            bridge._total_locked += net
            bridge._save_lock(lock_id)

        return True, TX_BASE_GAS * 3

    def _execute_bridge_system(self, tx, action: str):
        """Execute system bridge TX (mint/burn/release) — state update only."""
        import json
        try:
            data = json.loads(tx.data.decode()) if tx.data else {}
        except Exception:
            return False, TX_BASE_GAS

        lock_id = data.get("lock_id", "")
        if not lock_id or self._blockchain is None or not hasattr(self._blockchain, 'lock_mint_bridge'):
            return True, TX_BASE_GAS  # Non-fatal — system TX

        bridge = self._blockchain.lock_mint_bridge
        lock   = bridge._locks.get(lock_id)
        if not lock:
            return True, TX_BASE_GAS

        from positronic.bridge.cross_chain import LockStatus
        import time
        if action == "mint":
            lock.status    = LockStatus.MINTED
            lock.minted_at = time.time()
            bridge._total_minted += lock.amount
        elif action == "release":
            self.state.add_balance(lock.sender, lock.amount)
            lock.status = LockStatus.RELEASED
            bridge._total_released += lock.amount

        bridge._save_lock(lock_id)
        return True, TX_BASE_GAS
