"""
Positronic - World State Manager
Manages all account states, contract storage, and state transitions.
Supports snapshots and rollback for transaction execution.
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import copy

logger = logging.getLogger(__name__)

from positronic.core.account import Account
from positronic.core.mpt import MerklePatriciaTrie, StateProof
from positronic.crypto.hashing import sha512
from positronic.crypto.address import (
    ZERO_ADDRESS,
    TREASURY_ADDRESS,
    BURN_ADDRESS,
    address_from_pubkey,
)
from positronic.utils.serialization import to_json_bytes


@dataclass
class StateSnapshot:
    """Snapshot of state for rollback support."""
    snapshot_id: int
    accounts: Dict[bytes, Account]
    contract_storage: Dict[bytes, Dict[bytes, bytes]]
    contract_code: Dict[bytes, bytes] = None


class StateManager:
    """
    Manages the global world state of Positronic.
    In-memory state with periodic persistence to database.
    Supports snapshot/rollback for atomic transaction execution.
    """

    def __init__(self):
        self.accounts: Dict[bytes, Account] = {}
        self.contract_code: Dict[bytes, bytes] = {}        # code_hash -> bytecode
        self.contract_storage: Dict[bytes, Dict[bytes, bytes]] = {}  # addr -> {key: val}
        self._snapshot_counter: int = 0
        self._snapshots: Dict[int, StateSnapshot] = {}
        self._trie = MerklePatriciaTrie()

    def get_account(self, address: bytes) -> Account:
        """Get account by address. Creates empty account if not exists."""
        if address not in self.accounts:
            self.accounts[address] = Account(address=address)
        return self.accounts[address]

    def _sync_account_to_trie(self, address: bytes):
        """Sync an account's current state into the MPT."""
        acc = self.accounts.get(address)
        if acc is not None:
            self._trie.put(address, to_json_bytes(acc.to_dict()))

    def set_account(self, address: bytes, account: Account):
        """Set account state."""
        self.accounts[address] = account
        self._sync_account_to_trie(address)

    def account_exists(self, address: bytes) -> bool:
        """Check if an account exists and is non-empty."""
        acc = self.accounts.get(address)
        return acc is not None and not acc.is_empty

    def get_balance(self, address: bytes) -> int:
        """Get account balance."""
        return self.get_account(address).balance

    def get_nonce(self, address: bytes) -> int:
        """Get account nonce."""
        return self.get_account(address).nonce

    def increment_nonce(self, address: bytes):
        """Increment account nonce."""
        self.get_account(address).nonce += 1
        self._sync_account_to_trie(address)

    def add_balance(self, address: bytes, amount: int):
        """Add to account balance. Amount must be positive."""
        if amount < 0:
            raise ValueError("add_balance: amount must be non-negative")
        acc = self.get_account(address)
        acc.balance += amount
        self._sync_account_to_trie(address)

    def sub_balance(self, address: bytes, amount: int) -> bool:
        """Subtract from account balance. Returns False if insufficient.
        Uses effective_balance to respect staked (locked) funds."""
        acc = self.get_account(address)
        if acc.effective_balance < amount:
            return False
        acc.balance -= amount
        self._sync_account_to_trie(address)
        return True

    def transfer(self, sender: bytes, recipient: bytes, amount: int) -> bool:
        """
        Transfer value between accounts.
        Returns True on success, False if insufficient effective balance.
        Staked funds are locked and cannot be transferred.
        """
        if amount < 0:
            return False
        if amount == 0:
            return True

        sender_acc = self.get_account(sender)
        if sender_acc.effective_balance < amount:
            return False

        sender_acc.balance -= amount
        self.get_account(recipient).balance += amount
        self._sync_account_to_trie(sender)
        self._sync_account_to_trie(recipient)
        return True

    # === Contract Operations ===

    def deploy_contract(self, address: bytes, code: bytes) -> bytes:
        """Deploy contract code and return the code hash."""
        code_hash = sha512(code)
        self.contract_code[code_hash] = code
        acc = self.get_account(address)
        acc.code_hash = code_hash
        self._sync_account_to_trie(address)
        return code_hash

    def get_code(self, address: bytes) -> bytes:
        """Get contract bytecode by address."""
        acc = self.get_account(address)
        if not acc.code_hash:
            return b""
        return self.contract_code.get(acc.code_hash, b"")

    def get_code_by_hash(self, code_hash: bytes) -> bytes:
        """Get contract bytecode by hash."""
        return self.contract_code.get(code_hash, b"")

    def get_storage(self, contract_addr: bytes, key: bytes) -> bytes:
        """Get a contract storage value."""
        storage = self.contract_storage.get(contract_addr, {})
        return storage.get(key, b"\x00" * 32)

    def set_storage(self, contract_addr: bytes, key: bytes, value: bytes):
        """Set a contract storage value."""
        if contract_addr not in self.contract_storage:
            self.contract_storage[contract_addr] = {}
        self.contract_storage[contract_addr][key] = value

    # === Staking ===

    def stake(self, address: bytes, amount: int) -> bool:
        """Stake tokens for DPoS validation. Total stake must meet MIN_STAKE."""
        from positronic.constants import MIN_STAKE
        acc = self.get_account(address)
        if acc.effective_balance < amount:
            return False
        if acc.staked_amount + amount < MIN_STAKE:
            return False  # total stake must meet minimum
        acc.staked_amount += amount
        self._sync_account_to_trie(address)
        return True

    def unstake(self, address: bytes, amount: int) -> bool:
        """Unstake tokens with 7-epoch unbonding period."""
        import time as _time
        from positronic.constants import MIN_STAKE
        UNBONDING_EPOCHS = 7
        EPOCH_DURATION = 384  # 12s * 32 slots
        acc = self.get_account(address)
        if acc.staked_amount < amount:
            return False
        remaining = acc.staked_amount - amount
        if remaining > 0 and remaining < MIN_STAKE:
            return False  # can't unstake to below MIN_STAKE unless fully unstaking
        acc.staked_amount -= amount
        acc.unstaking_amount += amount
        acc.unstake_available_at = _time.time() + (UNBONDING_EPOCHS * EPOCH_DURATION)
        self._sync_account_to_trie(address)
        return True

    def complete_unstaking(self, address: bytes) -> int:
        """Release matured unstaking funds. Returns amount released."""
        import time as _time
        acc = self.get_account(address)
        if acc.unstaking_amount > 0 and _time.time() >= acc.unstake_available_at:
            released = acc.unstaking_amount
            acc.unstaking_amount = 0
            acc.unstake_available_at = 0.0
            self._sync_account_to_trie(address)
            return released
        return 0

    def add_pending_rewards(self, address: bytes, amount: int):
        """Add to pending staking/attestation rewards."""
        if amount <= 0:
            return
        acc = self.get_account(address)
        acc.pending_rewards += amount
        self._sync_account_to_trie(address)

    def get_staking_rewards(self, address: bytes) -> int:
        """Get pending rewards for an address."""
        return self.get_account(address).pending_rewards

    def claim_rewards(self, address: bytes) -> int:
        """Move pending_rewards to balance. Returns amount claimed."""
        acc = self.get_account(address)
        if acc.pending_rewards <= 0:
            return 0
        claimed = acc.pending_rewards
        acc.balance += claimed
        acc.pending_rewards = 0
        self._sync_account_to_trie(address)
        return claimed

    # === Snapshot / Rollback ===

    def snapshot(self) -> int:
        """
        Create a state snapshot for potential rollback.
        Returns snapshot ID.
        """
        self._snapshot_counter += 1
        sid = self._snapshot_counter

        # Deep copy accounts and storage
        snap_accounts = {}
        for addr, acc in self.accounts.items():
            snap_accounts[addr] = Account(
                address=acc.address,
                nonce=acc.nonce,
                balance=acc.balance,
                code_hash=acc.code_hash,
                storage_root=acc.storage_root,
                staked_amount=acc.staked_amount,
                delegated_to=acc.delegated_to,
                is_validator=acc.is_validator,
                is_nvn=acc.is_nvn,
                validator_pubkey=acc.validator_pubkey,
                ai_reputation=acc.ai_reputation,
                quarantine_count=acc.quarantine_count,
                trust_score=acc.trust_score,
                trust_level=acc.trust_level,
                pending_rewards=acc.pending_rewards,
                unstaking_amount=acc.unstaking_amount,
                unstake_available_at=acc.unstake_available_at,
            )

        snap_storage = {}
        for addr, storage in self.contract_storage.items():
            snap_storage[addr] = dict(storage)

        # Copy contract code to prevent leaking deployed code on revert
        snap_code = dict(self.contract_code)

        self._snapshots[sid] = StateSnapshot(
            snapshot_id=sid,
            accounts=snap_accounts,
            contract_storage=snap_storage,
            contract_code=snap_code,
        )

        return sid

    def revert(self, snapshot_id: int):
        """Revert state to a snapshot.

        If the snapshot has already been consumed (committed or reverted
        by an earlier call), this is a no-op with a warning log rather
        than a hard crash.  This prevents cascading failures when nested
        snapshot/revert pairs (e.g. executor inside create_block) interact
        with exception handling.
        """
        snap = self._snapshots.get(snapshot_id)
        if not snap:
            import logging
            logging.getLogger(__name__).warning(
                "Snapshot %d already consumed (active: %s), skipping revert",
                snapshot_id,
                sorted(self._snapshots.keys())[-5:] if self._snapshots else "none",
            )
            return

        self.accounts = snap.accounts
        self.contract_storage = snap.contract_storage
        if snap.contract_code is not None:
            self.contract_code = snap.contract_code

        # Rebuild trie from restored accounts
        self._rebuild_trie()

        # Clean up this snapshot and any newer ones
        to_remove = [sid for sid in self._snapshots if sid >= snapshot_id]
        for sid in to_remove:
            del self._snapshots[sid]

    def commit_snapshot(self, snapshot_id: int):
        """Commit a snapshot (discard it, keeping current state)."""
        self._snapshots.pop(snapshot_id, None)

    def _rebuild_trie(self):
        """Rebuild the MPT from current accounts dict.

        Keys are sorted lexicographically to guarantee deterministic
        state roots regardless of account insertion / loading order.
        """
        self._trie = MerklePatriciaTrie()
        for address in sorted(self.accounts.keys()):
            self._sync_account_to_trie(address)

    # === State Root ===

    def compute_state_root(self) -> bytes:
        """
        Compute the state root hash via the Merkle Patricia Trie.
        O(1) when trie is up-to-date (cached root hash).
        Returns 64-byte SHA-512 root.
        """
        if not self.accounts:
            return b"\x00" * 64
        return self._trie.root_hash

    # === State Proofs ===

    def get_state_proof(self, address: bytes) -> Optional[StateProof]:
        """Generate a Merkle proof for an account address."""
        return self._trie.get_proof(address)

    def verify_state_proof(
        self, root_hash: bytes, address: bytes, proof: StateProof
    ) -> bool:
        """Verify a state proof against a given root hash."""
        return MerklePatriciaTrie.verify_proof(root_hash, address, proof) is not None

    # === Persistence ===

    def load_from_db(self, state_db):
        """Load full state from database: accounts, contract code, and contract storage."""
        import json

        # Load all accounts (skip corrupted entries gracefully)
        rows = state_db.db.execute("SELECT account_json FROM accounts").fetchall()
        for row in rows:
            try:
                acc = Account.from_dict(json.loads(row["account_json"]))
                if acc.address:
                    self.accounts[acc.address] = acc
            except Exception as e:
                logger.debug("Skipping malformed account row during load: %s", e)

        # Load contract code
        code_rows = state_db.db.execute(
            "SELECT code_hash, bytecode FROM contract_code"
        ).fetchall()
        for row in code_rows:
            try:
                code_hash = bytes.fromhex(row["code_hash"])
                self.contract_code[code_hash] = bytes(row["bytecode"])
            except Exception as e:
                logger.warning("Skipping corrupted contract code: %s", e)

        # Load contract storage
        storage_rows = state_db.db.execute(
            "SELECT contract_address, storage_key, storage_value FROM contract_storage"
        ).fetchall()
        for row in storage_rows:
            try:
                addr = bytes.fromhex(row["contract_address"])
                key = bytes.fromhex(row["storage_key"])
                value = bytes.fromhex(row["storage_value"])
                if addr not in self.contract_storage:
                    self.contract_storage[addr] = {}
                self.contract_storage[addr][key] = value
            except Exception as e:
                logger.warning("Skipping corrupted storage row: %s", e)

        # Rebuild MPT from loaded accounts so state root is correct
        self._rebuild_trie()

    def save_to_db(self, state_db, commit: bool = True):
        """Persist full state to database: accounts, contract code, and contract storage.

        Args:
            state_db: The state database to persist to.
            commit: If True (default), commit after writing. Set to False for
                    atomic multi-component commits (FIX 3/4: block+state atomicity).
        """
        # Save all accounts
        for addr, acc in self.accounts.items():
            state_db.put_account(acc)

        # Save contract code (FIX 4: ensures contract code is included in atomic commit)
        for code_hash, bytecode in self.contract_code.items():
            state_db.put_contract_code(code_hash, bytecode)

        # Save contract storage
        for contract_addr, storage in self.contract_storage.items():
            for key, value in storage.items():
                state_db.put_storage(contract_addr, key, value)

        if commit:
            state_db.commit()

    # === Stats ===

    def get_total_supply(self) -> int:
        """Calculate total supply (sum of all balances)."""
        return sum(acc.balance for acc in self.accounts.values())

    def get_stats(self) -> dict:
        return {
            "account_count": len(self.accounts),
            "contract_count": len(self.contract_code),
            "total_supply": self.get_total_supply(),
            "tx_count": sum(acc.nonce for acc in self.accounts.values()),
            "snapshots": len(self._snapshots),
        }
