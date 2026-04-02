"""
Positronic - State Storage
Account state, contract code, and contract storage persistence.
"""

from typing import Optional, List, Dict
import json

from positronic.storage.database import Database
from positronic.core.account import Account


class StateDB:
    """Account and contract state storage operations."""

    def __init__(self, db: Database):
        self.db = db

    # === Account Operations ===

    def put_account(self, account: Account):
        """Store or update an account."""
        self.db.execute(
            """INSERT OR REPLACE INTO accounts
               (address, nonce, balance, code_hash, storage_root,
                staked_amount, delegated_to, is_validator, is_nvn,
                ai_reputation, quarantine_count, account_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                account.address.hex(),
                account.nonce,
                str(account.balance),
                account.code_hash.hex() if account.code_hash else "",
                account.storage_root.hex() if account.storage_root else "",
                str(account.staked_amount),
                account.delegated_to.hex() if account.delegated_to else "",
                int(account.is_validator),
                int(account.is_nvn),
                account.ai_reputation,
                account.quarantine_count,
                json.dumps(account.to_dict()),
            ),
        )

    def get_account(self, address: bytes) -> Optional[Account]:
        """Retrieve an account by address."""
        row = self.db.execute(
            "SELECT account_json FROM accounts WHERE address = ?",
            (address.hex(),),
        ).fetchone()
        if not row:
            return None
        return Account.from_dict(json.loads(row["account_json"]))

    def get_balance(self, address: bytes) -> int:
        """Get account balance."""
        row = self.db.execute(
            "SELECT balance FROM accounts WHERE address = ?",
            (address.hex(),),
        ).fetchone()
        return row["balance"] if row else 0

    def get_nonce(self, address: bytes) -> int:
        """Get account nonce."""
        row = self.db.execute(
            "SELECT nonce FROM accounts WHERE address = ?",
            (address.hex(),),
        ).fetchone()
        return row["nonce"] if row else 0

    def get_all_validators(self) -> List[Account]:
        """Get all validator accounts."""
        rows = self.db.execute(
            "SELECT account_json FROM accounts WHERE is_validator = 1"
        ).fetchall()
        return [Account.from_dict(json.loads(row["account_json"])) for row in rows]

    def get_all_nvns(self) -> List[Account]:
        """Get all Neural Validator Node accounts."""
        rows = self.db.execute(
            "SELECT account_json FROM accounts WHERE is_nvn = 1"
        ).fetchall()
        return [Account.from_dict(json.loads(row["account_json"])) for row in rows]

    def get_richest_accounts(self, limit: int = 100) -> List[Account]:
        """Get accounts with highest balances."""
        rows = self.db.execute(
            """SELECT account_json FROM accounts
               ORDER BY balance DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [Account.from_dict(json.loads(row["account_json"])) for row in rows]

    # === Contract Code ===

    def put_contract_code(self, code_hash: bytes, bytecode: bytes):
        """Store contract bytecode."""
        self.db.execute(
            "INSERT OR REPLACE INTO contract_code (code_hash, bytecode) VALUES (?, ?)",
            (code_hash.hex(), bytecode),
        )

    def get_contract_code(self, code_hash: bytes) -> Optional[bytes]:
        """Retrieve contract bytecode."""
        row = self.db.execute(
            "SELECT bytecode FROM contract_code WHERE code_hash = ?",
            (code_hash.hex(),),
        ).fetchone()
        return bytes(row["bytecode"]) if row else None

    # === Contract Storage ===

    def put_storage(self, contract_address: bytes, key: bytes, value: bytes):
        """Store a contract storage slot."""
        self.db.execute(
            """INSERT OR REPLACE INTO contract_storage
               (contract_address, storage_key, storage_value)
               VALUES (?, ?, ?)""",
            (contract_address.hex(), key.hex(), value.hex()),
        )

    def get_storage(self, contract_address: bytes, key: bytes) -> Optional[bytes]:
        """Retrieve a contract storage slot."""
        row = self.db.execute(
            """SELECT storage_value FROM contract_storage
               WHERE contract_address = ? AND storage_key = ?""",
            (contract_address.hex(), key.hex()),
        ).fetchone()
        return bytes.fromhex(row["storage_value"]) if row else None

    def get_all_storage(self, contract_address: bytes) -> Dict[bytes, bytes]:
        """Get all storage for a contract."""
        rows = self.db.execute(
            """SELECT storage_key, storage_value FROM contract_storage
               WHERE contract_address = ?""",
            (contract_address.hex(),),
        ).fetchall()
        return {
            bytes.fromhex(row["storage_key"]): bytes.fromhex(row["storage_value"])
            for row in rows
        }

    # === Bulk Operations ===

    def commit(self):
        """Commit all pending changes."""
        self.db.safe_commit()

    def put_accounts_batch(self, accounts: List[Account]):
        """Store multiple accounts in a single transaction."""
        for account in accounts:
            self.put_account(account)
        self.db.safe_commit()
