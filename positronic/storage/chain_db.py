"""
Positronic - Blockchain Storage Queries
Block and transaction persistence.
"""

from typing import Optional, List
import json

from positronic.storage.database import Database
from positronic.core.block import Block, BlockHeader
from positronic.core.transaction import Transaction


class ChainDB:
    """Block and transaction storage operations."""

    def __init__(self, db: Database):
        self.db = db

    def put_block(self, block: Block, commit: bool = True):
        """Store a block and its transactions.

        Args:
            block: The block to persist.
            commit: If True (default), commit the transaction immediately.
                    Set to False when batching with other writes (e.g. state)
                    to achieve atomic block+state commits.
        """
        header_json = json.dumps(block.header.to_dict())
        self.db.execute(
            """INSERT OR REPLACE INTO blocks
               (height, hash, header_json, timestamp, tx_count, gas_used)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                block.height,
                block.hash.hex(),
                header_json,
                block.header.timestamp,
                block.tx_count,
                block.header.gas_used,
            ),
        )

        for idx, tx in enumerate(block.transactions):
            self.put_transaction(tx, block.hash, block.height, idx)

        if commit:
            self.db.safe_commit()

    def put_transaction(
        self,
        tx: Transaction,
        block_hash: bytes,
        block_height: int,
        tx_index: int,
    ):
        """Store a transaction."""
        tx_json = json.dumps(tx.to_dict())
        self.db.execute(
            """INSERT OR REPLACE INTO transactions
               (tx_hash, block_hash, block_height, tx_index, tx_json,
                sender, recipient, value, tx_type, ai_score, status, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tx.tx_hash.hex(),
                block_hash.hex(),
                block_height,
                tx_index,
                tx_json,
                tx.sender.hex(),
                tx.recipient.hex(),
                str(tx.value),
                int(tx.tx_type),
                tx.ai_score,
                int(tx.status),
                tx.timestamp,
            ),
        )

    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Retrieve a block by height."""
        row = self.db.execute(
            "SELECT hash, header_json FROM blocks WHERE height = ?", (height,)
        ).fetchone()
        if not row:
            return None
        header = BlockHeader.from_dict(json.loads(row["header_json"]))
        txs = self.get_transactions_by_block_height(height)
        # Use the stored hash (from when block was originally produced)
        # to ensure backward compatibility when header format evolves.
        stored_hash = bytes.fromhex(row["hash"]) if row["hash"] else None
        return Block(header=header, transactions=txs, _cached_hash=stored_hash)

    def get_block_by_hash(self, block_hash: bytes) -> Optional[Block]:
        """Retrieve a block by hash."""
        row = self.db.execute(
            "SELECT hash, height, header_json FROM blocks WHERE hash = ?",
            (block_hash.hex(),),
        ).fetchone()
        if not row:
            return None
        header = BlockHeader.from_dict(json.loads(row["header_json"]))
        txs = self.get_transactions_by_block_height(row["height"])
        stored_hash = bytes.fromhex(row["hash"]) if row["hash"] else None
        return Block(header=header, transactions=txs, _cached_hash=stored_hash)

    def get_transaction(self, tx_hash: bytes) -> Optional[Transaction]:
        """Retrieve a transaction by hash."""
        row = self.db.execute(
            "SELECT tx_json FROM transactions WHERE tx_hash = ?",
            (tx_hash.hex(),),
        ).fetchone()
        if not row:
            return None
        return Transaction.from_dict(json.loads(row["tx_json"]))

    def get_transactions_by_block_height(self, height: int) -> List[Transaction]:
        """Get all transactions in a block."""
        rows = self.db.execute(
            """SELECT tx_json FROM transactions
               WHERE block_height = ?
               ORDER BY tx_index""",
            (height,),
        ).fetchall()
        return [Transaction.from_dict(json.loads(row["tx_json"])) for row in rows]

    def get_transactions_by_sender(
        self, sender: bytes, limit: int = 50
    ) -> List[Transaction]:
        """Get recent transactions by sender."""
        rows = self.db.execute(
            """SELECT tx_json FROM transactions
               WHERE sender = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (sender.hex(), limit),
        ).fetchall()
        return [Transaction.from_dict(json.loads(row["tx_json"])) for row in rows]

    def get_transactions_by_address(
        self, address: str, limit: int = 50
    ) -> list:
        """Get recent transactions where address is sender OR recipient.
        Returns list of dicts (not Transaction objects) for RPC serialization."""
        addr_hex = address.lower().replace("0x", "")
        rows = self.db.execute(
            """SELECT tx_json, block_height FROM transactions
               WHERE sender = ? OR recipient = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (addr_hex, addr_hex, limit),
        ).fetchall()
        results = []
        for row in rows:
            tx_dict = json.loads(row["tx_json"])
            tx_dict["block_height"] = row["block_height"]
            results.append(tx_dict)
        return results

    def get_latest_blocks(self, count: int = 10) -> List[Block]:
        """Get the most recent blocks."""
        rows = self.db.execute(
            """SELECT hash, height, header_json FROM blocks
               ORDER BY height DESC LIMIT ?""",
            (count,),
        ).fetchall()
        blocks = []
        for row in rows:
            header = BlockHeader.from_dict(json.loads(row["header_json"]))
            txs = self.get_transactions_by_block_height(row["height"])
            stored_hash = bytes.fromhex(row["hash"]) if row["hash"] else None
            blocks.append(Block(header=header, transactions=txs, _cached_hash=stored_hash))
        return blocks

    def get_chain_height(self) -> int:
        """Get the current chain height."""
        return self.db.get_chain_height()

    def update_block_hash(self, height: int, new_hash_hex: str):
        """Update the stored hash for a block (genesis hash reconciliation)."""
        self.db.execute(
            "UPDATE blocks SET hash = ? WHERE height = ?",
            (new_hash_hex, height),
        )
        self.db.conn.commit()

    def block_exists(self, height: int) -> bool:
        """Check if a block at given height exists."""
        row = self.db.execute(
            "SELECT 1 FROM blocks WHERE height = ?", (height,)
        ).fetchone()
        return row is not None
