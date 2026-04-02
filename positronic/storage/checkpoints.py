"""
Positronic - Checkpoints & Snapshots
Enables fast node bootstrap by providing periodic state checkpoints
and full state snapshots. New nodes can sync from a checkpoint instead
of replaying the entire blockchain from genesis.

Superior to Bitcoin: Built-in checkpoint system with AI integrity verification.

Checkpoint: A verified block hash at a specific height (lightweight)
Snapshot: Full state at a specific height (heavy but complete)
Supports transparent AES-256-GCM encryption at rest.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from positronic.crypto.hashing import sha512
from positronic.constants import (
    CHECKPOINT_INTERVAL,
    SNAPSHOT_INTERVAL,
    HASH_SIZE,
)
from positronic.storage.database import DatabaseEncryption

logger = logging.getLogger("positronic.storage.checkpoints")


@dataclass
class Checkpoint:
    """
    A verified point-in-time reference for the blockchain.
    Contains the block hash, state root, and validator set
    at a specific height.
    """
    height: int
    block_hash: bytes
    state_root: bytes
    transactions_root: bytes
    validator_set_hash: bytes  # Hash of active validator set
    total_supply_at_height: int = 0
    total_accounts: int = 0
    created_at: float = 0.0
    verified_by: int = 0  # Number of validators who verified this
    finalized: bool = False

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def checkpoint_hash(self) -> bytes:
        """Unique hash of this checkpoint."""
        data = (
            self.height.to_bytes(8, "big")
            + self.block_hash
            + self.state_root
            + self.validator_set_hash
        )
        return sha512(data)

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "block_hash": self.block_hash.hex(),
            "state_root": self.state_root.hex(),
            "transactions_root": self.transactions_root.hex(),
            "validator_set_hash": self.validator_set_hash.hex(),
            "total_supply_at_height": self.total_supply_at_height,
            "total_accounts": self.total_accounts,
            "created_at": self.created_at,
            "verified_by": self.verified_by,
            "finalized": self.finalized,
            "checkpoint_hash": self.checkpoint_hash.hex(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            height=d["height"],
            block_hash=bytes.fromhex(d["block_hash"]),
            state_root=bytes.fromhex(d["state_root"]),
            transactions_root=bytes.fromhex(d.get("transactions_root", "00" * HASH_SIZE)),
            validator_set_hash=bytes.fromhex(d.get("validator_set_hash", "00" * HASH_SIZE)),
            total_supply_at_height=d.get("total_supply_at_height", 0),
            total_accounts=d.get("total_accounts", 0),
            created_at=d.get("created_at", 0.0),
            verified_by=d.get("verified_by", 0),
            finalized=d.get("finalized", False),
        )


@dataclass
class StateSnapshot:
    """
    A full state snapshot at a specific height.
    Contains all account states, contract storage, etc.
    Used for fast node bootstrap.
    """
    height: int
    block_hash: bytes
    state_root: bytes
    accounts_count: int
    contracts_count: int
    snapshot_hash: bytes  # Hash of the complete snapshot data
    size_bytes: int = 0
    created_at: float = 0.0
    compressed: bool = False

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "height": self.height,
            "block_hash": self.block_hash.hex(),
            "state_root": self.state_root.hex(),
            "accounts_count": self.accounts_count,
            "contracts_count": self.contracts_count,
            "snapshot_hash": self.snapshot_hash.hex(),
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StateSnapshot":
        return cls(
            height=d["height"],
            block_hash=bytes.fromhex(d["block_hash"]),
            state_root=bytes.fromhex(d["state_root"]),
            accounts_count=d["accounts_count"],
            contracts_count=d["contracts_count"],
            snapshot_hash=bytes.fromhex(d["snapshot_hash"]),
            size_bytes=d.get("size_bytes", 0),
            created_at=d.get("created_at", 0.0),
            compressed=d.get("compressed", False),
        )


class CheckpointManager:
    """
    Manages blockchain checkpoints and state snapshots.
    Supports optional AES-256-GCM encryption at rest.

    Features:
    - Automatic checkpoint creation at CHECKPOINT_INTERVAL
    - State snapshots at SNAPSHOT_INTERVAL
    - Checkpoint verification by validators
    - Fast sync from latest checkpoint
    - Checkpoint chain validation
    """

    def __init__(
        self,
        db_path: str = None,
        encryption_password: Optional[str] = None,
    ):
        self._checkpoints: Dict[int, Checkpoint] = {}  # height -> checkpoint
        self._snapshots: Dict[int, StateSnapshot] = {}  # height -> snapshot
        self._latest_checkpoint_height: int = 0
        self._latest_snapshot_height: int = 0
        self._db = None
        self._encryption_password = encryption_password
        self._active_db_path = db_path
        if db_path:
            self._init_db(db_path)
            self._load_from_db()

    def _init_db(self, db_path: str):
        """Initialize SQLite database for checkpoint persistence."""
        import sqlite3

        self._in_memory = False
        if self._encryption_password and DatabaseEncryption.is_encrypted(db_path):
            self._db = DatabaseEncryption.load_to_memory_connection(
                db_path, self._encryption_password
            )
            self._in_memory = True
        else:
            self._active_db_path = db_path
            self._db = sqlite3.connect(self._active_db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                height INTEGER PRIMARY KEY,
                block_hash BLOB NOT NULL,
                state_root BLOB NOT NULL,
                transactions_root BLOB NOT NULL,
                validator_set_hash BLOB NOT NULL,
                total_supply INTEGER DEFAULT 0,
                total_accounts INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                verified_by INTEGER DEFAULT 0,
                finalized INTEGER DEFAULT 0
            )
        """)
        self._db.commit()

    @property
    def is_encrypted(self) -> bool:
        """Check if encryption is enabled."""
        return self._encryption_password is not None

    def _load_from_db(self):
        """Load all checkpoints from database on startup."""
        if self._db is None:
            return
        rows = self._db.execute(
            "SELECT * FROM checkpoints ORDER BY height"
        ).fetchall()
        for row in rows:
            cp = Checkpoint(
                height=row["height"],
                block_hash=row["block_hash"],
                state_root=row["state_root"],
                transactions_root=row["transactions_root"],
                validator_set_hash=row["validator_set_hash"],
                total_supply_at_height=row["total_supply"],
                total_accounts=row["total_accounts"],
                created_at=row["created_at"],
                verified_by=row["verified_by"],
                finalized=bool(row["finalized"]),
            )
            self._checkpoints[cp.height] = cp
            if cp.height > self._latest_checkpoint_height:
                self._latest_checkpoint_height = cp.height
        logger.info(f"Loaded {len(rows)} checkpoints from database")

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Persist a single checkpoint to database."""
        if self._db is None:
            return
        self._db.execute("""
            INSERT OR REPLACE INTO checkpoints
            (height, block_hash, state_root, transactions_root,
             validator_set_hash, total_supply, total_accounts,
             created_at, verified_by, finalized)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint.height,
            checkpoint.block_hash,
            checkpoint.state_root,
            checkpoint.transactions_root,
            checkpoint.validator_set_hash,
            checkpoint.total_supply_at_height,
            checkpoint.total_accounts,
            checkpoint.created_at,
            checkpoint.verified_by,
            int(checkpoint.finalized),
        ))
        import sqlite3
        try:
            self._db.commit()
        except sqlite3.OperationalError as e:
            err_msg = str(e).lower()
            if "disk" in err_msg or "full" in err_msg or "i/o" in err_msg:
                try:
                    self._db.rollback()
                except Exception as rb_err:
                    logger.debug("Rollback failed during disk-full handling: %s", rb_err)
                from positronic.storage import DiskFullError
                raise DiskFullError(f"Checkpoint DB disk full: {e}") from e
            raise

    def should_create_checkpoint(self, height: int) -> bool:
        """Check if a checkpoint should be created at this height."""
        if height == 0:
            return True  # Always checkpoint genesis
        return height % CHECKPOINT_INTERVAL == 0

    def should_create_snapshot(self, height: int) -> bool:
        """Check if a snapshot should be created at this height."""
        return height > 0 and height % SNAPSHOT_INTERVAL == 0

    def create_checkpoint(
        self,
        height: int,
        block_hash: bytes,
        state_root: bytes,
        transactions_root: bytes = b"",
        validator_set_hash: bytes = b"",
        total_supply: int = 0,
        total_accounts: int = 0,
    ) -> Checkpoint:
        """Create a new checkpoint at the given height."""
        if not transactions_root:
            transactions_root = b"\x00" * HASH_SIZE
        if not validator_set_hash:
            validator_set_hash = b"\x00" * HASH_SIZE

        checkpoint = Checkpoint(
            height=height,
            block_hash=block_hash,
            state_root=state_root,
            transactions_root=transactions_root,
            validator_set_hash=validator_set_hash,
            total_supply_at_height=total_supply,
            total_accounts=total_accounts,
        )

        self._checkpoints[height] = checkpoint
        if height > self._latest_checkpoint_height:
            self._latest_checkpoint_height = height

        self._save_checkpoint(checkpoint)
        logger.info(f"Checkpoint created at height {height}")
        return checkpoint

    def verify_checkpoint(
        self,
        height: int,
        validator_address: bytes,
    ) -> bool:
        """Record that a validator has verified a checkpoint."""
        cp = self._checkpoints.get(height)
        if not cp:
            return False
        cp.verified_by += 1
        self._save_checkpoint(cp)
        return True

    def finalize_checkpoint(self, height: int) -> bool:
        """Mark a checkpoint as finalized (enough verifications)."""
        cp = self._checkpoints.get(height)
        if not cp:
            return False
        cp.finalized = True
        self._save_checkpoint(cp)
        return True

    def register_snapshot(
        self,
        height: int,
        block_hash: bytes,
        state_root: bytes,
        accounts_count: int,
        contracts_count: int,
        snapshot_data_hash: bytes,
        size_bytes: int = 0,
    ) -> StateSnapshot:
        """Register a state snapshot."""
        snapshot = StateSnapshot(
            height=height,
            block_hash=block_hash,
            state_root=state_root,
            accounts_count=accounts_count,
            contracts_count=contracts_count,
            snapshot_hash=snapshot_data_hash,
            size_bytes=size_bytes,
        )

        self._snapshots[height] = snapshot
        if height > self._latest_snapshot_height:
            self._latest_snapshot_height = height

        logger.info(
            f"Snapshot registered at height {height} "
            f"({accounts_count} accounts, {size_bytes} bytes)"
        )
        return snapshot

    def get_checkpoint(self, height: int) -> Optional[Checkpoint]:
        return self._checkpoints.get(height)

    def get_snapshot(self, height: int) -> Optional[StateSnapshot]:
        return self._snapshots.get(height)

    @property
    def latest_checkpoint(self) -> Optional[Checkpoint]:
        if self._latest_checkpoint_height == 0 and 0 not in self._checkpoints:
            return None
        return self._checkpoints.get(self._latest_checkpoint_height)

    @property
    def latest_snapshot(self) -> Optional[StateSnapshot]:
        if self._latest_snapshot_height == 0:
            return None
        return self._snapshots.get(self._latest_snapshot_height)

    @property
    def latest_finalized_checkpoint(self) -> Optional[Checkpoint]:
        """Get the latest finalized checkpoint."""
        finalized = [
            cp for cp in self._checkpoints.values()
            if cp.finalized
        ]
        if not finalized:
            return None
        return max(finalized, key=lambda c: c.height)

    def get_sync_start_point(self) -> Tuple[int, Optional[bytes]]:
        """
        Get the best starting point for a new node sync.
        Returns (height, state_root) from the latest checkpoint/snapshot.
        """
        # Prefer latest finalized checkpoint
        cp = self.latest_finalized_checkpoint
        if cp:
            return cp.height, cp.state_root

        # Fall back to latest checkpoint
        cp = self.latest_checkpoint
        if cp:
            return cp.height, cp.state_root

        # No checkpoints, sync from genesis
        return 0, None

    def get_all_checkpoints(self) -> List[dict]:
        """Get all checkpoints as dicts."""
        return [
            cp.to_dict() for cp in sorted(
                self._checkpoints.values(), key=lambda c: c.height
            )
        ]

    def get_all_snapshots(self) -> List[dict]:
        """Get all snapshots as dicts."""
        return [
            s.to_dict() for s in sorted(
                self._snapshots.values(), key=lambda s: s.height
            )
        ]

    def validate_checkpoint_chain(self) -> bool:
        """
        Validate that checkpoints form a consistent chain.
        Each checkpoint's height should be a multiple of CHECKPOINT_INTERVAL.
        """
        heights = sorted(self._checkpoints.keys())
        for h in heights:
            if h != 0 and h % CHECKPOINT_INTERVAL != 0:
                logger.warning(f"Invalid checkpoint height: {h}")
                return False
        return True

    def encrypt_at_rest(self) -> None:
        """Re-encrypt the database file on disk (call on shutdown)."""
        if not self._encryption_password or not self._active_db_path:
            return
        if self._in_memory:
            DatabaseEncryption.save_from_memory_connection(
                self._db,
                self._active_db_path,
                self._encryption_password,
            )
            return
        try:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.debug("WAL checkpoint failed during encrypt_at_rest: %s", e)
        if self._db:
            self._db.close()
            self._db = None
        DatabaseEncryption.encrypt_file(self._active_db_path, self._encryption_password)

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None

    # ================================================================
    # Phase 33: State Sync & Checkpoint Verification
    # ================================================================

    def export_state_snapshot(self, height: int, state_db, state: 'StateManager') -> bytes:
        """Export compressed state at height as zlib-compressed JSON.

        Returns bytes containing all accounts, contract code, and storage
        at the given height, compressed with zlib.
        """
        import json, zlib
        snapshot = {
            'height': height,
            'accounts': {},
            'contract_code': {},
            'contract_storage': {},
        }
        # Serialize all accounts
        for addr, acc in state.accounts.items():
            snapshot['accounts'][addr.hex()] = acc.to_dict()
        # Serialize contract code
        for code_hash, bytecode in state.contract_code.items():
            snapshot['contract_code'][code_hash.hex()] = bytecode.hex()
        # Serialize contract storage
        for addr, storage in state.contract_storage.items():
            snapshot['contract_storage'][addr.hex()] = {
                k.hex(): v.hex() for k, v in storage.items()
            }
        data = json.dumps(snapshot, sort_keys=True).encode()
        return zlib.compress(data, level=6)

    def import_state_snapshot(self, snapshot_data: bytes, expected_state_root: bytes,
                              state_db, state: 'StateManager') -> bool:
        """Import snapshot, verify state root matches.
        Returns True if import succeeded and state root verified."""
        import json, zlib
        from positronic.chain.account import Account
        try:
            data = zlib.decompress(snapshot_data)
            snapshot = json.loads(data)
        except Exception:
            return False

        # Clear current state
        state.accounts.clear()
        state.contract_code.clear()
        state.contract_storage.clear()

        # Load accounts
        for addr_hex, acc_dict in snapshot['accounts'].items():
            addr = bytes.fromhex(addr_hex)
            state.accounts[addr] = Account.from_dict(acc_dict)

        # Load contract code
        for hash_hex, code_hex in snapshot.get('contract_code', {}).items():
            state.contract_code[bytes.fromhex(hash_hex)] = bytes.fromhex(code_hex)

        # Load contract storage
        for addr_hex, storage in snapshot.get('contract_storage', {}).items():
            addr = bytes.fromhex(addr_hex)
            state.contract_storage[addr] = {
                bytes.fromhex(k): bytes.fromhex(v)
                for k, v in storage.items()
            }

        # Rebuild trie and verify state root
        state._rebuild_trie()
        computed_root = state.compute_state_root()
        if computed_root != expected_state_root:
            return False

        # Persist to DB
        state.save_to_db(state_db)
        return True

    def get_available_checkpoints(self) -> list:
        """Return list of checkpoint metadata dicts for RPC."""
        if self._db is None:
            # Fall back to in-memory checkpoints
            result = []
            for cp in sorted(self._checkpoints.values(), key=lambda c: c.height, reverse=True)[:100]:
                result.append({
                    'height': cp.height,
                    'block_hash': cp.block_hash.hex(),
                    'state_root': cp.state_root.hex(),
                    'created_at': cp.created_at,
                    'finalized': cp.finalized,
                })
            return result
        checkpoints = []
        rows = self._db.execute(
            "SELECT height, block_hash, state_root, created_at, finalized FROM checkpoints ORDER BY height DESC LIMIT 100"
        ).fetchall()
        for row in rows:
            checkpoints.append({
                'height': row['height'],
                'block_hash': row['block_hash'].hex() if isinstance(row['block_hash'], bytes) else row['block_hash'],
                'state_root': row['state_root'].hex() if isinstance(row['state_root'], bytes) else row['state_root'],
                'created_at': row['created_at'],
                'finalized': bool(row['finalized']),
            })
        return checkpoints

    def get_stats(self) -> dict:
        finalized = sum(1 for cp in self._checkpoints.values() if cp.finalized)
        return {
            "total_checkpoints": len(self._checkpoints),
            "finalized_checkpoints": finalized,
            "total_snapshots": len(self._snapshots),
            "latest_checkpoint_height": self._latest_checkpoint_height,
            "latest_snapshot_height": self._latest_snapshot_height,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "snapshot_interval": SNAPSHOT_INTERVAL,
            "encrypted": self.is_encrypted,
        }
