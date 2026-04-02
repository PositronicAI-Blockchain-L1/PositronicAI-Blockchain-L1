"""
Positronic - Cross-Chain Bridge Interface (Abstraction Layer)

NOTE: This module provides the interface and data structures for cross-chain
operations. External blockchain connectivity (Ethereum, Bitcoin, etc.) is NOT
yet implemented. Currently operates with in-memory simulation for development
and testing purposes. No web3.py, ethers, or external RPC calls are made.

Phase 1 (current): Hash anchoring data structures and in-memory simulation.
    Defines the data model for anchoring ASF block/transaction hashes on
    external chains (Ethereum, Bitcoin) and verifying external proofs.
    The interface is designed so that a real connector (web3.py, Bitcoin RPC,
    etc.) can be plugged in without changing the public API.

    What IS implemented:
        - AnchoredHash / ExternalProof data structures
        - In-memory anchor storage and retrieval
        - Proof submission and placeholder verification

    What is NOT yet implemented:
        - Actual RPC/WebSocket connections to Ethereum, Bitcoin, Polygon, BSC
        - On-chain transaction submission for hash anchoring
        - Merkle proof verification against external block headers

Phase 2 (roadmap): Lock/mint asset transfer bridge.
    Data structures and state machine for token transfers between ASF and
    external chains via lock-on-source / mint-on-target pattern. See
    LockMintBridge below.

    What IS implemented:
        - Lock/mint/burn/release state machine with full lifecycle
        - Relayer quorum management (M-of-N confirmation)
        - Fee calculation and challenge period logic
        - Integration with ASF StateManager for balance deductions

    What is NOT yet implemented:
        - Relayer daemon that watches external chains
        - Smart contract deployment/interaction on target chains
        - Actual wrapped-token minting on Ethereum/Polygon/BSC
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import IntEnum
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


class TargetChain(IntEnum):
    ETHEREUM = 0
    BITCOIN = 1
    POLYGON = 2
    BSC = 3


class AnchorStatus(IntEnum):
    PENDING = 0
    CONFIRMED = 1
    FAILED = 2


@dataclass
class AnchoredHash:
    """A hash anchored on an external chain."""
    anchor_id: str
    source_hash: bytes  # Hash from ASF chain
    target_chain: TargetChain
    target_tx_hash: str = ""  # TX hash on target chain
    status: AnchorStatus = AnchorStatus.PENDING
    created_at: float = 0.0
    confirmed_at: float = 0.0
    block_height: int = 0  # ASF block height

    def confirm(self, target_tx_hash: str):
        self.target_tx_hash = target_tx_hash
        self.status = AnchorStatus.CONFIRMED
        self.confirmed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "anchor_id": self.anchor_id,
            "source_hash": self.source_hash.hex(),
            "target_chain": self.target_chain.name,
            "target_tx_hash": self.target_tx_hash,
            "status": self.status.name,
            "created_at": self.created_at,
            "confirmed_at": self.confirmed_at,
            "block_height": self.block_height,
        }


@dataclass
class ExternalProof:
    """Proof from an external blockchain."""
    proof_id: str
    source_chain: TargetChain
    tx_hash: str
    block_number: int = 0
    data: bytes = b""
    verified: bool = False
    verified_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "proof_id": self.proof_id,
            "source_chain": self.source_chain.name,
            "tx_hash": self.tx_hash,
            "block_number": self.block_number,
            "verified": self.verified,
            "verified_at": self.verified_at,
        }


class CrossChainBridge:
    """Cross-chain bridge interface for hash anchoring and proof verification.

    This class provides the abstraction layer for cross-chain hash anchoring.
    All operations currently run against in-memory data structures.

    To connect to real external chains, implement a connector class and pass
    it to the constructor (see TODO below).

    NOTE: No actual blockchain RPC calls are made. The ``anchor_hash`` method
    stores anchors locally; a real implementation would broadcast a transaction
    to the target chain. Similarly, ``verify_external_proof`` performs only a
    basic presence check -- real verification would validate Merkle proofs
    against external block headers.
    """

    def __init__(self):
        self._anchors: Dict[str, AnchoredHash] = {}
        self._external_proofs: Dict[str, ExternalProof] = {}
        self._total_anchored: int = 0
        self._total_verified: int = 0
        # TODO: Connect to actual external chains — accept an optional
        # connector/provider (e.g. web3.py instance, Bitcoin RPC client)
        # that handles real transaction submission and proof retrieval.

    def _generate_id(self, prefix: str) -> str:
        data = f"{prefix}_{time.time()}".encode()
        return prefix + "_" + hashlib.sha256(data).hexdigest()[:12]

    def anchor_hash(self, source_hash: bytes, target_chain: TargetChain,
                    block_height: int = 0) -> AnchoredHash:
        """Anchor a hash from ASF chain to an external chain.

        Currently stores the anchor in memory only.  A real implementation
        would submit a transaction to ``target_chain`` containing
        ``source_hash`` and return the external TX hash via
        :meth:`confirm_anchor`.

        .. todo:: Submit anchor transaction to external chain via RPC.
        """
        # TODO: Connect to actual external chains — replace in-memory
        # storage with a real transaction broadcast to target_chain.
        anchor_id = self._generate_id("anchor")
        anchor = AnchoredHash(
            anchor_id=anchor_id,
            source_hash=source_hash,
            target_chain=target_chain,
            created_at=time.time(),
            block_height=block_height,
        )
        self._anchors[anchor_id] = anchor
        self._total_anchored += 1
        return anchor

    def confirm_anchor(self, anchor_id: str, target_tx_hash: str) -> bool:
        """Confirm that a hash was successfully anchored."""
        anchor = self._anchors.get(anchor_id)
        if anchor is None:
            return False
        anchor.confirm(target_tx_hash)
        return True

    def get_anchor(self, anchor_id: str) -> Optional[AnchoredHash]:
        return self._anchors.get(anchor_id)

    def get_anchors_for_chain(self, chain: TargetChain) -> List[dict]:
        return [a.to_dict() for a in self._anchors.values() if a.target_chain == chain]

    def submit_external_proof(self, source_chain: TargetChain,
                               tx_hash: str, block_number: int = 0,
                               data: bytes = b"") -> ExternalProof:
        """Submit a proof from an external chain for verification."""
        proof_id = self._generate_id("proof")
        proof = ExternalProof(
            proof_id=proof_id,
            source_chain=source_chain,
            tx_hash=tx_hash,
            block_number=block_number,
            data=data,
        )
        self._external_proofs[proof_id] = proof
        return proof

    def verify_external_proof(self, proof_id: str) -> bool:
        """Verify an external proof.

        **PLACEHOLDER** -- This method currently performs only a basic
        presence/sanity check (tx_hash is non-empty).  A real implementation
        would:
          1. Fetch the transaction receipt from the source chain via RPC.
          2. Validate the Merkle proof against the block header.
          3. Confirm sufficient block confirmations for finality.

        Returns True if the proof passes the (placeholder) check.

        .. todo:: Implement real Merkle proof verification against external
                  block headers via RPC.
        """
        # TODO: Connect to actual external chains — fetch TX receipt from
        # source_chain and verify Merkle inclusion proof.
        proof = self._external_proofs.get(proof_id)
        if proof is None:
            return False
        if not proof.tx_hash:
            return False
        proof.verified = True
        proof.verified_at = time.time()
        self._total_verified += 1
        return True

    def get_stats(self) -> dict:
        return {
            "total_anchored": self._total_anchored,
            "total_verified": self._total_verified,
            "pending_anchors": sum(1 for a in self._anchors.values() if a.status == AnchorStatus.PENDING),
            "confirmed_anchors": sum(1 for a in self._anchors.values() if a.status == AnchorStatus.CONFIRMED),
            "external_proofs": len(self._external_proofs),
            "phase": 1,  # Hash anchoring only (Phase 1)
        }


class LockStatus(IntEnum):
    PENDING = 0       # Locked, waiting for relayer confirmations
    CONFIRMED = 1     # Quorum reached, ready to mint
    MINTED = 2        # Wrapped tokens minted on target
    CHALLENGED = 3    # Under fraud challenge
    RELEASED = 4      # Burned and released back to source
    EXPIRED = 5       # Challenge period passed without minting


@dataclass
class LockRecord:
    """Record of tokens locked for bridging."""
    lock_id: str
    sender: bytes
    amount: int
    fee: int
    target_chain: TargetChain
    recipient_external: str
    lock_hash: bytes  # H(sender || amount || nonce)
    status: LockStatus = LockStatus.PENDING
    confirmations: List[bytes] = field(default_factory=list)  # Relayer addresses
    created_at: float = 0.0
    confirmed_at: float = 0.0
    minted_at: float = 0.0
    nonce: int = 0

    @property
    def is_quorum(self) -> bool:
        from positronic.constants import BRIDGE_QUORUM
        return len(self.confirmations) >= BRIDGE_QUORUM

    def to_dict(self) -> dict:
        return {
            "lock_id": self.lock_id,
            "sender": self.sender.hex(),
            "amount": self.amount,
            "fee": self.fee,
            "target_chain": self.target_chain.name,
            "recipient_external": self.recipient_external,
            "lock_hash": self.lock_hash.hex(),
            "status": self.status.name,
            "confirmations": len(self.confirmations),
            "created_at": self.created_at,
        }


class LockMintBridge:
    """Phase 2: Lock/Mint asset transfer bridge (in-memory state machine).

    Implements the full lock/confirm/mint/burn/release lifecycle and relayer
    quorum logic.  All state transitions run in-memory against the ASF
    StateManager.

    What IS implemented (and tested):
        - Token locking with fee calculation and balance deduction
        - M-of-N relayer quorum confirmation (3-of-5 default)
        - Minting, burning, and releasing lifecycle
        - Fraud challenge within 24-hour challenge period
        - Full round-trip state machine

    What is NOT yet implemented:
        - Relayer daemon process that monitors external chains
        - Actual smart-contract interaction on target chains
        - Wrapped-token minting on Ethereum/Polygon/BSC
        - External event listeners for burn/release triggers

    The interface is designed so that real external-chain connectors can be
    added without changing the public API.  A future ``ExternalChainConnector``
    would be injected alongside the StateManager.

    Features:
    - Relayer quorum (3-of-5 default)
    - 0.3% bridge fee
    - 24-hour challenge period
    - Fraud proof mechanism
    """

    def __init__(self, state_manager=None, blockchain=None, db_path: str = None,
                 connector=None):
        self._locks: Dict[str, LockRecord] = {}
        self._relayers: Dict[bytes, bool] = {}  # address → active
        self._nonce: int = 0
        self._total_locked: int = 0
        self._total_minted: int = 0
        self._total_released: int = 0
        self._state_manager = state_manager
        self._blockchain = blockchain  # for recording system TXs on-chain
        self._connector = connector  # ExternalChainConnector (optional)
        self._db = None
        if db_path:
            self._init_bridge_db(db_path)
            self._load_locks_from_db()

    # ------------------------------------------------------------------
    # Relayer management
    # ------------------------------------------------------------------

    def register_relayer(self, address: bytes) -> bool:
        """Register a new bridge relayer."""
        if address in self._relayers:
            return False
        self._relayers[address] = True
        return True

    def remove_relayer(self, address: bytes) -> bool:
        """Remove a bridge relayer."""
        if address not in self._relayers:
            return False
        self._relayers[address] = False
        return True

    @property
    def active_relayer_count(self) -> int:
        return sum(1 for active in self._relayers.values() if active)

    def is_relayer(self, address: bytes) -> bool:
        return self._relayers.get(address, False)

    # ------------------------------------------------------------------
    # Lock tokens
    # ------------------------------------------------------------------

    def lock_tokens(self, sender: bytes, amount: int,
                    target_chain: TargetChain,
                    recipient_external: str) -> Optional[LockRecord]:
        """Lock ASF tokens for bridging to an external chain.

        Deducts amount from sender's balance (via StateManager),
        computes a 0.3% fee, and creates a lock record in memory.

        NOTE: The lock is recorded locally only.  A real implementation
        would also emit an event / notify relayers watching the chain.

        Returns LockRecord on success, None on failure.
        """
        # TODO: Connect to actual external chains — emit lock event
        # that external relayer daemons can observe.
        from positronic.constants import BRIDGE_MIN_LOCK, BRIDGE_FEE_BPS

        if amount < BRIDGE_MIN_LOCK:
            return None

        # Compute fee (basis points)
        fee = (amount * BRIDGE_FEE_BPS) // 10_000
        total_deduct = amount  # Fee comes from the locked amount

        # Deduct from sender balance if state manager is available
        if self._state_manager is not None:
            if not self._state_manager.sub_balance(sender, total_deduct):
                return None

        # Generate lock
        self._nonce += 1
        lock_data = sender + amount.to_bytes(32, "big") + self._nonce.to_bytes(8, "big")
        lock_hash = hashlib.sha256(lock_data).digest()
        lock_id = "lock_" + lock_hash.hex()[:16]

        lock = LockRecord(
            lock_id=lock_id,
            sender=sender,
            amount=amount - fee,  # Net amount after fee
            fee=fee,
            target_chain=target_chain,
            recipient_external=recipient_external,
            lock_hash=lock_hash,
            created_at=time.time(),
            nonce=self._nonce,
        )
        self._locks[lock_id] = lock
        self._total_locked += amount
        return lock

    # ------------------------------------------------------------------
    # Relayer confirmation
    # ------------------------------------------------------------------

    def confirm_lock(self, lock_id: str, relayer: bytes) -> bool:
        """A relayer confirms a lock. Returns True if this pushed to quorum."""
        lock = self._locks.get(lock_id)
        if lock is None:
            return False
        if lock.status != LockStatus.PENDING:
            return False
        if not self.is_relayer(relayer):
            return False
        if relayer in lock.confirmations:
            return False  # No double-confirm

        lock.confirmations.append(relayer)
        self._save_lock(lock_id)

        if lock.is_quorum:
            lock.status = LockStatus.CONFIRMED
            lock.confirmed_at = time.time()

        return True

    # ------------------------------------------------------------------
    # Mint wrapped tokens (after quorum)
    # ------------------------------------------------------------------

    def mint_tokens(self, lock_id: str) -> bool:
        """Mint wrapped-ASF on target chain after quorum confirmation.

        Currently only updates the in-memory lock status to MINTED.
        A real implementation would call a mint function on the target
        chain's wrapped-token smart contract.

        Returns True on success.
        """
        # TODO: Connect to actual external chains — call mint() on the
        # target chain's wrapped-ASF smart contract.
        lock = self._locks.get(lock_id)
        if lock is None:
            return False
        if lock.status != LockStatus.CONFIRMED:
            return False

        lock.status = LockStatus.MINTED
        lock.minted_at = time.time()
        self._total_minted += lock.amount
        self._save_lock(lock_id)

        # Record on-chain so other nodes see the bridge mint
        if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
            self._blockchain._create_system_tx(
                0, lock.sender, lock.sender, lock.amount, b"bridge_mint",
            )

        # Attempt external chain mint if connector is available
        if self._connector is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        self._connector.mint_wrapped(
                            target_chain=lock.target_chain.value,
                            recipient=lock.recipient_external,
                            amount=lock.amount,
                            lock_id=lock_id,
                        )
                    )
                else:
                    loop.run_until_complete(
                        self._connector.mint_wrapped(
                            target_chain=lock.target_chain.value,
                            recipient=lock.recipient_external,
                            amount=lock.amount,
                            lock_id=lock_id,
                        )
                    )
            except Exception as e:
                logger.warning(f"External mint failed for {lock_id}: {e}")

        return True

    # ------------------------------------------------------------------
    # Burn and release
    # ------------------------------------------------------------------

    def burn_and_release(self, lock_id: str) -> bool:
        """Burn wrapped tokens and release locked ASF back to sender.

        Currently only updates in-memory state (credits balance back via
        StateManager). A real implementation would first verify the burn
        event on the target chain before releasing locked tokens.

        Returns True on success.
        """
        # TODO: Connect to actual external chains — verify burn event on
        # target chain before releasing locked ASF tokens.
        lock = self._locks.get(lock_id)
        if lock is None:
            return False
        if lock.status != LockStatus.MINTED:
            return False

        # Verify burn on external chain if connector available
        if self._connector is not None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule async verification — actual release happens via system TX
                    asyncio.create_task(self._async_verify_and_release(lock_id))
                    return True
            except Exception as e:
                logger.warning(f"Burn verification error for {lock_id}: {e}")
        # No connector — trust relayer quorum (relay-only mode)

        # Credit back to sender (net amount, fee was already taken)
        if self._state_manager is not None:
            self._state_manager.add_balance(lock.sender, lock.amount)

        # Record on-chain so other nodes see the bridge release
        if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
            self._blockchain._create_system_tx(
                0, b'\x00' * 20, lock.sender, lock.amount, b"bridge_release",
            )

        lock.status = LockStatus.RELEASED
        self._total_released += lock.amount
        self._save_lock(lock_id)
        return True

    async def _async_verify_and_release(self, lock_id: str):
        """Verify external burn and then release locked tokens."""
        lock = self._locks.get(lock_id)
        if not lock:
            return
        verified = await self._connector.verify_burn(
            target_chain=lock.target_chain.value,
            lock_id=lock_id,
        )
        if verified:
            if self._state_manager:
                self._state_manager.add_balance(lock.sender, lock.amount)
            lock.status = LockStatus.RELEASED
            self._total_released += lock.amount
            self._save_lock(lock_id)
            logger.info(f"Lock {lock_id} released after burn verified")
        else:
            logger.warning(f"Burn NOT verified for {lock_id} — tokens remain locked")

    # ------------------------------------------------------------------
    # Challenge (fraud proof)
    # ------------------------------------------------------------------

    def challenge_lock(self, lock_id: str) -> bool:
        """Challenge a lock during the challenge period.

        Moves lock to CHALLENGED status, preventing minting.
        """
        lock = self._locks.get(lock_id)
        if lock is None:
            return False
        if lock.status not in (LockStatus.PENDING, LockStatus.CONFIRMED):
            return False

        from positronic.constants import BRIDGE_CHALLENGE_PERIOD
        # Only challenge within the challenge period
        if time.time() - lock.created_at > BRIDGE_CHALLENGE_PERIOD:
            return False

        lock.status = LockStatus.CHALLENGED
        self._save_lock(lock_id)
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_lock(self, lock_id: str) -> Optional[LockRecord]:
        return self._locks.get(lock_id)

    def get_locks_by_sender(self, sender: bytes) -> List[dict]:
        return [
            l.to_dict() for l in self._locks.values()
            if l.sender == sender
        ]

    def get_stats(self) -> dict:
        return {
            "total_locked": self._total_locked,
            "total_minted": self._total_minted,
            "total_released": self._total_released,
            "active_locks": sum(
                1 for l in self._locks.values()
                if l.status in (LockStatus.PENDING, LockStatus.CONFIRMED)
            ),
            "relayer_count": self.active_relayer_count,
            "phase": 2,
        }

    def _init_bridge_db(self, db_path: str):
        """Initialize SQLite storage for bridge lock records."""
        import sqlite3, os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS bridge_locks (
                lock_id TEXT PRIMARY KEY,
                data    TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _load_locks_from_db(self):
        """Load persisted lock records on startup."""
        if self._db is None:
            return
        import json
        rows = self._db.execute("SELECT lock_id, data FROM bridge_locks").fetchall()
        for lock_id, data_json in rows:
            try:
                d = json.loads(data_json)
                record = LockRecord(
                    lock_id=d["lock_id"],
                    sender=bytes.fromhex(d["sender"]),
                    amount=d["amount"],
                    fee=d.get("fee", 0),
                    target_chain=TargetChain(d.get("target_chain", 0)),
                    recipient_external=d.get("recipient_external", ""),
                    lock_hash=bytes.fromhex(d.get("lock_hash", "")),
                    status=LockStatus(d.get("status", 0)),
                    confirmations=[bytes.fromhex(c) for c in d.get("confirmations", [])],
                    created_at=d.get("created_at", 0.0),
                    confirmed_at=d.get("confirmed_at", 0.0),
                    minted_at=d.get("minted_at", 0.0),
                    nonce=d.get("nonce", 0),
                )
                self._locks[lock_id] = record
            except Exception as e:
                logger.warning("Failed to load lock %s: %s", lock_id, e)

    def _save_lock(self, lock_id: str):
        """Persist a lock record to SQLite."""
        if self._db is None:
            return
        lock = self._locks.get(lock_id)
        if not lock:
            return
        import json
        data = {
            "lock_id":            lock.lock_id,
            "sender":             lock.sender.hex(),
            "amount":             lock.amount,
            "fee":                lock.fee,
            "target_chain":       lock.target_chain.value,
            "recipient_external": lock.recipient_external,
            "lock_hash":          lock.lock_hash.hex(),
            "status":             lock.status.value,
            "confirmations":      [c.hex() for c in lock.confirmations],
            "created_at":         lock.created_at,
            "confirmed_at":       lock.confirmed_at,
            "minted_at":          lock.minted_at,
            "nonce":              lock.nonce,
        }
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO bridge_locks (lock_id, data) VALUES (?, ?)",
                (lock_id, json.dumps(data))
            )
            self._db.commit()
        except Exception as e:
            logger.warning("Failed to save lock %s: %s", lock_id, e)
