"""
Positronic - Network Message Types
Defines all P2P message types and their serialization.
Uses NETWORK_MAGIC prefix for message framing on the wire.
"""

import time
import json
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict

from positronic.crypto.hashing import sha512
from positronic.constants import NETWORK_MAGIC, PROTOCOL_VERSION
from positronic import __version__


class MessageType(IntEnum):
    """P2P network message types."""
    # Handshake
    HELLO = 0x01
    HELLO_ACK = 0x02
    DISCONNECT = 0x03

    # Block propagation
    NEW_BLOCK = 0x10
    GET_BLOCKS = 0x11
    BLOCKS = 0x12
    NEW_BLOCK_HASH = 0x13

    # Transaction propagation
    NEW_TX = 0x20
    GET_TXS = 0x21
    TXS = 0x22

    # Sync
    STATUS = 0x30
    GET_HEADERS = 0x31
    HEADERS = 0x32
    SYNC_REQUEST = 0x33
    SYNC_RESPONSE = 0x34
    REQUEST_STATUS = 0x35  # ask peer to send STATUS (fresh height) for re-sync

    # Consensus
    ATTESTATION = 0x40
    PROPOSAL = 0x41
    AI_SCORE_GOSSIP = 0x42

    # Peer discovery
    GET_PEERS = 0x50
    PEERS = 0x51

    # Health
    PING = 0x60
    PONG = 0x61

    # SPV/Light Client
    GET_MERKLE_PROOF = 0x70
    MERKLE_PROOF = 0x71
    GET_LIGHT_HEADERS = 0x72
    LIGHT_HEADERS = 0x73

    # Checkpoints
    GET_CHECKPOINT = 0x80
    CHECKPOINT = 0x81
    GET_STATE_SNAPSHOT = 0x82
    STATE_SNAPSHOT = 0x83

    # Multisig
    MULTISIG_PROPOSAL = 0x90
    MULTISIG_SIGNATURE = 0x91

    # Phase 5: Federated AI model sync
    AI_MODEL_SYNC = 0x45

    # Phase 33: Tendermint BFT voting
    PREVOTE = 0x43
    PRECOMMIT = 0x44

    # Phase 17: Compact block relay
    COMPACT_BLOCK = 0xA0
    GET_BLOCK_TXS = 0xA1

    # System TX propagation (stake/unstake/claim across nodes)
    SYSTEM_TX = 0xB0

    # Layer 2: PoW anti-Sybil handshake challenge
    POW_CHALLENGE = 0xC0
    POW_SOLUTION  = 0xC1


# Categories for filtering and rate limiting
HANDSHAKE_MESSAGES = {MessageType.HELLO, MessageType.HELLO_ACK, MessageType.DISCONNECT}
BLOCK_MESSAGES = {MessageType.NEW_BLOCK, MessageType.GET_BLOCKS, MessageType.BLOCKS, MessageType.NEW_BLOCK_HASH}
TX_MESSAGES = {MessageType.NEW_TX, MessageType.GET_TXS, MessageType.TXS}
SYNC_MESSAGES = {MessageType.STATUS, MessageType.GET_HEADERS, MessageType.HEADERS, MessageType.SYNC_REQUEST, MessageType.SYNC_RESPONSE, MessageType.REQUEST_STATUS}
CONSENSUS_MESSAGES = {MessageType.ATTESTATION, MessageType.PROPOSAL, MessageType.AI_SCORE_GOSSIP, MessageType.AI_MODEL_SYNC, MessageType.PREVOTE, MessageType.PRECOMMIT}
DISCOVERY_MESSAGES = {MessageType.GET_PEERS, MessageType.PEERS}
HEALTH_MESSAGES = {MessageType.PING, MessageType.PONG}
SPV_MESSAGES = {MessageType.GET_MERKLE_PROOF, MessageType.MERKLE_PROOF, MessageType.GET_LIGHT_HEADERS, MessageType.LIGHT_HEADERS}
CHECKPOINT_MESSAGES = {MessageType.GET_CHECKPOINT, MessageType.CHECKPOINT, MessageType.GET_STATE_SNAPSHOT, MessageType.STATE_SNAPSHOT}
MULTISIG_MESSAGES = {MessageType.MULTISIG_PROPOSAL, MessageType.MULTISIG_SIGNATURE}
COMPACT_BLOCK_MESSAGES = {MessageType.COMPACT_BLOCK, MessageType.GET_BLOCK_TXS}


@dataclass
class NetworkMessage:
    """A P2P network message with Positronic wire format."""
    msg_type: MessageType
    payload: dict
    sender_id: str = ""
    timestamp: float = 0.0
    nonce: int = 0
    msg_id: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.msg_id:
            raw = f"{self.msg_type}{self.timestamp}{self.nonce}".encode()
            self.msg_id = sha512(raw).hex()[:16]

    def serialize(self) -> str:
        """Serialize to JSON string for wire transmission."""
        return json.dumps({
            "magic": NETWORK_MAGIC.decode(),
            "version": PROTOCOL_VERSION,
            "type": int(self.msg_type),
            "payload": self.payload,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "msg_id": self.msg_id,
        })

    @classmethod
    def deserialize(cls, data: str) -> "NetworkMessage":
        """Deserialize from JSON string. Validates magic bytes."""
        d = json.loads(data)

        # Validate magic (backwards compatible: accept messages without magic)
        magic = d.get("magic", NETWORK_MAGIC.decode())
        if magic != NETWORK_MAGIC.decode():
            raise ValueError(f"Invalid network magic: {magic}")

        return cls(
            msg_type=MessageType(d["type"]),
            payload=d["payload"],
            sender_id=d.get("sender_id", ""),
            timestamp=d.get("timestamp", 0.0),
            nonce=d.get("nonce", 0),
            msg_id=d.get("msg_id", ""),
        )

    @property
    def age(self) -> float:
        """Message age in seconds."""
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Messages older than 5 minutes are considered expired."""
        return self.age > 300

    def __repr__(self) -> str:
        return (
            f"NetworkMessage(type={self.msg_type.name}, "
            f"sender={self.sender_id[:8]}..., "
            f"id={self.msg_id[:8]})"
        )


class MessageIDCache:
    """
    LRU cache for tracking recently seen message IDs to prevent
    re-processing and relay loops.
    """

    def __init__(self, max_size: int = 10_000):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size

    def has_seen(self, msg_id: str) -> bool:
        """Check if we've already seen this message."""
        return msg_id in self._cache

    def mark_seen(self, msg_id: str):
        """Mark a message as seen."""
        if msg_id in self._cache:
            self._cache.move_to_end(msg_id)
        else:
            self._cache[msg_id] = time.time()
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def add_and_check(self, msg_id: str) -> bool:
        """
        Atomically check and mark a message.
        Returns True if the message was NEW (not seen before).
        Returns False if it was already seen (duplicate).
        """
        if self.has_seen(msg_id):
            return False
        self.mark_seen(msg_id)
        return True

    def prune_expired(self, max_age: float = 300):
        """Remove entries older than max_age seconds."""
        now = time.time()
        expired = [
            mid for mid, ts in self._cache.items()
            if now - ts > max_age
        ]
        for mid in expired:
            del self._cache[mid]

    @property
    def size(self) -> int:
        return len(self._cache)


# === Message Factory Functions ===

def make_hello(
    node_id: str,
    chain_id: int,
    height: int,
    best_hash: str,
    protocol_version: int = PROTOCOL_VERSION,
    listen_port: int = 0,
) -> NetworkMessage:
    """Create a HELLO handshake message."""
    return NetworkMessage(
        msg_type=MessageType.HELLO,
        sender_id=node_id,
        payload={
            "node_id": node_id,
            "chain_id": chain_id,
            "height": height,
            "best_hash": best_hash,
            "protocol_version": protocol_version,
            "client": f"Positronic/{__version__}",
            "listen_port": listen_port,
        },
    )


def make_hello_ack(node_id: str, height: int, best_hash: str) -> NetworkMessage:
    """Create a HELLO_ACK response to a handshake."""
    return NetworkMessage(
        msg_type=MessageType.HELLO_ACK,
        sender_id=node_id,
        payload={"node_id": node_id, "height": height, "best_hash": best_hash},
    )


def make_disconnect(node_id: str, reason: str = "") -> NetworkMessage:
    """Create a DISCONNECT message with optional reason."""
    return NetworkMessage(
        msg_type=MessageType.DISCONNECT,
        sender_id=node_id,
        payload={"reason": reason},
    )


def make_new_block(block_dict: dict, node_id: str) -> NetworkMessage:
    """Create a NEW_BLOCK message to announce a new block."""
    return NetworkMessage(
        msg_type=MessageType.NEW_BLOCK,
        sender_id=node_id,
        payload={"block": block_dict},
    )


def make_new_block_hash(block_hash: str, height: int, node_id: str) -> NetworkMessage:
    """Create a NEW_BLOCK_HASH announcement (lightweight)."""
    return NetworkMessage(
        msg_type=MessageType.NEW_BLOCK_HASH,
        sender_id=node_id,
        payload={"block_hash": block_hash, "height": height},
    )


def make_new_tx(tx_dict: dict, node_id: str) -> NetworkMessage:
    """Create a NEW_TX message to announce a new transaction."""
    return NetworkMessage(
        msg_type=MessageType.NEW_TX,
        sender_id=node_id,
        payload={"transaction": tx_dict},
    )


def make_system_tx(method: str, params: list, node_id: str) -> NetworkMessage:
    """Create a SYSTEM_TX message to propagate stake/unstake/claim to peers."""
    return NetworkMessage(
        msg_type=MessageType.SYSTEM_TX,
        sender_id=node_id,
        payload={"method": method, "params": params},
    )


def make_get_blocks(start_height: int, count: int, node_id: str) -> NetworkMessage:
    """Request a range of full blocks."""
    return NetworkMessage(
        msg_type=MessageType.GET_BLOCKS,
        sender_id=node_id,
        payload={"start_height": start_height, "count": count},
    )


def make_blocks(blocks: List[dict], node_id: str) -> NetworkMessage:
    """Respond with a batch of full blocks."""
    return NetworkMessage(
        msg_type=MessageType.BLOCKS,
        sender_id=node_id,
        payload={"blocks": blocks},
    )


def make_get_headers(start_height: int, count: int, node_id: str) -> NetworkMessage:
    """Request a range of block headers (lighter than full blocks)."""
    return NetworkMessage(
        msg_type=MessageType.GET_HEADERS,
        sender_id=node_id,
        payload={"start_height": start_height, "count": count},
    )


def make_headers(headers: List[dict], node_id: str) -> NetworkMessage:
    """Respond with a batch of block headers."""
    return NetworkMessage(
        msg_type=MessageType.HEADERS,
        sender_id=node_id,
        payload={"headers": headers},
    )


def make_status(node_id: str, height: int, best_hash: str) -> NetworkMessage:
    """Create a STATUS message advertising chain state."""
    return NetworkMessage(
        msg_type=MessageType.STATUS,
        sender_id=node_id,
        payload={"height": height, "best_hash": best_hash},
    )


def make_request_status(node_id: str) -> NetworkMessage:
    """Ask peer to reply with STATUS so we get fresh chain height (for re-sync)."""
    return NetworkMessage(
        msg_type=MessageType.REQUEST_STATUS,
        sender_id=node_id,
        payload={},
    )


def make_sync_request(
    node_id: str,
    start_height: int,
    end_height: int,
) -> NetworkMessage:
    """Request a sync range of blocks."""
    return NetworkMessage(
        msg_type=MessageType.SYNC_REQUEST,
        sender_id=node_id,
        payload={
            "start_height": start_height,
            "end_height": end_height,
        },
    )


def make_sync_response(
    node_id: str,
    blocks: List[dict],
    has_more: bool = False,
) -> NetworkMessage:
    """Respond to a sync request with blocks."""
    return NetworkMessage(
        msg_type=MessageType.SYNC_RESPONSE,
        sender_id=node_id,
        payload={
            "blocks": blocks,
            "has_more": has_more,
        },
    )


def make_get_peers(node_id: str) -> NetworkMessage:
    """Request peer addresses from a peer."""
    return NetworkMessage(
        msg_type=MessageType.GET_PEERS,
        sender_id=node_id,
        payload={},
    )


def make_peers(peers: List[str], node_id: str) -> NetworkMessage:
    """Share peer addresses."""
    return NetworkMessage(
        msg_type=MessageType.PEERS,
        sender_id=node_id,
        payload={"peers": peers},
    )


def make_ping(node_id: str) -> NetworkMessage:
    """Create a PING message for liveness checking."""
    return NetworkMessage(
        msg_type=MessageType.PING,
        sender_id=node_id,
        payload={"time": time.time()},
    )


def make_pong(node_id: str, ping_time: float) -> NetworkMessage:
    """Create a PONG response with round-trip timing."""
    return NetworkMessage(
        msg_type=MessageType.PONG,
        sender_id=node_id,
        payload={"ping_time": ping_time, "pong_time": time.time()},
    )


def make_attestation(
    block_hash: str,
    height: int,
    validator_id: str,
    signature: str,
    node_id: str,
) -> NetworkMessage:
    """Create an ATTESTATION message for BFT finality."""
    return NetworkMessage(
        msg_type=MessageType.ATTESTATION,
        sender_id=node_id,
        payload={
            "block_hash": block_hash,
            "height": height,
            "validator_id": validator_id,
            "signature": signature,
        },
    )


def make_proposal(
    block_dict: dict,
    slot: int,
    epoch: int,
    node_id: str,
) -> NetworkMessage:
    """Create a block PROPOSAL message for consensus."""
    return NetworkMessage(
        msg_type=MessageType.PROPOSAL,
        sender_id=node_id,
        payload={
            "block": block_dict,
            "slot": slot,
            "epoch": epoch,
        },
    )


def make_ai_score_gossip(
    tx_hash: str, score: float, model_version: int, node_id: str
) -> NetworkMessage:
    """Gossip an AI validation score for a transaction."""
    return NetworkMessage(
        msg_type=MessageType.AI_SCORE_GOSSIP,
        sender_id=node_id,
        payload={
            "tx_hash": tx_hash,
            "score": score,
            "model_version": model_version,
        },
    )


# === Phase 17: Compact Block Relay ===

def make_compact_block(compact_block_dict: dict, node_id: str) -> NetworkMessage:
    """Create a COMPACT_BLOCK message for efficient block relay."""
    return NetworkMessage(
        msg_type=MessageType.COMPACT_BLOCK,
        sender_id=node_id,
        payload={"compact_block": compact_block_dict},
    )


def make_get_block_txs(
    block_hash: str, tx_hashes: List[str], node_id: str
) -> NetworkMessage:
    """Request missing transactions for compact block reconstruction."""
    return NetworkMessage(
        msg_type=MessageType.GET_BLOCK_TXS,
        sender_id=node_id,
        payload={"block_hash": block_hash, "tx_hashes": tx_hashes},
    )


# === Phase 33: State Sync & Checkpoint Verification ===

def make_get_state_snapshot(height: int, node_id: str) -> NetworkMessage:
    """Request a state snapshot at a specific height."""
    return NetworkMessage(
        msg_type=MessageType.GET_STATE_SNAPSHOT,
        sender_id=node_id,
        payload={"height": height},
    )


def make_state_snapshot(
    height: int, snapshot_data: str, state_root: str, node_id: str
) -> NetworkMessage:
    """Send a state snapshot (hex-encoded compressed data)."""
    return NetworkMessage(
        msg_type=MessageType.STATE_SNAPSHOT,
        sender_id=node_id,
        payload={
            "height": height,
            "snapshot_data": snapshot_data,
            "state_root": state_root,
        },
    )


# === Phase 33: Tendermint BFT Voting Messages ===

def make_prevote(
    block_hash: str,
    height: int,
    round_num: int,
    validator_id: str,
    signature: str,
    node_id: str,
) -> NetworkMessage:
    """Create a PREVOTE message for Tendermint BFT consensus."""
    return NetworkMessage(
        msg_type=MessageType.PREVOTE,
        sender_id=node_id,
        payload={
            "block_hash": block_hash,
            "height": height,
            "round": round_num,
            "validator_id": validator_id,
            "signature": signature,
        },
    )


def make_precommit(
    block_hash: str,
    height: int,
    round_num: int,
    validator_id: str,
    signature: str,
    node_id: str,
) -> NetworkMessage:
    """Create a PRECOMMIT message for Tendermint BFT consensus."""
    return NetworkMessage(
        msg_type=MessageType.PRECOMMIT,
        sender_id=node_id,
        payload={
            "block_hash": block_hash,
            "height": height,
            "round": round_num,
            "validator_id": validator_id,
            "signature": signature,
        },
    )


# === Phase 5: Federated AI Model Sync ===

def make_ai_model_sync(
    model_hash: str,
    weights_compressed: str,
    epoch: int,
    node_id: str,
) -> NetworkMessage:
    """Create an AI_MODEL_SYNC message for federated model weight exchange."""
    return NetworkMessage(
        msg_type=MessageType.AI_MODEL_SYNC,
        sender_id=node_id,
        payload={
            "model_hash": model_hash,
            "weights_compressed": weights_compressed,
            "epoch": epoch,
        },
    )


def make_pow_challenge(node_id: str, nonce: str) -> NetworkMessage:
    """Server → Client: issue a PoW challenge nonce before HELLO."""
    return NetworkMessage(
        msg_type=MessageType.POW_CHALLENGE,
        sender_id=node_id,
        payload={"nonce": nonce},
    )


def make_pow_solution(node_id: str, nonce: str, solution: str) -> NetworkMessage:
    """Client → Server: submit solution to PoW challenge."""
    return NetworkMessage(
        msg_type=MessageType.POW_SOLUTION,
        sender_id=node_id,
        payload={"nonce": nonce, "solution": solution},
    )
