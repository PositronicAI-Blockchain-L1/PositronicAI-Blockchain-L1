"""
Positronic - P2P Networking Layer

Core networking components for the Positronic blockchain:

- Node: Main orchestrator that ties together all components
- P2PServer: WebSocket-based P2P server with JSON-RPC
- PeerManager/Peer: Peer lifecycle and reputation management
- PeerDiscovery: Bootstrap and peer-exchange discovery
- BlockSync: Batch block synchronization protocol
- Mempool: Transaction pool with priority ordering
- ProtocolHandler: Message routing, rate limiting, and handshake
- NetworkMessage: Wire protocol message format

Message types:
  HELLO/HELLO_ACK/DISCONNECT - Handshake protocol
  NEW_BLOCK/GET_BLOCKS/BLOCKS - Block propagation
  NEW_TX/GET_TXS/TXS          - Transaction propagation
  STATUS/GET_HEADERS/HEADERS   - Chain state sync
  SYNC_REQUEST/SYNC_RESPONSE   - Batch sync protocol
  GET_PEERS/PEERS              - Peer discovery
  PING/PONG                    - Liveness checking
  ATTESTATION/PROPOSAL         - BFT consensus
  AI_SCORE_GOSSIP              - AI validation score gossip
"""

from positronic.network.node import Node
from positronic.network.server import P2PServer
from positronic.network.peer import (
    Peer,
    PeerState,
    PeerManager,
    ConnectionDirection,
)
from positronic.network.discovery import PeerDiscovery
from positronic.network.sync import BlockSync, SyncState, SyncPhase
from positronic.network.mempool import Mempool
from positronic.network.protocol import (
    ProtocolHandler,
    PeerProtocolState,
    PeerProtocolInfo,
)
from positronic.network.messages import (
    MessageType,
    NetworkMessage,
    MessageIDCache,
)

__all__ = [
    # Core
    "Node",
    "P2PServer",
    # Peers
    "Peer",
    "PeerState",
    "PeerManager",
    "ConnectionDirection",
    # Discovery
    "PeerDiscovery",
    # Sync
    "BlockSync",
    "SyncState",
    "SyncPhase",
    # Mempool
    "Mempool",
    # Protocol
    "ProtocolHandler",
    "PeerProtocolState",
    "PeerProtocolInfo",
    # Messages
    "MessageType",
    "NetworkMessage",
    "MessageIDCache",
]
