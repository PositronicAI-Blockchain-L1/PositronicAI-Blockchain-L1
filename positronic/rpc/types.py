"""
Positronic - JSON-RPC Types
Request/response types for the JSON-RPC API.
Compatible with Ethereum JSON-RPC for MetaMask integration.
"""

from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field, asdict


# --- JSON-RPC Base Types ---

@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request."""
    method: str
    params: List[Any] = field(default_factory=list)
    id: Any = 1
    jsonrpc: str = "2.0"

    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCRequest":
        return cls(
            method=data.get("method", ""),
            params=data.get("params", []),
            id=data.get("id", 1),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response."""
    result: Any = None
    error: Optional[Dict] = None
    id: Any = 1
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        d = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result
        return d


@dataclass
class JSONRPCError:
    """JSON-RPC error object."""
    code: int
    message: str
    data: Any = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


# --- Ethereum-Compatible Types (for MetaMask) ---

@dataclass
class EthBlock:
    """Ethereum-compatible block representation."""
    number: str  # hex
    hash: str  # 0x-prefixed
    parentHash: str
    timestamp: str  # hex
    miner: str
    gasLimit: str  # hex
    gasUsed: str  # hex
    transactions: List[Any] = field(default_factory=list)
    difficulty: str = "0x0"
    totalDifficulty: str = "0x0"
    size: str = "0x0"
    nonce: str = "0x0000000000000000"
    sha3Uncles: str = "0x" + "0" * 64
    logsBloom: str = "0x" + "0" * 512
    transactionsRoot: str = "0x" + "0" * 64
    stateRoot: str = "0x" + "0" * 64
    receiptsRoot: str = "0x" + "0" * 64
    uncles: List[str] = field(default_factory=list)
    extraData: str = "0x"
    mixHash: str = "0x" + "0" * 64
    baseFeePerGas: str = "0x0"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EthTransaction:
    """Ethereum-compatible transaction representation."""
    hash: str
    nonce: str  # hex
    blockHash: str
    blockNumber: str  # hex
    transactionIndex: str  # hex
    from_addr: str  # "from" is reserved in Python
    to: Optional[str]
    value: str  # hex
    gas: str  # hex
    gasPrice: str  # hex
    input: str = "0x"
    v: str = "0x0"
    r: str = "0x0"
    s: str = "0x0"
    type: str = "0x0"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_addr")
        return d


@dataclass
class EthTransactionReceipt:
    """Ethereum-compatible transaction receipt."""
    transactionHash: str
    transactionIndex: str  # hex
    blockHash: str
    blockNumber: str  # hex
    from_addr: str
    to: Optional[str]
    cumulativeGasUsed: str  # hex
    gasUsed: str  # hex
    contractAddress: Optional[str] = None
    logs: List[dict] = field(default_factory=list)
    logsBloom: str = "0x" + "0" * 512
    status: str = "0x1"
    type: str = "0x0"
    effectiveGasPrice: str = "0x1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["from"] = d.pop("from_addr")
        return d


@dataclass
class EthLog:
    """Ethereum-compatible log entry."""
    address: str
    topics: List[str] = field(default_factory=list)
    data: str = "0x"
    blockNumber: str = "0x0"
    transactionHash: str = "0x" + "0" * 64
    transactionIndex: str = "0x0"
    blockHash: str = "0x" + "0" * 64
    logIndex: str = "0x0"
    removed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CallParams:
    """Parameters for eth_call."""
    from_addr: Optional[str] = None
    to: Optional[str] = None
    gas: Optional[str] = None
    gasPrice: Optional[str] = None
    value: Optional[str] = None
    data: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CallParams":
        return cls(
            from_addr=data.get("from"),
            to=data.get("to"),
            gas=data.get("gas"),
            gasPrice=data.get("gasPrice"),
            value=data.get("value"),
            data=data.get("data"),
        )


@dataclass
class FilterParams:
    """Parameters for eth_newFilter."""
    fromBlock: str = "latest"
    toBlock: str = "latest"
    address: Optional[str] = None
    topics: Optional[List[Optional[str]]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "FilterParams":
        return cls(
            fromBlock=data.get("fromBlock", "latest"),
            toBlock=data.get("toBlock", "latest"),
            address=data.get("address"),
            topics=data.get("topics"),
        )


# --- Positronic-specific Types ---

@dataclass
class PositronicAIScore:
    """AI validation score for a transaction."""
    tx_hash: str
    score: float
    status: str
    model_version: int
    components: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositronicNodeInfo:
    """Node information response."""
    node_id: str
    address: str
    chain_height: int
    peers: int
    mempool: int
    validator: bool
    ai_enabled: bool
    ai_model_version: int = 1
    nvn_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositronicValidatorInfo:
    """Validator information."""
    address: str
    stake: str
    delegated: str
    is_active: bool
    blocks_produced: int = 0
    ai_accuracy: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)
