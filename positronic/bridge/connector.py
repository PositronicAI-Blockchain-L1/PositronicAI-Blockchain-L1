"""
Positronic Bridge — External Chain Connector

Provides web3.py-based connectivity to Ethereum, Polygon, BSC for:
- Minting wrapped ASF tokens when a lock is confirmed
- Verifying burn events before releasing locked tokens
- Anchoring Positronic block hashes on external chains

Gracefully degrades if web3 is not installed — bridge still works
for testnet with manual relayer confirmations.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

logger = logging.getLogger("positronic.bridge.connector")

# ──────────────────────────────────────────────────────────
#  Graceful web3 import
# ──────────────────────────────────────────────────────────
try:
    from web3 import Web3
    from web3.exceptions import ContractLogicError
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.info("web3 not installed — external chain connector in relay-only mode")


# ──────────────────────────────────────────────────────────
#  Minimal ERC-20 ABI for wrapped ASF token
# ──────────────────────────────────────────────────────────
WRAPPED_ASF_ABI = [
    {
        "inputs": [
            {"name": "to",     "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "lockId", "type": "string"}
        ],
        "name": "mint",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "amount", "type": "uint256"},
            {"name": "lockId", "type": "bytes32"}
        ],
        "name": "burnForRelease",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "burner",  "type": "address"},
            {"indexed": False, "name": "amount",  "type": "uint256"},
            {"indexed": True,  "name": "lockId",  "type": "bytes32"}
        ],
        "name": "BurnForRelease",
        "type": "event"
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


# ──────────────────────────────────────────────────────────
#  Chain configuration
# ──────────────────────────────────────────────────────────
@dataclass
class ChainConfig:
    """Configuration for an external chain."""
    name:             str
    chain_id:         int
    rpc_url:          str          # HTTP/HTTPS RPC endpoint
    contract_address: str = ""     # Deployed wrapped-ASF contract address
    relayer_privkey:  str = ""     # Set via BRIDGE_RELAYER_PRIVKEY env var on relayer nodes ONLY
    confirmations:    int = 12     # Blocks to wait for finality
    gas_limit:        int = 200_000
    enabled:          bool = False  # Disabled until contract is deployed


# Default chain configs (use public testnets until mainnet contracts deployed)
DEFAULT_CHAINS: Dict[int, ChainConfig] = {
    0: ChainConfig(       # TargetChain.ETHEREUM = 0
        name="Ethereum",
        chain_id=11155111,  # Sepolia testnet
        rpc_url="https://rpc.sepolia.org",
        confirmations=12,
    ),
    2: ChainConfig(       # TargetChain.POLYGON = 2
        name="Polygon",
        chain_id=80001,     # Mumbai testnet
        rpc_url="https://rpc-mumbai.maticvigil.com",
        confirmations=5,
    ),
    3: ChainConfig(       # TargetChain.BSC = 3
        name="BSC",
        chain_id=97,        # BSC testnet
        rpc_url="https://data-seed-prebsc-1-s1.binance.org:8545",
        confirmations=5,
    ),
}


# ──────────────────────────────────────────────────────────
#  Connector
# ──────────────────────────────────────────────────────────
class ExternalChainConnector:
    """
    Connects Positronic bridge to external chains via web3.py.

    Usage:
        connector = ExternalChainConnector()
        connector.configure_chain(0, rpc_url="https://...", contract_address="0x...", relayer_privkey="abc...")
        ok = await connector.mint_wrapped(target_chain=0, recipient="0x...", amount=32*10**18, lock_id="lock_abc")
        verified = await connector.verify_burn(target_chain=0, lock_id="lock_abc")
    """

    def __init__(self, chain_configs: Dict[int, ChainConfig] = None,
                 config_path: str = None):
        self._chains: Dict[int, ChainConfig] = dict(DEFAULT_CHAINS)
        if chain_configs:
            self._chains.update(chain_configs)
        self._w3: Dict[int, Any] = {}
        self._stats = {"mints": 0, "burns_verified": 0, "errors": 0}
        # Auto-load bridge_config.json
        cfg_file = config_path or self._default_config_path()
        if cfg_file and os.path.exists(cfg_file):
            self._load_config_file(cfg_file)

    @staticmethod
    def _default_config_path() -> str:
        """Locate bridge_config.json relative to this file."""
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "bridge_config.json")

    def _load_config_file(self, path: str):
        """Load chain configuration from bridge_config.json."""
        try:
            with open(path) as f:
                data = json.load(f)
            chains = data.get("chains", {})
            loaded = 0
            for key, chain_data in chains.items():
                try:
                    cid = int(key)
                    cfg = self._chains.get(cid)
                    if cfg is None:
                        continue
                    if chain_data.get("rpc_url"):
                        cfg.rpc_url = chain_data["rpc_url"]
                    if chain_data.get("contract_address"):
                        cfg.contract_address = chain_data["contract_address"]
                    # Relayer privkey comes ONLY from environment (never config file)
                    import os as _os
                    env_key = (
                        _os.environ.get(f"BRIDGE_RELAYER_PRIVKEY_{cid}")
                        or _os.environ.get("BRIDGE_RELAYER_PRIVKEY", "")
                    )
                    if env_key:
                        cfg.relayer_privkey = env_key

                    # Enable chain if contract is deployed (users can verify)
                    # Minting only works if relayer key is also present
                    has_contract = bool(chain_data.get("contract_address", "").strip())
                    cfg.enabled = bool(chain_data.get("enabled", False) and has_contract)
                    if cfg.enabled:
                        loaded += 1
                except Exception as e:
                    logger.warning(f"Bad chain config for key {key}: {e}")
            if loaded:
                logger.info(f"Bridge: {loaded} chain(s) enabled from {path}")
            else:
                logger.info(
                    f"Bridge: config loaded ({len(chains)} chain(s)) — "
                    f"set BRIDGE_RELAYER_PRIVKEY env var on relayer nodes to enable minting"
                )
        except Exception as e:
            logger.warning(f"Failed to load bridge_config.json: {e}")

    def configure_chain(self, chain_id: int, rpc_url: str = "",
                        contract_address: str = "", relayer_privkey: str = "",
                        enabled: bool = True):
        """Update configuration for a specific chain."""
        cfg = self._chains.get(chain_id)
        if cfg is None:
            logger.warning(f"Unknown chain_id {chain_id}")
            return
        if rpc_url:
            cfg.rpc_url = rpc_url
        if contract_address:
            cfg.contract_address = contract_address
        if relayer_privkey:
            cfg.relayer_privkey = relayer_privkey
        cfg.enabled = enabled
        # Clear cached web3 instance to force reconnect
        self._w3.pop(chain_id, None)

    def _get_w3(self, chain_id: int) -> Optional[Any]:
        """Get or create a web3 connection for a chain."""
        if not WEB3_AVAILABLE:
            return None
        if chain_id in self._w3:
            w3 = self._w3[chain_id]
            if w3.is_connected():
                return w3
        cfg = self._chains.get(chain_id)
        if not cfg or not cfg.rpc_url:
            return None
        try:
            w3 = Web3(Web3.HTTPProvider(cfg.rpc_url, request_kwargs={"timeout": 10}))
            if w3.is_connected():
                self._w3[chain_id] = w3
                logger.info(f"Connected to {cfg.name} (chain_id={cfg.chain_id})")
                return w3
            logger.warning(f"Cannot connect to {cfg.name} at {cfg.rpc_url}")
        except Exception as e:
            logger.warning(f"web3 connection error for {cfg.name}: {e}")
        return None

    def is_connected(self, chain_id: int) -> bool:
        """Check if connected to an external chain."""
        return self._get_w3(chain_id) is not None

    async def mint_wrapped(self, target_chain: int, recipient: str,
                           amount: int, lock_id: str) -> bool:
        """
        Mint wrapped ASF tokens on an external chain.
        Called after relayer quorum is reached.

        Returns True on success, False on failure or unavailable.
        """
        cfg = self._chains.get(target_chain)
        if not cfg or not cfg.enabled:
            logger.info(f"Chain {target_chain} not enabled — skipping external mint")
            return False

        if not cfg.contract_address or not cfg.relayer_privkey:
            logger.info(f"Chain {target_chain} missing contract/relayer config — relay-only mode")
            return False

        w3 = self._get_w3(target_chain)
        if w3 is None:
            logger.warning(f"Cannot reach {cfg.name} — mint skipped")
            self._stats["errors"] += 1
            return False

        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(cfg.contract_address),
                abi=WRAPPED_ASF_ABI,
            )
            relayer_account = w3.eth.account.from_key(cfg.relayer_privkey)
            recipient_cs    = Web3.to_checksum_address(recipient)

            # Build and sign transaction
            tx = contract.functions.mint(recipient_cs, amount, lock_id).build_transaction({
                "from":     relayer_account.address,
                "nonce":    w3.eth.get_transaction_count(relayer_account.address),
                "gas":      cfg.gas_limit,
                "gasPrice": w3.eth.gas_price,
                "chainId":  cfg.chain_id,
            })
            signed = relayer_account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)

            # Wait for confirmation
            receipt = w3.eth.wait_for_transaction_receipt(
                tx_hash, timeout=120, poll_latency=2
            )
            if receipt.status == 1:
                logger.info(
                    f"Minted {amount} wASF to {recipient} on {cfg.name} "
                    f"(lock={lock_id}, tx={tx_hash.hex()})"
                )
                self._stats["mints"] += 1
                return True
            else:
                logger.warning(f"Mint TX reverted on {cfg.name}: {tx_hash.hex()}")
                self._stats["errors"] += 1
                return False

        except Exception as e:
            logger.error(f"mint_wrapped failed on {cfg.name}: {e}")
            self._stats["errors"] += 1
            return False

    async def verify_burn(self, target_chain: int, lock_id: str,
                          from_block: int = 0) -> bool:
        """
        Verify that wrapped ASF tokens were burned on the external chain.
        Checks for BurnForRelease event matching lock_id.

        Returns True if burn confirmed, False otherwise.
        """
        cfg = self._chains.get(target_chain)
        if not cfg or not cfg.enabled:
            logger.info(f"Chain {target_chain} not enabled — burn auto-approved in relay mode")
            return True   # In relay-only mode, relayer signature is the proof

        if not cfg.contract_address:
            return True   # No contract deployed — trust relayer quorum

        w3 = self._get_w3(target_chain)
        if w3 is None:
            logger.warning(f"Cannot reach {cfg.name} — burn unverifiable, denying release")
            return False

        try:
            contract = w3.eth.contract(
                address=Web3.to_checksum_address(cfg.contract_address),
                abi=WRAPPED_ASF_ABI,
            )
            # Convert lock_id to bytes32
            lock_bytes = lock_id.encode().ljust(32, b"\x00")[:32]

            # Scan for BurnForRelease events matching this lock_id
            latest = w3.eth.block_number
            scan_from = max(0, from_block or latest - 10000)  # Scan last 10k blocks

            events = contract.events.BurnForRelease.get_logs(
                from_block=scan_from,
                to_block=latest,
                argument_filters={"lockId": lock_bytes},
            )

            if events:
                logger.info(
                    f"Burn confirmed on {cfg.name} for lock={lock_id} "
                    f"({len(events)} event(s) found)"
                )
                self._stats["burns_verified"] += 1
                return True
            else:
                logger.info(f"No BurnForRelease event found for lock={lock_id} on {cfg.name}")
                return False

        except Exception as e:
            logger.error(f"verify_burn failed on {cfg.name}: {e}")
            self._stats["errors"] += 1
            return False

    async def anchor_hash(self, chain_id: int, block_hash: bytes,
                          block_height: int) -> Optional[str]:
        """
        Anchor a Positronic block hash on an external chain via a simple
        ETH transfer with data (no smart contract needed).

        Returns the external chain TX hash on success, None on failure.
        """
        cfg = self._chains.get(chain_id)
        if not cfg or not cfg.enabled or not cfg.relayer_privkey:
            return None

        w3 = self._get_w3(chain_id)
        if w3 is None:
            return None

        try:
            relayer_account = w3.eth.account.from_key(cfg.relayer_privkey)
            data = b"POSITRONIC:" + block_height.to_bytes(8, "big") + block_hash[:32]
            tx = {
                "from":     relayer_account.address,
                "to":       relayer_account.address,  # Self-transfer (data-only)
                "value":    0,
                "data":     data,
                "nonce":    w3.eth.get_transaction_count(relayer_account.address),
                "gas":      50_000,
                "gasPrice": w3.eth.gas_price,
                "chainId":  cfg.chain_id,
            }
            signed   = relayer_account.sign_transaction(tx)
            tx_hash  = w3.eth.send_raw_transaction(signed.raw_transaction)
            logger.info(f"Anchored block #{block_height} on {cfg.name}: {tx_hash.hex()}")
            return tx_hash.hex()
        except Exception as e:
            logger.warning(f"anchor_hash failed on {cfg.name}: {e}")
            return None

    def get_stats(self) -> dict:
        return {
            "web3_available": WEB3_AVAILABLE,
            "chains_enabled": sum(1 for c in self._chains.values() if c.enabled),
            "chains_connected": sum(1 for cid in self._chains if self.is_connected(cid)),
            **self._stats,
        }
