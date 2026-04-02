"""
Positronic - Configuration Management
Loads node configuration from YAML file or defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

from positronic.constants import (
    DEFAULT_P2P_PORT,
    DEFAULT_RPC_PORT,
    MAX_PEERS,
    CHAIN_ID,
    AI_ACCEPT_THRESHOLD,
    AI_QUARANTINE_THRESHOLD,
)


@dataclass
class TLSConfig:
    """TLS/SSL configuration for secure P2P connections."""
    enabled: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None        # Certificate authority for peer verification
    verify_peers: bool = False            # Verify peer certificates (False = accept self-signed)
    generate_self_signed: bool = True     # Auto-generate self-signed cert if none provided


@dataclass
class NetworkConfig:
    p2p_port: int = DEFAULT_P2P_PORT
    rpc_port: int = DEFAULT_RPC_PORT
    p2p_host: str = "0.0.0.0"
    rpc_host: str = "0.0.0.0"
    max_peers: int = MAX_PEERS
    bootstrap_nodes: List[str] = field(default_factory=list)
    network_type: str = "local"  # "local", "testnet", "mainnet"
    tls: TLSConfig = field(default_factory=TLSConfig)


@dataclass
class ValidatorConfig:
    enabled: bool = False
    private_key_path: Optional[str] = None
    keystore_path: Optional[str] = None  # Encrypted keystore file
    nvn_enabled: bool = False  # Neural Validator Node


@dataclass
class StorageConfig:
    data_dir: str = "./data"
    db_name: str = "positronic.db"

    @property
    def db_path(self) -> str:
        return os.path.join(self.data_dir, self.db_name)

    @property
    def keystore_dir(self) -> str:
        return os.path.join(self.data_dir, "keystore")

    @property
    def ai_models_dir(self) -> str:
        return os.path.join(self.data_dir, "ai_models")


@dataclass
class AIConfig:
    enabled: bool = True
    model_path: Optional[str] = None
    model_dir: Optional[str] = None
    auto_train: bool = True
    accept_threshold: Optional[float] = None
    quarantine_threshold: Optional[float] = None
    kill_switch_threshold: float = 0.05


@dataclass
class NodeConfig:
    chain_id: int = CHAIN_ID
    network: NetworkConfig = field(default_factory=NetworkConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "NodeConfig":
        """Create config from dictionary (parsed YAML)."""
        cfg = cls()
        if "chain_id" in data:
            cfg.chain_id = data["chain_id"]
        if "log_level" in data:
            cfg.log_level = data["log_level"]

        if "network" in data:
            net = data["network"]
            tls_data = net.get("tls", {})
            tls_config = TLSConfig(
                enabled=tls_data.get("enabled", True),
                cert_path=tls_data.get("cert_path"),
                key_path=tls_data.get("key_path"),
                ca_path=tls_data.get("ca_path"),
                verify_peers=tls_data.get("verify_peers", False),
                generate_self_signed=tls_data.get("generate_self_signed", True),
            )
            cfg.network = NetworkConfig(
                p2p_port=net.get("p2p_port", DEFAULT_P2P_PORT),
                rpc_port=net.get("rpc_port", DEFAULT_RPC_PORT),
                p2p_host=net.get("p2p_host", "0.0.0.0"),
                rpc_host=net.get("rpc_host", "0.0.0.0"),
                max_peers=net.get("max_peers", MAX_PEERS),
                bootstrap_nodes=net.get("bootstrap_nodes", []),
                network_type=net.get("network_type", "local"),
                tls=tls_config,
            )

        if "validator" in data:
            val = data["validator"]
            cfg.validator = ValidatorConfig(
                enabled=val.get("enabled", False),
                private_key_path=val.get("private_key_path"),
                keystore_path=val.get("keystore_path"),
                nvn_enabled=val.get("nvn_enabled", False),
            )

        if "storage" in data:
            stor = data["storage"]
            cfg.storage = StorageConfig(
                data_dir=stor.get("data_dir", "./data"),
                db_name=stor.get("db_name", "positronic.db"),
            )

        if "ai" in data:
            ai = data["ai"]
            cfg.ai = AIConfig(
                enabled=ai.get("enabled", True),
                model_path=ai.get("model_path"),
                model_dir=ai.get("model_dir"),
                auto_train=ai.get("auto_train", True),
                accept_threshold=ai.get("accept_threshold"),
                quarantine_threshold=ai.get("quarantine_threshold"),
                kill_switch_threshold=ai.get("kill_switch_threshold", 0.05),
            )

        return cfg

    @classmethod
    def load(cls, path: str) -> "NodeConfig":
        """Load config from YAML file."""
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except (ImportError, FileNotFoundError):
            return cls()

    def ensure_dirs(self):
        """Create necessary data directories."""
        os.makedirs(self.storage.data_dir, exist_ok=True)
        os.makedirs(self.storage.keystore_dir, exist_ok=True)
        os.makedirs(self.storage.ai_models_dir, exist_ok=True)
