"""
DesktopApi — high-level data layer for the Positronic desktop app.

Wraps RPCClient with wallet, staking, and mining-history functionality.
All methods return plain dicts/lists safe for UI consumption.

Security:
  - Input validation on all address/amount parameters
  - Thread-safe shared state via threading.Lock
  - Error messages sanitized (no internal paths leaked)
"""

import collections
import logging
import os
import threading
import time
from typing import Optional

from positronic.app.rpc_client import RPCClient

logger = logging.getLogger("positronic.app.api")

# Base unit for ASF (10^18)
BASE_UNIT = 10 ** 18

# Password minimum requirements
_MIN_PASSWORD_LENGTH = 12
_MIN_PASSWORD_ENTROPY_CLASSES = 3  # Must have at least 3 of: upper, lower, digit, special

# Common weak passwords to reject
_COMMON_PASSWORDS = frozenset([
    "password1234", "123456789012", "qwerty123456", "admin1234567",
    "letmein12345", "welcome12345", "password1!", "changeme1234",
])


def _password_strength(password: str) -> tuple[bool, str]:
    """Check password strength. Returns (ok, reason)."""
    if len(password) < _MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {_MIN_PASSWORD_LENGTH} characters"
    if password.lower() in _COMMON_PASSWORDS:
        return False, "This password is too common, please choose a stronger one"
    classes = 0
    if any(c.isupper() for c in password):
        classes += 1
    if any(c.islower() for c in password):
        classes += 1
    if any(c.isdigit() for c in password):
        classes += 1
    if any(not c.isalnum() for c in password):
        classes += 1
    if classes < _MIN_PASSWORD_ENTROPY_CLASSES:
        return False, "Password must include at least 3 of: uppercase, lowercase, digits, special characters"
    return True, ""


def _sanitize_error(exc: Exception) -> str:
    """Sanitize exception message — strip internal filesystem paths."""
    msg = str(exc)
    # Remove filesystem paths (Windows: C:\..., C:..., and Unix: /home/...)
    import re
    msg = re.sub(r'[A-Za-z]:[\\\/][^\s\'\"]*', '[path]', msg)
    msg = re.sub(r'/(?:home|tmp|var|usr|etc|root)[^\s\'\"]*', '[path]', msg)
    # Remove any remaining paths with common OS directories
    msg = re.sub(r'(?:Users|AppData|Documents|Desktop)[\\\/][^\s\'\"]*', '[path]', msg)
    # Truncate long messages
    if len(msg) > 200:
        msg = msg[:200] + "..."
    return msg


class DesktopApi:
    """Data provider for the desktop dashboard."""

    def __init__(self, rpc_client: RPCClient, data_dir: str):
        self._rpc = rpc_client
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._mining_history = collections.deque(maxlen=60)
        self._last_block = 0
        self._start_time = time.time()
        # Persistent thread pool for ecosystem RPC calls (prevents OOM)
        from concurrent.futures import ThreadPoolExecutor
        self._ecosystem_pool = ThreadPoolExecutor(max_workers=4)
        self._activity_log = collections.deque(maxlen=50)
        self._log_buffer = collections.deque(maxlen=500)

    # ── Dashboard data ────────────────────────────────────────────

    def get_dashboard(self) -> dict:
        """Get all data for the dashboard tab in one call."""
        info = self._rpc.get_node_info()
        if info is None:
            return {"online": False, "uptime": self._format_uptime()}

        height = info.get("height", 0)
        consensus = info.get("consensus", {})
        network = info.get("network", {})
        ai = info.get("ai", {})
        state = info.get("state", {})

        # Track mining/sync activity (blocks per poll) — thread-safe
        with self._lock:
            if self._last_block == 0:
                # First poll: record 0 as baseline, set _last_block
                new_blocks = 0
            else:
                new_blocks = max(0, height - self._last_block)
                new_blocks = min(new_blocks, 5)  # Cap to prevent sync spikes
            self._last_block = height
            self._mining_history.append(new_blocks)

        if new_blocks > 0:
            self._add_activity("block", f"Block #{height:,} synced")

        peers = network.get("peer_count", info.get("peers", 0))
        is_val = consensus.get("is_validator", False)
        staked = consensus.get("total_staked", 0)
        staked_asf = staked / BASE_UNIT if staked > 0 else 0

        # AI stats
        avg_score = ai.get("avg_score")
        accuracy = (1.0 - avg_score) * 100 if avg_score is not None else 0

        # True network status: "online" only when connected to at least 1 peer.
        # Node running but with 0 peers = "connecting" (not yet on the network).
        has_peers = peers > 0

        return {
            "online": has_peers,
            "connecting": not has_peers,
            "block_height": height,
            "peers": peers,
            "max_peers": network.get("max_peers", 12),
            "is_validator": is_val,
            "staked": f"{staked_asf:,.3f}" if staked_asf > 0 else "0",
            "chain_id": info.get("chain_id", 420420),
            "network_type": network.get("network_type", info.get("network_type", "mainnet")),
            "synced": info.get("synced", False),
            "mempool_size": info.get("mempool_size", network.get("mempool_size", 0)),
            "uptime": self._format_uptime(),
            # AI
            "ai_enabled": ai.get("ai_enabled", False),
            "ai_model": f"v{ai.get('model_version', 1)}",
            "ai_samples": ai.get("training_samples", 0),
            "ai_accuracy": f"{accuracy:.1f}%" if avg_score is not None else "--",
            "ai_threats": ai.get("quarantine_pending", 0),
            # Mining history for chart
            "mining_history": list(self._mining_history),
            # Accounts
            "accounts": state.get("account_count", 0),
            "total_txs": state.get("tx_count", 0),
            # Head hash
            "head_hash": info.get("head_hash", "")[:20],
            # Activity log
            "activity": list(self._activity_log),
        }

    # ── Validator data ────────────────────────────────────────────

    def get_validator_info(self) -> dict:
        """Get validator-specific information."""
        info = self._rpc.get_node_info()
        if info is None:
            return {"online": False}

        consensus = info.get("consensus", {})
        return {
            "online": True,
            "network_has_validators": consensus.get("active_validators", 0) > 0,
            "validator_count": consensus.get("total_validators", consensus.get("validator_count", 0)),
            "active_validators": consensus.get("active_validators", 0),
            "epoch": consensus.get("epoch", consensus.get("current_epoch", 0)),
            "proposer": consensus.get("proposer", "unknown"),
            "total_staked": consensus.get("total_staked", 0),
            "block_height": info.get("height", 0),
        }

    # ── Wallet operations ─────────────────────────────────────────

    def scan_wallet(self, address: str) -> dict:
        """Get balance and info for an address."""
        if not address:
            return {"error": "No address provided"}
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format (expected 0x + 40 hex chars)"}

        balance = self._rpc.call("eth_getBalance", [address, "latest"])
        nonce = self._rpc.call("eth_getTransactionCount", [address, "latest"])

        if balance is None:
            return {"error": "Could not fetch balance (node offline?)"}

        # Parse hex balance — safe parsing
        balance_int = RPCClient.sanitize_hex(balance) or 0
        balance_asf = balance_int / BASE_UNIT
        nonce_int = RPCClient.sanitize_hex(nonce) or 0

        return {
            "address": address,
            "balance": balance_asf,
            "balance_raw": balance_int,
            "balance_display": f"{balance_asf:,.6f} ASF",
            "nonce": nonce_int,
            "tx_count": nonce_int,
        }

    def create_wallet(self, password: str) -> dict:
        """Create a new wallet and return the address + mnemonic."""
        # Validate password strength
        ok, reason = _password_strength(password)
        if not ok:
            return {"error": reason}
        try:
            from positronic.wallet.hd_wallet import HDWallet
            wallet = HDWallet.create(word_count=24)
            kp = wallet.derive_account(account=0)

            # Save to keystore
            keystore_dir = os.path.join(self._data_dir, "keystore")
            os.makedirs(keystore_dir, exist_ok=True)

            from positronic.wallet.keystore import KeyStore
            filepath = os.path.join(keystore_dir, f"{kp.address_hex}.json")
            KeyStore.save_key(kp, password, filepath)

            self._add_activity("wallet", f"Wallet created: {kp.address_hex[:12]}...")
            return {
                "address": kp.address_hex,
                "mnemonic": wallet._mnemonic if hasattr(wallet, '_mnemonic') else "(saved in keystore)",
                "keystore": filepath,
            }
        except Exception as e:
            logger.error("Failed to create wallet: %s", e)
            return {"error": _sanitize_error(e)}

    def stake(self, amount_asf: float, address: str, password: str) -> dict:
        """Build and submit a STAKE transaction."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            amount = int(amount_asf * BASE_UNIT)
            if amount < 32 * BASE_UNIT:
                return {"error": "Minimum stake is 32 ASF"}

            # Load keypair from keystore
            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{address}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for this address"}

            from positronic.wallet.keystore import KeyStore
            from positronic.wallet.wallet import Wallet
            kp = KeyStore.load_key(filepath, password)

            wallet = Wallet(keystore_dir=keystore_dir)
            wallet.active_account = kp

            # Get nonce
            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = wallet.build_stake(amount, nonce, kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("stake", f"Staked {amount_asf} ASF")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected by node"}
        except Exception as e:
            logger.error("Stake failed: %s", e)
            return {"error": _sanitize_error(e)}

    def unstake(self, address: str, password: str) -> dict:
        """Build and submit an UNSTAKE transaction."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{address}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for this address"}

            from positronic.wallet.keystore import KeyStore
            from positronic.core.transaction import Transaction, TxType
            from positronic.constants import CHAIN_ID, TX_BASE_GAS
            kp = KeyStore.load_key(filepath, password)

            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = Transaction(
                tx_type=TxType.UNSTAKE,
                nonce=nonce,
                sender=kp.public_key_bytes,
                recipient=kp.address,
                value=0,
                gas_price=1,
                gas_limit=TX_BASE_GAS,
                chain_id=CHAIN_ID,
            )
            tx.sign(kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("unstake", "Unstake requested")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected"}
        except Exception as e:
            logger.error("Unstake failed: %s", e)
            return {"error": _sanitize_error(e)}

    # ── Wallet list & transfer ────────────────────────────────────

    def list_wallets(self) -> list[dict]:
        """List all keystore files in the data directory.
        Returns list of {address, filename} dicts."""
        keystore_dir = os.path.join(self._data_dir, "keystore")
        if not os.path.isdir(keystore_dir):
            return []
        wallets = []
        for fname in sorted(os.listdir(keystore_dir)):
            if fname.endswith(".json"):
                addr = fname.replace(".json", "")
                wallets.append({"address": addr, "filename": fname})
        return wallets

    def send_transfer(self, from_addr: str, to_addr: str,
                      amount_asf: float, password: str) -> dict:
        """Build and submit a TRANSFER transaction."""
        if not RPCClient.validate_address(from_addr):
            return {"error": "Invalid sender address"}
        if not RPCClient.validate_address(to_addr):
            return {"error": "Invalid recipient address"}
        if amount_asf <= 0:
            return {"error": "Amount must be positive"}
        try:
            amount = int(amount_asf * BASE_UNIT)

            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{from_addr}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for sender address"}

            from positronic.wallet.keystore import KeyStore
            from positronic.core.transaction import Transaction, TxType
            from positronic.constants import CHAIN_ID, TX_BASE_GAS
            kp = KeyStore.load_key(filepath, password)

            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = Transaction(
                tx_type=TxType.TRANSFER,
                nonce=nonce,
                sender=kp.public_key_bytes,
                recipient=bytes.fromhex(to_addr[2:]) if to_addr.startswith("0x") else bytes.fromhex(to_addr),
                value=amount,
                gas_price=1,
                gas_limit=TX_BASE_GAS,
                chain_id=CHAIN_ID,
            )
            tx.sign(kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("tx", f"Sent {amount_asf} ASF to {to_addr[:12]}...")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected by node"}
        except Exception as e:
            logger.error("Transfer failed: %s", e)
            return {"error": _sanitize_error(e)}

    # ── Secret-key based methods (new wallet model) ──────────────

    def _kp_from_secret(self, secret_key_hex: str):
        """Create KeyPair from a 64-char hex secret key."""
        from positronic.crypto.keys import KeyPair
        seed = bytes.fromhex(secret_key_hex)
        return KeyPair.from_seed(seed)

    def send_transfer_with_key(self, from_addr: str, to_addr: str,
                                amount_asf: float, secret_key: str) -> dict:
        """Build and submit a TRANSFER using in-memory secret key."""
        if not RPCClient.validate_address(from_addr):
            return {"error": "Invalid sender address"}
        if not RPCClient.validate_address(to_addr):
            return {"error": "Invalid recipient address"}
        if amount_asf <= 0:
            return {"error": "Amount must be positive"}
        try:
            amount = int(amount_asf * BASE_UNIT)
            kp = self._kp_from_secret(secret_key)

            from positronic.core.transaction import Transaction, TxType
            from positronic.constants import CHAIN_ID, TX_BASE_GAS

            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = Transaction(
                tx_type=TxType.TRANSFER,
                nonce=nonce,
                sender=kp.public_key_bytes,
                recipient=bytes.fromhex(to_addr[2:]) if to_addr.startswith("0x") else bytes.fromhex(to_addr),
                value=amount,
                gas_price=1,
                gas_limit=TX_BASE_GAS,
                chain_id=CHAIN_ID,
            )
            tx.sign(kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("tx", f"Sent {amount_asf} ASF to {to_addr[:12]}...")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected by node"}
        except Exception as e:
            logger.error("Transfer (key) failed: %s", e)
            return {"error": _sanitize_error(e)}

    def stake_with_key(self, amount_asf: float, address: str,
                        secret_key: str) -> dict:
        """Stake ASF with pubkey for on-chain validator registration."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            if amount_asf < 32:
                return {"error": "Minimum stake is 32 ASF"}
            kp = self._kp_from_secret(secret_key)
            pubkey_hex = kp.public_key_bytes.hex()
            # Call positronic_stake with pubkey so all nodes can register this validator
            result = self._rpc.call("positronic_stake",
                                     [address, str(amount_asf), pubkey_hex])
            if result and isinstance(result, dict):
                if result.get("success"):
                    self._add_activity("stake", f"Staked {amount_asf} ASF")
                return result
            return {"error": "Stake failed: no response"}
        except Exception as e:
            logger.error("Stake (key) failed: %s", e)
            return {"error": _sanitize_error(e)}

    def unstake_with_key(self, address: str, secret_key: str,
                          amount_asf: float = 0) -> dict:
        """Unstake ASF via positronic_unstake RPC (propagates via P2P).
        If amount_asf=0, unstakes full staked balance."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        # Auto-fetch staked amount if not specified
        if amount_asf <= 0:
            try:
                info = self.get_staking_info(address)
                from positronic.constants import BASE_UNIT
                amount_asf = info.get("staked", 0) / BASE_UNIT if info else 0
            except Exception:
                amount_asf = 0
        try:
            kp = self._kp_from_secret(secret_key)
            pubkey_hex = kp.public_key_bytes.hex()
            result = self._rpc.call("positronic_unstake",
                                     [address, str(amount_asf), pubkey_hex])
            if result and isinstance(result, dict):
                if result.get("success"):
                    self._add_activity("unstake", f"Unstaked {amount_asf} ASF")
                return result
            return {"error": "Unstake failed: no response"}
        except Exception as e:
            logger.error("Unstake (key) failed: %s", e)
            return {"error": _sanitize_error(e)}

    def claim_rewards_with_key(self, address: str, secret_key: str) -> dict:
        """Claim staking rewards via positronic_claimStakingRewards RPC (propagates via P2P)."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            kp = self._kp_from_secret(secret_key)
            pubkey_hex = kp.public_key_bytes.hex()
            result = self._rpc.call("positronic_claimStakingRewards",
                                     [address, pubkey_hex])
            if result and isinstance(result, dict):
                if result.get("success"):
                    self._add_activity("stake", "Rewards claimed")
                return result
            return {"error": "Claim rewards failed: no response"}
        except Exception as e:
            logger.error("Claim rewards (key) failed: %s", e)
            return {"error": _sanitize_error(e)}

    def export_private_key(self, address: str, password: str) -> dict:
        """Decrypt keystore and return the 64-char hex private key."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{address}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for this address"}

            from positronic.wallet.keystore import KeyStore
            kp = KeyStore.load_key(filepath, password)
            # Extract hex private key (64 chars)
            if hasattr(kp, 'private_key_hex'):
                pk_hex = kp.private_key_hex
            elif hasattr(kp, 'private_key'):
                pk_hex = kp.private_key.hex() if isinstance(kp.private_key, bytes) else str(kp.private_key)
            elif hasattr(kp, '_private_key'):
                pk_hex = kp._private_key.hex() if isinstance(kp._private_key, bytes) else str(kp._private_key)
            else:
                return {"error": "Cannot extract private key from keystore format"}

            # Strip leading 0x if present
            if pk_hex.startswith("0x"):
                pk_hex = pk_hex[2:]

            return {"success": True, "private_key": pk_hex}
        except Exception as e:
            logger.error("Export key failed: %s", e)
            return {"error": _sanitize_error(e)}

    def get_staking_info(self, address: str) -> dict:
        """Get staking info for a specific address via RPC."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address"}
        result = self._rpc.call("positronic_getStakingInfo", [address])
        if result is None:
            return {"staked": 0, "rewards": 0, "is_active": False}
        return result

    def claim_rewards(self, address: str, password: str) -> dict:
        """Build and submit a CLAIM_REWARDS transaction."""
        if not RPCClient.validate_address(address):
            return {"error": "Invalid address format"}
        try:
            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{address}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for this address"}

            from positronic.wallet.keystore import KeyStore
            from positronic.core.transaction import Transaction, TxType
            from positronic.constants import CHAIN_ID, TX_BASE_GAS
            kp = KeyStore.load_key(filepath, password)

            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = Transaction(
                tx_type=TxType.CLAIM_REWARDS,
                nonce=nonce,
                sender=kp.public_key_bytes,
                recipient=kp.address,
                value=0,
                gas_price=1,
                gas_limit=TX_BASE_GAS,
                chain_id=CHAIN_ID,
            )
            tx.sign(kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("stake", "Rewards claimed")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected"}
        except Exception as e:
            logger.error("Claim rewards failed: %s", e)
            return {"error": _sanitize_error(e)}

    def get_token_balances(self, address: str) -> list[dict]:
        """Get PRC-20 token balances for an address."""
        if not RPCClient.validate_address(address):
            return []
        result = self._rpc.call("positronic_getTokensByCreator", [address])
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def create_token(self, name: str, symbol: str, supply: int,
                     creator_addr: str, password: str) -> dict:
        """Create a new PRC-20 token via RPC."""
        if not RPCClient.validate_address(creator_addr):
            return {"error": "Invalid creator address"}
        if not name or not symbol:
            return {"error": "Name and symbol are required"}
        if supply <= 0:
            return {"error": "Supply must be positive"}
        try:
            keystore_dir = os.path.join(self._data_dir, "keystore")
            filepath = os.path.join(keystore_dir, f"{creator_addr}.json")
            if not os.path.exists(filepath):
                return {"error": "Keystore not found for creator address"}

            from positronic.wallet.keystore import KeyStore
            kp = KeyStore.load_key(filepath, password)

            result = self._rpc.call("positronic_createToken", [{
                "name": name,
                "symbol": symbol,
                "total_supply": supply,
                "creator": creator_addr,
            }])
            if result and result.get("token_address"):
                self._add_activity("tx", f"Token {symbol} created")
                return {"success": True, "token_address": result["token_address"]}
            return {"error": result.get("error", "Token creation failed")}
        except Exception as e:
            logger.error("Create token failed: %s", e)
            return {"error": _sanitize_error(e)}

    def create_token_with_key(self, name: str, symbol: str, supply: int,
                              creator_addr: str, secret_key: str) -> dict:
        """Create a new PRC-20 token using in-memory secret key (signed TX)."""
        if not RPCClient.validate_address(creator_addr):
            return {"error": "Invalid creator address"}
        if not name or not symbol:
            return {"error": "Name and symbol are required"}
        if supply <= 0:
            return {"error": "Supply must be positive"}
        try:
            kp = self._kp_from_secret(secret_key)

            from positronic.core.transaction import Transaction, TxType
            from positronic.constants import CHAIN_ID, TX_BASE_GAS

            nonce_hex = self._rpc.call("eth_getTransactionCount",
                                       [kp.address_hex, "latest"])
            nonce = RPCClient.sanitize_hex(nonce_hex) or 0

            tx = Transaction(
                tx_type=TxType.TOKEN_CREATE,
                nonce=nonce,
                sender=kp.public_key_bytes,
                recipient=kp.address,
                value=0,
                gas_price=1,
                gas_limit=TX_BASE_GAS,
                chain_id=CHAIN_ID,
                data={
                    "name": name,
                    "symbol": symbol,
                    "total_supply": supply,
                    "creator": creator_addr,
                },
            )
            tx.sign(kp)
            tx_hex = "0x" + tx.to_bytes().hex()
            result = self._rpc.call("eth_sendRawTransaction", [tx_hex])

            if result:
                self._add_activity("tx", f"Token {symbol} created")
                return {"success": True, "tx_hash": result}
            return {"error": "Transaction rejected by node"}
        except Exception as e:
            logger.error("Create token (key) failed: %s", e)
            return {"error": _sanitize_error(e)}

    def get_address_transactions(self, address: str, limit: int = 20) -> list:
        """Fetch transaction history for address from chain."""
        try:
            result = self._rpc.call("positronic_getAddressTransactions", [address, limit])
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    # ── Emergency status ─────────────────────────────────────────

    def get_emergency_status(self) -> dict:
        """Get the network emergency state."""
        result = self._rpc.call("positronic_emergencyStatus")
        if result is None:
            return {"state": 0, "state_name": "NORMAL"}
        return result

    # ── Network data ──────────────────────────────────────────────

    def get_network(self) -> dict:
        """Get network/peer information."""
        health = self._rpc.get_network_health()
        info = self._rpc.get_node_info()
        if info is None:
            return {"online": False}

        network = info.get("network", {})

        # Fetch actual peer details from positronic_getPeers
        peer_list = self._rpc.call("positronic_getPeers") or []

        # Fetch mempool detail (total_gas, oldest_age)
        mempool_status = self._rpc.call("positronic_getMempoolStatus") or {}

        return {
            "online": True,
            "peers": network.get("peer_count", info.get("peers", 0)),
            "max_peers": network.get("max_peers", 12),
            "p2p_port": network.get("p2p_port", info.get("p2p_port", 9000)),
            "rpc_port": network.get("rpc_port", info.get("rpc_port", 8545)),
            "network_type": network.get("network_type", "mainnet"),
            "synced": info.get("synced", False),
            "mempool_size": mempool_status.get("size", info.get("mempool_size", 0)),
            "mempool_info": {
                "total_gas": mempool_status.get("total_gas", 0),
                "oldest_age": mempool_status.get("oldest_age", "--"),
            },
            "health": health or {},
            "peer_list": peer_list,
        }

    # ── Logging ───────────────────────────────────────────────────

    def add_log(self, msg: str, level: str = "INFO"):
        """Add a log entry to the buffer."""
        self._log_buffer.append({
            "time": time.strftime("%H:%M:%S"),
            "level": level,
            "msg": msg,
        })

    def get_logs(self, count: int = 100) -> list:
        """Get recent log entries."""
        return list(self._log_buffer)[-count:]

    # ── Ecosystem data (all network features) ────────────────────

    def get_ecosystem(self) -> dict:
        """Get comprehensive ecosystem status — all network features.
        Uses persistent thread pool for performance (18 endpoints)."""
        from concurrent.futures import as_completed

        calls = {
            "neural": self._rpc.get_neural_status,
            "governance": self._rpc.get_governance_stats,
            "did": self._rpc.get_did_stats,
            "bridge": self._rpc.get_bridge_stats,
            "depin": self._rpc.get_depin_stats,
            "agents": self._rpc.get_agent_stats,
            "marketplace": self._rpc.get_marketplace_stats,
            "rwa": self._rpc.get_rwa_stats,
            "zkml": self._rpc.get_zkml_stats,
            "trust": self._rpc.get_trust_stats,
            "tokens": self._rpc.get_token_registry_stats,
            "consensus": self._rpc.get_consensus_info,
            "pq": self._rpc.get_pq_stats,
            "checkpoint": self._rpc.get_checkpoint_stats,
            "cold_start": self._rpc.get_cold_start_status,
            "pathway": self._rpc.get_pathway_health,
            "drift": self._rpc.get_drift_alerts,
            "immune": self._rpc.get_immune_status,
        }
        results = {}
        pool = self._ecosystem_pool
        futures = {pool.submit(fn): name for name, fn in calls.items()}
        for future in as_completed(futures, timeout=8):
            name = futures[future]
            try:
                results[name] = future.result(timeout=5) or {}
            except Exception:
                results[name] = {}

        neural = results.get("neural", {})
        governance = results.get("governance", {})
        did = results.get("did", {})
        bridge = results.get("bridge", {})
        depin = results.get("depin", {})
        agents = results.get("agents", {})
        marketplace = results.get("marketplace", {})
        rwa = results.get("rwa", {})
        zkml = results.get("zkml", {})
        trust = results.get("trust", {})
        tokens = results.get("tokens", {})
        consensus = results.get("consensus", {})
        pq = results.get("pq", {})
        checkpoint = results.get("checkpoint", {})
        cold_start = results.get("cold_start", {})
        pathway = results.get("pathway", {})
        drift = results.get("drift", {})
        immune = results.get("immune", {})

        return {
            # Neural AI (Phase 32)
            "neural_status": neural.get("status", "unknown"),
            "neural_degradation_level": neural.get("degradation_level", 0),
            "neural_pathways": pathway.get("active_pathways", 0),
            "neural_drift_alerts": len(drift) if isinstance(drift, list) else drift.get("count", 0),
            "cold_start_phase": cold_start.get("phase", "unknown"),
            # Governance
            "gov_proposals": governance.get("total_proposals", 0),
            "gov_pending": governance.get("pending_proposals", 0),
            "gov_participation": governance.get("participation_rate", 0),
            # DID
            "did_total": did.get("total_identities", 0),
            "did_active": did.get("active_identities", 0),
            "did_credentials": did.get("total_credentials", 0),
            # Bridge
            "bridge_locked": bridge.get("total_locked", 0),
            "bridge_transfers": bridge.get("total_transfers", 0),
            # DePIN
            "depin_devices": depin.get("total_devices", 0),
            "depin_active": depin.get("active_devices", 0),
            # Agents
            "agents_total": agents.get("total_agents", 0),
            "agents_active": agents.get("active_agents", 0),
            # Marketplace
            "mkt_agents": marketplace.get("total_agents", 0),
            "mkt_tasks": marketplace.get("total_tasks", 0),
            # RWA
            "rwa_assets": rwa.get("total_assets", 0),
            "rwa_value": rwa.get("total_value", 0),
            # ZKML
            "zkml_proofs": zkml.get("total_proofs", 0),
            "zkml_verified": zkml.get("verified_proofs", 0),
            # Trust
            "trust_profiles": trust.get("total_profiles", 0),
            # Tokens
            "prc20_tokens": tokens.get("total_tokens", tokens.get("prc20_count", 0)),
            "nft_collections": tokens.get("total_collections", tokens.get("nft_collections", 0)),
            # Consensus
            "consensus_validators": consensus.get("active_validators", consensus.get("validator_count", 0)),
            "consensus_epoch": consensus.get("current_epoch", 0),
            "consensus_finalized": consensus.get("finalized_height", 0),
            # Post-Quantum
            "pq_enabled": pq.get("enabled", False),
            "pq_keys": pq.get("total_keys", 0),
            # Checkpoints
            "checkpoint_latest": checkpoint.get("latest_checkpoint_height", checkpoint.get("latest_height", 0)),
            "checkpoint_count": checkpoint.get("total_checkpoints", 0),
            # Immune
            "immune_threats": immune.get("total_threats", 0),
            "immune_blocked": immune.get("blocked_addresses", immune.get("blocked_count", 0)),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _add_activity(self, kind: str, text: str):
        with self._lock:
            self._activity_log.appendleft({
                "kind": kind,
                "text": text,
                "time": time.strftime("%H:%M:%S"),
                "ago": "now",
            })

    def _format_uptime(self) -> str:
        elapsed = int(time.time() - self._start_time)
        hours = elapsed // 3600
        mins = (elapsed % 3600) // 60
        return f"{hours}h {mins}m"
