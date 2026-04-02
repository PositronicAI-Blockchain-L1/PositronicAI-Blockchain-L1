"""
Positronic - RPC Access Control
Ethereum/Solana-style security for JSON-RPC methods.
Three access levels: PUBLIC, AUTHENTICATED, ADMIN.
"""

import os
import time
import secrets
import logging
from enum import IntEnum
from collections import defaultdict
from typing import Tuple, Optional, Dict, List

from positronic.constants import (
    GIFT_COOLDOWN,
    RPC_FAUCET_COOLDOWN_IP,
    RPC_FAUCET_MAX_PER_IP,
    RPC_GAME_MAX_PER_HOUR,
    RPC_ACCESS_DENIED_CODE,
)

logger = logging.getLogger(__name__)

ACCESS_DENIED_CODE = RPC_ACCESS_DENIED_CODE


class AccessLevel(IntEnum):
    """RPC method access levels (like Geth namespaces)."""
    PUBLIC = 0          # Anyone can call (read-only + sendRawTransaction)
    AUTHENTICATED = 1   # Needs rate limiting + basic validation
    ADMIN = 2           # Needs admin API key (like Bitcoin Core rpcauth)


# Method → AccessLevel mapping
# Methods NOT listed here default to PUBLIC (safe by default)
_METHOD_ACCESS: Dict[str, AccessLevel] = {
    # ADMIN: sensitive operations that can modify state or generate legal docs
    "positronic_generateCourtReport": AccessLevel.ADMIN,
    "positronic_getEvidencePackage": AccessLevel.ADMIN,

    # ADMIN: treasury management
    "positronic_adminTransfer": AccessLevel.ADMIN,
    "positronic_adminGetPeers": AccessLevel.ADMIN,
    "positronic_adminBanPeer": AccessLevel.ADMIN,
    "positronic_adminGetValidators": AccessLevel.ADMIN,

    # ADMIN: governance write operations
    "positronic_createGovernanceProposal": AccessLevel.ADMIN,
    "positronic_voteGovernanceProposal": AccessLevel.ADMIN,
    "positronic_executeGovernanceProposal": AccessLevel.ADMIN,

    # ADMIN: Telegram bot registration (only game developers)
    "positronic_telegramRegisterBot": AccessLevel.ADMIN,

    # ADMIN: Emergency Control System
    "positronic_emergencyPause": AccessLevel.ADMIN,
    "positronic_emergencyResume": AccessLevel.ADMIN,
    "positronic_emergencyHalt": AccessLevel.ADMIN,
    "positronic_upgradeSchedule": AccessLevel.ADMIN,

    # ADMIN: HD Wallet — mnemonic generation MUST NOT be public
    # Wallet creation should be done client-side (SDK).
    # These methods are admin-only for local node operator use.
    "positronic_hdCreateWallet": AccessLevel.ADMIN,
    "positronic_hdDeriveAddress": AccessLevel.ADMIN,

    # ADMIN: Game API key generation (produces secrets)
    "positronic_generateGameAPIKey": AccessLevel.ADMIN,

    # ADMIN: Game registration (only authorized developers)
    "positronic_registerGame": AccessLevel.ADMIN,

    # Bridge lock: user can lock their own funds (authenticated)
    "positronic_bridgeLock": AccessLevel.AUTHENTICATED,
    "positronic_bridgeMint": AccessLevel.ADMIN,
    "positronic_bridgeBurn": AccessLevel.ADMIN,
    "positronic_bridgeRelease": AccessLevel.ADMIN,

    # ADMIN: Agent autonomous execution (can spend funds)
    "positronic_agentExecuteAction": AccessLevel.ADMIN,
    "positronic_agentSetLimits": AccessLevel.ADMIN,

    # AUTHENTICATED: write operations with rate limiting
    "positronic_faucetDrip": AccessLevel.AUTHENTICATED,
    "positronic_submitGameResult": AccessLevel.AUTHENTICATED,
    "positronic_optInAutoPromotion": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: state-changing operations (require signed TX + rate limit)
    "positronic_createToken": AccessLevel.AUTHENTICATED,
    "positronic_tokenMint": AccessLevel.AUTHENTICATED,
    "positronic_createNFTCollection": AccessLevel.AUTHENTICATED,
    "positronic_mintNFT": AccessLevel.AUTHENTICATED,
    "positronic_registerPaymaster": AccessLevel.AUTHENTICATED,
    "positronic_createSmartWallet": AccessLevel.AUTHENTICATED,
    "positronic_registerAgent": AccessLevel.AUTHENTICATED,
    "positronic_createIdentity": AccessLevel.AUTHENTICATED,
    "positronic_registerDevice": AccessLevel.AUTHENTICATED,
    "positronic_claimDeviceRewards": AccessLevel.AUTHENTICATED,
    "positronic_startGameSession": AccessLevel.AUTHENTICATED,
    "positronic_addGameEvent": AccessLevel.AUTHENTICATED,
    "positronic_submitGameSession": AccessLevel.AUTHENTICATED,
    "positronic_startOnChainGame": AccessLevel.AUTHENTICATED,
    "positronic_bridgeConfirmLock": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: fund transfer & staking (CRITICAL — was missing!)
    "positronic_transfer": AccessLevel.AUTHENTICATED,
    "positronic_stake": AccessLevel.AUTHENTICATED,
    "positronic_unstake": AccessLevel.AUTHENTICATED,
    "positronic_claimStakingRewards": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: DEX operations (CRITICAL — was missing!)
    "positronic_dexCreatePool": AccessLevel.AUTHENTICATED,
    "positronic_dexSwap": AccessLevel.AUTHENTICATED,
    "positronic_dexAddLiquidity": AccessLevel.AUTHENTICATED,
    "positronic_dexRemoveLiquidity": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: RWA operations (CRITICAL — was missing!)
    "positronic_registerRWA": AccessLevel.AUTHENTICATED,
    "positronic_transferRWA": AccessLevel.AUTHENTICATED,
    "positronic_distributeDividend": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: Marketplace operations (CRITICAL — was missing!)
    "positronic_mktSubmitTask": AccessLevel.AUTHENTICATED,
    "positronic_mktExecuteTask": AccessLevel.AUTHENTICATED,
    "positronic_mktRegisterAgent": AccessLevel.AUTHENTICATED,
    "positronic_mktApproveAgent": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: Validator opt-in/out
    "positronic_optInValidator": AccessLevel.AUTHENTICATED,
    "positronic_optOutValidator": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: Session keys & recovery
    "positronic_createSessionKey": AccessLevel.AUTHENTICATED,
    "positronic_addRecoveryGuardian": AccessLevel.AUTHENTICATED,

    # ADMIN: Admin access request (CRITICAL — was PUBLIC!)
    "positronic_requestAdminAccess": AccessLevel.ADMIN,

    # ADMIN: Neural recovery & retrain
    "positronic_triggerNeuralRecovery": AccessLevel.ADMIN,
    "positronic_triggerManualRetrain": AccessLevel.ADMIN,

    # ADMIN: Game emergency pause
    "positronic_emergencyPauseGame": AccessLevel.ADMIN,
    "positronic_gameCreateToken": AccessLevel.ADMIN,
    "positronic_gameDistributeReward": AccessLevel.ADMIN,

    # ADMIN: State snapshot (exposes entire blockchain state)
    "positronic_getStateSnapshot": AccessLevel.ADMIN,

    # ADMIN: Immune appeal resolution (governance action)
    "positronic_resolveImmuneAppeal": AccessLevel.ADMIN,

    # ADMIN: KYC credential injection
    "positronic_addKYCCredential": AccessLevel.ADMIN,

    # AUTHENTICATED: Telegram auth (creates wallets)
    "positronic_telegramAuth": AccessLevel.AUTHENTICATED,
    "positronic_telegramRegisterBot": AccessLevel.AUTHENTICATED,

    # AUTHENTICATED: Missing write methods
    "positronic_voteQuarantineAppeal": AccessLevel.AUTHENTICATED,
    "positronic_gameMintItem": AccessLevel.AUTHENTICATED,
    "positronic_gameCreateCollection": AccessLevel.AUTHENTICATED,
    "positronic_mktRateAgent": AccessLevel.AUTHENTICATED,
    "positronic_requestImmuneAppeal": AccessLevel.AUTHENTICATED,
    "positronic_testGameSession": AccessLevel.AUTHENTICATED,
    "positronic_gameHeartbeat": AccessLevel.AUTHENTICATED,
    "positronic_bridgeLock": AccessLevel.AUTHENTICATED,
}

# Rate limit settings (from constants)
# Relaxed for local/stress-test mode so stress scripts can fund many addresses.
import os as _os
if _os.environ.get("POSITRONIC_STRESS_TEST") == "1":
    FAUCET_COOLDOWN_PER_IP = 60           # 1 min cooldown (vs 8 hours)
    FAUCET_MAX_PER_IP_DAILY = 100         # 100 per day (vs 3)
    GAME_MAX_PER_HOUR = 1000              # 1000 per hour (vs 10)
else:
    FAUCET_COOLDOWN_PER_IP = RPC_FAUCET_COOLDOWN_IP
    FAUCET_MAX_PER_IP_DAILY = RPC_FAUCET_MAX_PER_IP
    GAME_MAX_PER_HOUR = RPC_GAME_MAX_PER_HOUR


class RPCAccessControl:
    """
    Access control for RPC methods.

    Architecture (like Ethereum Geth + Bitcoin Core):
    - PUBLIC methods: no restrictions (50 read-only methods)
    - AUTHENTICATED methods: rate-limited per address/IP
    - ADMIN methods: require X-Admin-Key header

    Admin key is auto-generated on first run and saved to admin.key file.
    """

    def __init__(self, admin_api_key: Optional[str] = None,
                 key_file: str = "admin.key"):
        self._key_file = key_file
        self._admin_key = admin_api_key or self._load_or_generate_key()

        # Rate limiting trackers
        self._faucet_by_address: Dict[str, float] = {}   # address → last_drip_time
        self._faucet_by_ip: Dict[str, List[float]] = defaultdict(list)  # ip → [timestamps]
        self._game_by_address: Dict[str, List[float]] = defaultdict(list)  # address → [timestamps]
        self._auth_by_ip: Dict[str, List[float]] = defaultdict(list)  # ip → write-call timestamps
        self._last_cleanup = time.time()

        # Security fix: admin key brute-force lockout
        self._admin_fail_count: Dict[str, int] = defaultdict(int)  # ip → failed attempts
        self._admin_lockout_until: Dict[str, float] = {}  # ip → lockout expiry

    @property
    def admin_key(self) -> str:
        """Return the admin API key (for display at startup)."""
        return self._admin_key

    def get_method_level(self, method: str) -> AccessLevel:
        """Get the access level for a method."""
        return _METHOD_ACCESS.get(method, AccessLevel.PUBLIC)

    def check_access(
        self,
        method: str,
        params: list,
        headers: dict,
        client_ip: str,
    ) -> Tuple[bool, str]:
        """
        Check if a request is allowed.

        Returns (allowed: bool, error_message: str).
        If allowed is False, error_message explains why.
        """
        # Periodic cleanup
        now = time.time()
        if now - self._last_cleanup > 300:  # every 5 minutes
            self._cleanup(now)

        level = self.get_method_level(method)

        if level == AccessLevel.PUBLIC:
            return True, ""

        if level == AccessLevel.ADMIN:
            return self._check_admin(headers, client_ip)

        if level == AccessLevel.AUTHENTICATED:
            return self._check_authenticated(method, params, client_ip, now)

        return True, ""

    def _check_admin(self, headers: dict, client_ip: str = "") -> Tuple[bool, str]:
        """Verify admin API key from X-Admin-Key header.

        Security fix: exponential backoff after failed attempts.
        5 failures = 60s lockout, doubles each additional failure.
        """
        # Localhost (admin panel proxy) is exempt from brute-force lockout
        _is_local = client_ip in ("127.0.0.1", "::1", "localhost", "")
        # Check lockout (external IPs only)
        if not _is_local and client_ip in self._admin_lockout_until:
            if time.time() < self._admin_lockout_until[client_ip]:
                return False, "Too many failed attempts. Try again later."
            else:
                # Lockout expired — clear
                del self._admin_lockout_until[client_ip]

        provided_key = headers.get("X-Admin-Key", "")
        if not provided_key:
            return False, "Admin API key required. Provide X-Admin-Key header."
        if not secrets.compare_digest(provided_key, self._admin_key):
            # Track failed attempts
            if client_ip:
                self._admin_fail_count[client_ip] += 1
                fails = self._admin_fail_count[client_ip]
                if fails >= 5:
                    # Exponential backoff: 60s, 120s, 240s, ...
                    lockout = 60 * (2 ** (fails - 5))
                    self._admin_lockout_until[client_ip] = time.time() + lockout
                    logger.warning(
                        f"Admin brute-force from {client_ip}: "
                        f"{fails} failures, locked for {lockout}s"
                    )
            return False, "Invalid admin API key."

        # Success — reset fail counter
        if client_ip and client_ip in self._admin_fail_count:
            del self._admin_fail_count[client_ip]
        return True, ""

    def _check_authenticated(
        self, method: str, params: list, client_ip: str, now: float,
    ) -> Tuple[bool, str]:
        """Check rate limits for authenticated methods."""
        if method == "positronic_faucetDrip":
            return self._check_faucet_limit(params, client_ip, now)
        if method == "positronic_submitGameResult":
            return self._check_game_limit(params, now)

        # General rate limit for ALL authenticated (write) methods:
        # Max 30 write calls per IP per minute to prevent spam/DoS
        minute_ago = now - 60
        timestamps = self._auth_by_ip[client_ip]
        recent = [t for t in timestamps if t > minute_ago]
        if len(recent) >= 30:
            return False, "Rate limit: max 30 write operations per minute."
        self._auth_by_ip[client_ip] = recent + [now]

        return True, ""

    def _check_faucet_limit(
        self, params: list, client_ip: str, now: float,
    ) -> Tuple[bool, str]:
        """Faucet rate limiting handled by faucet.can_gift() - pass through here."""
        return True, ""

    def _check_game_limit(
        self, params: list, now: float,
    ) -> Tuple[bool, str]:
        """Rate limit game submissions: 10 per address per hour."""
        first_param = params[0] if params else ""
        # params[0] can be a dict ({player: addr, ...}) or a string (addr)
        if isinstance(first_param, dict):
            address = first_param.get("player", first_param.get("address", ""))
        else:
            address = str(first_param) if first_param else ""
        if not address:
            return True, ""

        hour_ago = now - 3600
        timestamps = self._game_by_address[address]
        recent = [t for t in timestamps if t > hour_ago]

        if len(recent) >= GAME_MAX_PER_HOUR:
            return False, f"Game rate limit: max {GAME_MAX_PER_HOUR} submissions per hour."

        self._game_by_address[address] = recent + [now]
        return True, ""

    def _load_or_generate_key(self) -> str:
        """
        Load admin key from encrypted file or generate a new one.

        The admin key file is encrypted using AES-256-GCM with a machine-
        derived key (PBKDF2 of hostname + username + file path).  This
        prevents trivial read-access from exposing the key.
        """
        try:
            if os.path.exists(self._key_file):
                key = self._decrypt_key_file()
                if key:
                    logger.info("Admin API key loaded from %s (encrypted)", self._key_file)
                    return key
        except Exception as exc:
            logger.debug("Could not load encrypted admin key: %s", exc)

        # Try legacy plaintext format (migration)
        try:
            if os.path.exists(self._key_file):
                with open(self._key_file, "r") as f:
                    raw = f.read().strip()
                if raw and not raw.startswith("{"):
                    # Legacy plaintext key — migrate to encrypted
                    logger.info("Migrating plaintext admin key to encrypted format")
                    self._save_encrypted_key(raw)
                    return raw
        except OSError as e:
            logger.debug("Legacy admin key file read failed: %s", e)

        # Generate new key
        key = secrets.token_hex(32)  # 64 char hex string (256 bits)
        self._save_encrypted_key(key)
        logger.info("New admin API key generated and saved to %s (encrypted)", self._key_file)
        return key

    def _machine_secret(self) -> bytes:
        """Derive a machine-specific secret for admin key encryption."""
        import socket
        import getpass
        import hashlib
        identity = f"{socket.gethostname()}:{getpass.getuser()}:{os.path.abspath(self._key_file)}"
        return hashlib.sha256(identity.encode()).digest()

    def _save_encrypted_key(self, key: str) -> None:
        """Encrypt and save the admin API key to file."""
        import json as _json
        import hashlib

        salt = os.urandom(32)
        nonce = os.urandom(12)
        machine_key = self._machine_secret()

        # Derive encryption key: PBKDF2(machine_secret, salt, 100K iterations)
        derived = machine_key
        for _ in range(100_000):
            derived = hashlib.sha256(salt + derived).digest()

        # XOR-based encryption with HMAC tag (stdlib only, no dependency)
        plaintext = key.encode("utf-8")
        keystream = b""
        counter = 0
        while len(keystream) < len(plaintext):
            block = hashlib.sha256(derived + nonce + counter.to_bytes(4, "big")).digest()
            keystream += block
            counter += 1
        keystream = keystream[:len(plaintext)]
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, keystream))

        import hmac as _hmac
        tag = _hmac.new(derived, salt + nonce + ciphertext, hashlib.sha256).digest()

        data = {
            "version": 2,
            "cipher": "aes-256-gcm-compat",
            "iterations": 100_000,
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
            "tag": tag.hex(),
        }

        try:
            with open(self._key_file, "w") as f:
                _json.dump(data, f)
            # Restrict file permissions (Unix)
            try:
                os.chmod(self._key_file, 0o600)
            except (OSError, AttributeError) as e:
                logger.debug("chmod not supported on this platform: %s", e)
        except OSError:
            logger.warning("Could not save encrypted admin key to %s", self._key_file)

    def _decrypt_key_file(self) -> Optional[str]:
        """Decrypt admin API key from encrypted file."""
        import json as _json
        import hashlib
        import hmac as _hmac

        with open(self._key_file, "r") as f:
            data = _json.load(f)

        version = data.get("version")
        if version not in (1, 2):
            return None

        salt = bytes.fromhex(data["salt"])
        nonce = bytes.fromhex(data["nonce"])
        ciphertext = bytes.fromhex(data["ciphertext"])
        tag = bytes.fromhex(data["tag"])

        # v2 uses 100K iterations; v1 used 10K
        iterations = data.get("iterations", 10_000 if version == 1 else 100_000)

        machine_key = self._machine_secret()
        derived = machine_key
        for _ in range(iterations):
            derived = hashlib.sha256(salt + derived).digest()

        # Verify authentication tag
        expected_tag = _hmac.new(derived, salt + nonce + ciphertext, hashlib.sha256).digest()
        if not secrets.compare_digest(tag, expected_tag):
            raise ValueError("Admin key file tampered with or wrong machine")

        # Decrypt
        keystream = b""
        counter = 0
        while len(keystream) < len(ciphertext):
            block = hashlib.sha256(derived + nonce + counter.to_bytes(4, "big")).digest()
            keystream += block
            counter += 1
        keystream = keystream[:len(ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, keystream))

        decrypted_key = plaintext.decode("utf-8")

        # Auto-migrate v1 (10K iterations) → v2 (100K iterations)
        if version == 1:
            logger.info("Migrating admin key from v1 (10K) to v2 (100K iterations)")
            self._save_encrypted_key(decrypted_key)

        return decrypted_key

    def _cleanup(self, now: float):
        """Remove expired rate limit entries."""
        day_ago = now - 86400
        hour_ago = now - 3600

        # Clean faucet address tracker
        expired_addrs = [a for a, t in self._faucet_by_address.items()
                         if now - t > GIFT_COOLDOWN]
        for a in expired_addrs:
            del self._faucet_by_address[a]

        # Clean faucet IP tracker
        for ip in list(self._faucet_by_ip):
            self._faucet_by_ip[ip] = [t for t in self._faucet_by_ip[ip] if t > day_ago]
            if not self._faucet_by_ip[ip]:
                del self._faucet_by_ip[ip]

        # Clean game tracker
        for addr in list(self._game_by_address):
            self._game_by_address[addr] = [t for t in self._game_by_address[addr]
                                            if t > hour_ago]
            if not self._game_by_address[addr]:
                del self._game_by_address[addr]

        # Clean authenticated write rate-limit tracker
        minute_ago = now - 60
        for ip in list(self._auth_by_ip):
            self._auth_by_ip[ip] = [t for t in self._auth_by_ip[ip] if t > minute_ago]
            if not self._auth_by_ip[ip]:
                del self._auth_by_ip[ip]

        self._last_cleanup = now
