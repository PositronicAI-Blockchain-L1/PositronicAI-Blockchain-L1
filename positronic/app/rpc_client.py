"""
Lightweight synchronous JSON-RPC client for localhost node queries.

Uses urllib.request (stdlib) — no extra dependencies needed.
Designed to be called from the tkinter main thread via root.after().
Localhost RPC calls return in <10ms so brief blocking is acceptable.

Security:
  - Input validation on all address parameters
  - Response type checking for known methods
  - TLS auto-detection for remote nodes
  - Bounded payload size (1 MB max response)
"""

import json
import logging
import re
import ssl
import urllib.request
import urllib.error
from typing import Any, Optional

logger = logging.getLogger("positronic.app.rpc_client")

# Max response size to prevent memory exhaustion (1 MB)
_MAX_RESPONSE_BYTES = 1_048_576

# Allowed RPC method name pattern (prevents injection)
_METHOD_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{1,64}$")

# Valid hex address: 0x + 40 hex chars
_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class RPCClient:
    """Synchronous JSON-RPC 2.0 client for the local Positronic node."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8545,
                 use_tls: bool = False, auto_detect_tls: bool = False):
        self._host = host
        self._port = port
        self._id_counter = 0
        self._use_tls = use_tls
        self._auto_detect = auto_detect_tls
        self._tls_detected = False  # True once auto-detection is done
        # Remote hosts get longer default timeout
        self._default_timeout = 8.0 if host not in ("127.0.0.1", "localhost", "::1") else 3.0

        scheme = "https" if use_tls else "http"
        self.url = f"{scheme}://{host}:{port}/"

        # Failure tracking for backoff
        self._consecutive_failures = 0
        self._backoff_until = 0.0  # time.time() when backoff expires

        # Cache for get_node_info (3-second TTL) to reduce redundant RPC calls
        self._node_info_cache: Optional[dict] = None
        self._node_info_cache_ts: float = 0.0

        # For self-signed certs on localhost, create permissive context
        self._ssl_ctx: Optional[ssl.SSLContext] = None
        self._init_ssl_ctx(host, use_tls)

    def _init_ssl_ctx(self, host: str, use_tls: bool):
        """Initialize SSL context for TLS connections."""
        self._ssl_ctx = None
        if use_tls:
            self._ssl_ctx = ssl.create_default_context()
            if host in ("127.0.0.1", "localhost", "::1"):
                self._ssl_ctx.check_hostname = False
                self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def _switch_protocol(self, to_tls: bool):
        """Switch between HTTP and HTTPS."""
        self._use_tls = to_tls
        scheme = "https" if to_tls else "http"
        self.url = f"{scheme}://{self._host}:{self._port}/"
        self._init_ssl_ctx(self._host, to_tls)
        self._tls_detected = True
        logger.debug("RPC protocol switched to %s", scheme.upper())

    @staticmethod
    def validate_address(address: str) -> bool:
        """Validate an Ethereum-style hex address (0x + 40 hex chars)."""
        return bool(_ADDRESS_RE.match(address))

    @staticmethod
    def sanitize_hex(value: str) -> Optional[int]:
        """Safely parse a hex string to int, returning None on failure."""
        if not isinstance(value, str):
            return int(value) if isinstance(value, (int, float)) else None
        value = value.strip()
        if value.startswith("0x") or value.startswith("0X"):
            try:
                return int(value, 16)
            except (ValueError, OverflowError):
                return None
        return None

    def _raw_call(self, method: str, payload: bytes,
                  timeout: float) -> Optional[Any]:
        """Execute a single RPC call with current protocol settings."""
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        open_kwargs = {"timeout": timeout}
        if self._ssl_ctx:
            open_kwargs["context"] = self._ssl_ctx

        with urllib.request.urlopen(req, **open_kwargs) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                logger.warning("RPC response too large (>1MB), "
                               "dropping: %s", method)
                return None
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                logger.warning("RPC response is not a JSON object")
                return None
            if "error" in data:
                logger.debug("RPC error: %s", data["error"])
                return None
            return data.get("result")

    def call(self, method: str, params: Optional[list] = None,
             timeout: float = None) -> Optional[Any]:
        """
        Make a JSON-RPC 2.0 call to the local node.
        Returns the 'result' field, or None on any error.

        If auto_detect_tls is enabled, tries current protocol first,
        then falls back to the other protocol and remembers the result.
        """
        # Skip during backoff after consecutive failures
        import time as _time
        if self._consecutive_failures >= 3 and _time.time() < self._backoff_until:
            return None

        # Validate method name to prevent injection
        if not _METHOD_RE.match(method):
            logger.warning("Invalid RPC method name rejected: %s",
                           method[:32])
            return None

        if timeout is None:
            timeout = self._default_timeout

        _SENSITIVE_METHODS = {"positronic_stake", "positronic_unstake",
                               "positronic_transfer",
                               "positronic_claimStakingRewards",
                               "eth_sendRawTransaction"}
        log_params = "[redacted]" if method in _SENSITIVE_METHODS else params
        logger.debug("RPC → %s %s", method, log_params)

        self._id_counter += 1
        payload = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or [],
            "id": self._id_counter,
        }).encode("utf-8")

        try:
            result = self._raw_call(method, payload, timeout)
            self._consecutive_failures = 0
            return result
        except (urllib.error.URLError, TimeoutError, OSError,
                json.JSONDecodeError, Exception) as exc:
            # If auto-detection enabled and not yet detected, try other protocol
            if self._auto_detect and not self._tls_detected:
                alt_tls = not self._use_tls
                logger.debug("RPC %s failed (%s), trying %s...",
                             method, exc,
                             "HTTPS" if alt_tls else "HTTP")
                self._switch_protocol(alt_tls)
                try:
                    result = self._raw_call(method, payload, timeout)
                    # Success on alt protocol — lock it in
                    self._tls_detected = True
                    return result
                except Exception as exc2:
                    # Both protocols failed — revert to original and lock to prevent retry storm
                    self._switch_protocol(not alt_tls)
                    self._tls_detected = True  # Lock to prevent retry storm every call
                    logger.debug("RPC call %s failed on both protocols: %s",
                                 method, exc2)
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 3:
                        self._backoff_until = _time.time() + 10.0
                    return None
            logger.debug("RPC call %s failed: %s", method, exc)
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._backoff_until = _time.time() + 10.0
            return None

    # --- Convenience methods ---

    def get_block_number(self) -> Optional[int]:
        result = self.call("eth_blockNumber")
        return self.sanitize_hex(result) if result is not None else None

    def get_node_info(self) -> Optional[dict]:
        import time as _time
        now = _time.time()
        if self._node_info_cache is not None and (now - self._node_info_cache_ts) < 3.0:
            return self._node_info_cache
        result = self.call("positronic_nodeInfo")
        if result is not None:
            self._node_info_cache = result
            self._node_info_cache_ts = now
        return result

    def get_ai_stats(self) -> Optional[dict]:
        return self.call("positronic_getAIStats")

    def get_network_health(self) -> Optional[dict]:
        return self.call("positronic_getNetworkHealth")

    def get_chain_id(self) -> Optional[int]:
        result = self.call("eth_chainId")
        return self.sanitize_hex(result) if result is not None else None

    def reset_detection(self):
        """Reset TLS auto-detection to allow re-discovery after failures."""
        if self._auto_detect:
            self._tls_detected = False

    def is_alive(self) -> bool:
        """Quick health check — is the RPC server responding?"""
        try:
            result = self.call("eth_chainId", timeout=3.0)
            if result is not None:
                self._consecutive_failures = 0
                return True
            return False
        except Exception as e:
            logger.debug("RPC health check failed: %s", e)
            return False

    # --- Ecosystem data methods ---

    def get_neural_status(self) -> Optional[dict]:
        return self.call("positronic_getNeuralStatus")

    def get_governance_stats(self) -> Optional[dict]:
        return self.call("positronic_getGovernanceStats")

    def get_did_stats(self) -> Optional[dict]:
        return self.call("positronic_getDIDStats")

    def get_bridge_stats(self) -> Optional[dict]:
        return self.call("positronic_getBridgeStats")

    def get_depin_stats(self) -> Optional[dict]:
        return self.call("positronic_getDePINStats")

    def get_agent_stats(self) -> Optional[dict]:
        return self.call("positronic_getAgentStats")

    def get_marketplace_stats(self) -> Optional[dict]:
        return self.call("positronic_mktGetStats")

    def get_rwa_stats(self) -> Optional[dict]:
        return self.call("positronic_getRWAStats")

    def get_zkml_stats(self) -> Optional[dict]:
        return self.call("positronic_getZKMLStats")

    def get_trust_stats(self) -> Optional[dict]:
        return self.call("positronic_getTrustStats")

    def get_token_registry_stats(self) -> Optional[dict]:
        return self.call("positronic_getTokenRegistryStats")

    def get_consensus_info(self) -> Optional[dict]:
        return self.call("positronic_getConsensusInfo")

    def get_pq_stats(self) -> Optional[dict]:
        return self.call("positronic_getPQStats")

    def get_checkpoint_stats(self) -> Optional[dict]:
        return self.call("positronic_getCheckpointStats")

    def get_cold_start_status(self) -> Optional[dict]:
        return self.call("positronic_getColdStartStatus")

    def get_pathway_health(self) -> Optional[dict]:
        return self.call("positronic_getPathwayHealth")

    def get_drift_alerts(self) -> Optional[dict]:
        return self.call("positronic_getDriftAlerts")

    def get_immune_status(self) -> Optional[dict]:
        return self.call("positronic_getImmuneStatus")
