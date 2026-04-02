"""
Positronic - Standalone JSON-RPC Server
Provides Ethereum-compatible JSON-RPC API for MetaMask/Trust Wallet integration
and Positronic-specific endpoints.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Optional, List

from positronic.rpc.types import (
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INTERNAL_ERROR,
)
from positronic.rpc.methods import RPCMethods
from positronic.rpc.security import RPCAccessControl, ACCESS_DENIED_CODE
from positronic.rpc.validator import sanitize_rpc_params, validate_params_size

logger = logging.getLogger(__name__)


class _RateLimiter:
    """
    Multi-tier per-IP rate limiter with burst protection.

    Three sliding windows protect against different attack patterns:
    - Burst:    max 50 requests / 1 second   (prevents flash floods)
    - Sustained: max 500 requests / 60 seconds (prevents sustained abuse)
    - Daily:    max 50,000 requests / 86400 sec (prevents long-term abuse)

    Also tracks per-IP ban list for repeat offenders (10+ violations → 15 min ban).
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0,
                 max_daily: int = 50_000, ban_enabled: bool = True):
        # Short-burst window (original behavior)
        self.max_burst = max_requests
        self.burst_window = window_seconds
        # Sustained window (1 minute)
        self.max_sustained = max_requests * 5  # 500/min default
        self.sustained_window = 60.0
        # Daily cap
        self.max_daily = max_daily
        self.daily_window = 86400.0
        # Whether to auto-ban repeat offenders
        self._ban_enabled = ban_enabled

        # ip -> list of request timestamps (monotonic)
        self._requests: dict = defaultdict(list)
        self._last_cleanup = time.monotonic()
        # ip -> (violations, ban_until)
        self._bans: dict = {}
        self._violations: dict = defaultdict(int)

    def is_allowed(self, ip: str) -> bool:
        """Return True if the request should be allowed."""
        now = time.monotonic()

        # Periodic cleanup to avoid memory leak
        if now - self._last_cleanup > 60:
            self._cleanup(now)

        # Check ban list
        if ip in self._bans:
            if now < self._bans[ip]:
                return False
            else:
                del self._bans[ip]

        timestamps = self._requests[ip]

        # Check burst limit (1 second window)
        burst_cutoff = now - self.burst_window
        burst_count = sum(1 for t in timestamps if t > burst_cutoff)
        if burst_count >= self.max_burst:
            self._record_violation(ip, now)
            return False

        # Check sustained limit (60 second window)
        sustained_cutoff = now - self.sustained_window
        sustained_count = sum(1 for t in timestamps if t > sustained_cutoff)
        if sustained_count >= self.max_sustained:
            self._record_violation(ip, now)
            return False

        # Check daily limit
        daily_cutoff = now - self.daily_window
        daily_count = sum(1 for t in timestamps if t > daily_cutoff)
        if daily_count >= self.max_daily:
            self._record_violation(ip, now)
            return False

        # Allowed — record timestamp
        self._requests[ip].append(now)

        # Trim old entries (keep only last 24h)
        if len(self._requests[ip]) > self.max_daily:
            self._requests[ip] = [
                t for t in self._requests[ip] if t > daily_cutoff
            ]

        return True

    def _record_violation(self, ip: str, now: float):
        """Track violations; auto-ban repeat offenders for 15 minutes."""
        if not self._ban_enabled:
            return  # Stress/local mode: no banning
        self._violations[ip] = self._violations.get(ip, 0) + 1
        if self._violations[ip] >= 10:
            self._bans[ip] = now + 900  # 15 minute ban
            logger.warning("IP %s banned for 15 min (repeated rate limit violations)", ip)
            self._violations[ip] = 0

    def _cleanup(self, now: float):
        """Remove stale entries."""
        cutoff = now - self.daily_window
        stale = [ip for ip, ts in self._requests.items()
                 if not ts or ts[-1] < cutoff]
        for ip in stale:
            del self._requests[ip]
        # Clean expired bans
        expired_bans = [ip for ip, until in self._bans.items() if now >= until]
        for ip in expired_bans:
            del self._bans[ip]
        # Clean old violations
        if len(self._violations) > 10_000:
            self._violations.clear()
        self._last_cleanup = now


class RPCServer:
    """
    JSON-RPC 2.0 Server for Positronic.

    Handles:
    - eth_* methods (MetaMask/Trust Wallet compatibility)
    - positronic_* methods (Positronic-specific features)
    - net_* methods (network info)
    - web3_* methods (web3 compatibility)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8545,
        blockchain=None,
        mempool=None,
        cors_origins: Optional[List[str]] = None,
        rate_limit: int = 100,
        admin_api_key: Optional[str] = None,
        network_type: str = "main",
        backlog: int = 128,
        max_connections: int = 500,
        ssl_context=None,
    ):
        self.host = host
        self.port = port
        self._network_type = network_type
        self._backlog = backlog
        self._max_connections = max_connections
        self._active_connections = 0
        self._ssl_context = ssl_context
        self._access_control = RPCAccessControl(admin_api_key=admin_api_key)
        self.rpc = RPCMethods(blockchain, mempool, access_control=self._access_control)
        self._app = None
        self._runner = None
        # CORS: default to localhost + production origins; pass ["*"] to allow all.
        # Security: no wildcard chrome-extension — prevents malicious extensions
        # from accessing the RPC. Add specific extension IDs if needed.
        self._cors_origins = cors_origins if cors_origins is not None else [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8545",
            "http://127.0.0.1",
            "http://127.0.0.1:8545",
            "https://positronic-ai.network",
            "https://www.positronic-ai.network",
            "https://rpc.positronic-ai.network",
            "https://testnet.positronic-ai.network",
            "https://testnet-rpc.positronic-ai.network",
        ]
        # Relaxed rate limits for local/testnet/stress-test mode so
        # development and testing can proceed without 429 blocking.
        import os
        if network_type in ("local", "testnet") or os.environ.get("POSITRONIC_STRESS_TEST") == "1":
            self._rate_limiter = _RateLimiter(
                max_requests=500, window_seconds=1.0,
                max_daily=100_000, ban_enabled=True,
            )
            logger.info("RPC rate limiter: RELAXED mode (%s)", network_type)
        else:
            self._rate_limiter = _RateLimiter(max_requests=rate_limit, window_seconds=1.0)

    def _cors_origin(self, request_origin: str = "") -> str:
        """Return the appropriate Access-Control-Allow-Origin value.

        Security fix: uses exact match instead of startswith() to prevent
        CORS prefix attacks (e.g., 'http://localhost:8545.evil.com' matching
        'http://localhost:8545').
        """
        if "*" in self._cors_origins:
            return "*"
        if not request_origin:
            return ""
        for allowed in self._cors_origins:
            if allowed.endswith("*"):
                # Wildcard pattern: match prefix up to the wildcard
                prefix = allowed[:-1]
                if request_origin.startswith(prefix):
                    return request_origin
            elif request_origin == allowed:
                # Security fix: exact match only (was startswith)
                return request_origin
        return ""

    async def start(self):
        """Start the RPC server."""
        try:
            from aiohttp import web

            @web.middleware
            async def connection_limit_middleware(request, handler):
                if self._active_connections >= self._max_connections:
                    return web.json_response(
                        {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Server busy"}, "id": None},
                        status=503,
                    )
                self._active_connections += 1
                try:
                    return await handler(request)
                finally:
                    self._active_connections -= 1

            self._app = web.Application(middlewares=[connection_limit_middleware])
            self._app.router.add_post("/", self._handle_rpc)
            self._app.router.add_options("/", self._handle_cors)
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/metrics", self._handle_metrics)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            site = web.TCPSite(
                self._runner, self.host, self.port,
                backlog=self._backlog, ssl_context=self._ssl_context,
            )
            await site.start()
            proto = "https" if self._ssl_context else "http"
            logger.info("RPC server started on %s://%s:%s", proto, self.host, self.port)
        except ImportError:
            logger.warning("aiohttp not installed. RPC server disabled.")

    async def stop(self):
        """Stop the RPC server."""
        if self._runner:
            await self._runner.cleanup()
            logger.info("RPC server stopped.")

    async def _handle_cors(self, request):
        """Handle CORS preflight requests."""
        from aiohttp import web

        origin = request.headers.get("Origin", "")
        allowed = self._cors_origin(origin)
        return web.Response(
            status=200,
            headers={
                "Access-Control-Allow-Origin": allowed or "null",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, X-Admin-Key",
            },
        )

    async def _handle_health(self, request):
        """Health check endpoint for Docker HEALTHCHECK and monitoring."""
        from aiohttp import web

        health = {"status": "ok", "chain": "Positronic"}
        if self.rpc.blockchain:
            try:
                health["block_height"] = self.rpc.blockchain.height
            except Exception as e:
                logger.debug("health_check: could not read block height: %s", e)
        return web.json_response(health)

    async def _handle_metrics(self, request):
        """Prometheus-compatible metrics endpoint (GET /metrics).

        Security: requires X-Admin-Key header to prevent information leakage
        about internal node state (memory, CPU, peer count, block processing
        times, etc.) to unauthenticated callers.
        """
        from aiohttp import web
        import secrets as _secrets

        # Verify admin key for metrics access
        provided_key = request.headers.get("X-Admin-Key", "")
        if not provided_key or not _secrets.compare_digest(
            provided_key, self._access_control.admin_key
        ):
            return web.json_response(
                {"error": "Admin key required for /metrics"},
                status=403,
            )

        try:
            from positronic.monitoring.metrics import REGISTRY
            body = REGISTRY.expose_all()
            return web.Response(
                text=body,
                content_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except Exception as e:
            logger.error("metrics_endpoint_failed: %s", e)
            return web.Response(text="# error collecting metrics\n", status=500)

    async def _handle_rpc(self, request):
        """Handle JSON-RPC requests."""
        from aiohttp import web

        from positronic.constants import API_VERSION

        origin = request.headers.get("Origin", "")
        allowed_origin = self._cors_origin(origin)
        # CORS + API versioning + security headers
        headers = {
            "Access-Control-Allow-Origin": allowed_origin or "null",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, X-Admin-Key",
            "X-Positronic-API-Version": API_VERSION,
            # Security headers
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'none'",
            "Cache-Control": "no-store, no-cache, must-revalidate",
        }
        # HSTS: only send when TLS is active (sending over plain HTTP is a spec violation)
        if self._ssl_context:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Rate limiting
        client_ip = request.headers.get("X-Real-IP", request.headers.get("X-Forwarded-For", request.headers.get("CF-Connecting-IP", request.remote or "unknown"))).split(",")[0].strip()
        if not self._rate_limiter.is_allowed(client_ip):
            error_resp = JSONRPCResponse(
                error=JSONRPCError(-32099, "Rate limit exceeded").to_dict(),
                id=None,
            )
            return web.json_response(
                error_resp.to_dict(), headers=headers, status=429,
            )

        try:
            body = await request.json()
        except json.JSONDecodeError:
            error_resp = JSONRPCResponse(
                error=JSONRPCError(PARSE_ERROR, "Parse error").to_dict(),
                id=None,
            )
            return web.json_response(error_resp.to_dict(), headers=headers)

        # Handle batch requests
        # Security fix: count each item in batch against rate limiter
        if isinstance(body, list):
            MAX_BATCH_SIZE = 20  # Limit batch size
            if len(body) > MAX_BATCH_SIZE:
                error_resp = JSONRPCResponse(
                    error=JSONRPCError(-32600, f"Batch too large: max {MAX_BATCH_SIZE} requests").to_dict(),
                    id=None,
                )
                return web.json_response(error_resp.to_dict(), headers=headers, status=400)

            # Count each batch item against rate limiter
            client_ip = request.headers.get("X-Real-IP", request.headers.get("X-Forwarded-For", request.headers.get("CF-Connecting-IP", request.remote or "unknown"))).split(",")[0].strip()
            for _ in body[1:]:  # First was already counted above
                if not self._rate_limiter.is_allowed(client_ip):
                    error_resp = JSONRPCResponse(
                        error=JSONRPCError(-32099, "Rate limit exceeded (batch)").to_dict(),
                        id=None,
                    )
                    return web.json_response(error_resp.to_dict(), headers=headers, status=429)

            results = []
            for req_data in body:
                result = await self._process_request(req_data, request)
                results.append(result)
            return web.json_response(results, headers=headers)

        # Single request
        result = await self._process_request(body, request)
        return web.json_response(result, headers=headers)

    async def _process_request(self, data: dict, request=None) -> dict:
        """Process a single JSON-RPC request with access control."""
        try:
            req = JSONRPCRequest.from_dict(data)
        except Exception as e:
            logger.debug("invalid_rpc_request: %s", e)
            return JSONRPCResponse(
                error=JSONRPCError(INVALID_REQUEST, "Invalid request").to_dict(),
                id=data.get("id"),
            ).to_dict()

        # Access control check
        req_headers = dict(request.headers) if request else {}
        client_ip = request.headers.get("X-Real-IP", request.headers.get("X-Forwarded-For", request.remote or "unknown")).split(",")[0].strip() if request else "unknown"
        try:
            allowed, reason = self._access_control.check_access(
                req.method, sanitize_rpc_params(req.params), req_headers, client_ip,
            )
        except Exception as e:
            logger.error("Access control error for %s: %s", req.method, e)
            allowed, reason = False, "Internal error"
        if not allowed:
            return JSONRPCResponse(
                error=JSONRPCError(ACCESS_DENIED_CODE, reason).to_dict(),
                id=req.id,
            ).to_dict()

        # Input validation: sanitize and size-check params
        safe_params = sanitize_rpc_params(req.params)
        if not validate_params_size(safe_params):
            return JSONRPCResponse(
                error=JSONRPCError(INVALID_REQUEST, "Parameters too large").to_dict(),
                id=req.id,
            ).to_dict()

        try:
            result = self.rpc.handle(req.method, safe_params, client_ip=client_ip)
            return JSONRPCResponse(result=result, id=req.id).to_dict()
        except KeyError:
            return JSONRPCResponse(
                error=JSONRPCError(
                    METHOD_NOT_FOUND,
                    f"Method not found: {req.method}",
                ).to_dict(),
                id=req.id,
            ).to_dict()
        except Exception as e:
            logger.error("RPC error for method %s: %s", data.get("method"), e, exc_info=True)
            # Security: never expose raw exception details to clients.
            # Internal errors are logged server-side; clients get generic message.
            return JSONRPCResponse(
                error=JSONRPCError(INTERNAL_ERROR, "Internal error").to_dict(),
                id=req.id,
            ).to_dict()

    def update_blockchain(self, blockchain):
        """Update the blockchain reference."""
        self.rpc.blockchain = blockchain

    def update_mempool(self, mempool):
        """Update the mempool reference."""
        self.rpc.mempool = mempool
