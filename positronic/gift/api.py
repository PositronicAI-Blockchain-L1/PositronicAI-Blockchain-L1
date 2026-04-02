"""
Positronic - Gift REST API
REST API endpoint for website integration.
Allows websites to gift ASF coins to customers.
"""

import json
import logging
from aiohttp import web
from typing import Optional

from positronic.gift.faucet import GiftFaucet
from positronic.constants import BASE_UNIT

logger = logging.getLogger("positronic.gift.api")


class GiftAPI:
    """
    REST API for the gift system.
    Integrate this with your website to gift ASF to customers.

    Endpoints:
    - POST /api/gift/send     - Send a gift
    - GET  /api/gift/check    - Check if address can receive gift
    - GET  /api/gift/stats    - Get faucet statistics
    - GET  /api/gift/history  - Get gift history
    """

    def __init__(self, faucet: GiftFaucet):
        self.faucet = faucet

    def setup_routes(self, app: web.Application):
        """Register routes on an aiohttp application."""
        app.router.add_post("/api/gift/send", self.handle_send)
        app.router.add_get("/api/gift/check/{address}", self.handle_check)
        app.router.add_get("/api/gift/stats", self.handle_stats)
        app.router.add_get("/api/gift/history", self.handle_history)

    async def handle_send(self, request: web.Request) -> web.Response:
        """Send a gift to an address."""
        try:
            body = await request.json()
            address = body.get("address", "")
            amount = body.get("amount")
            message = body.get("message", "")

            if not address or not address.startswith("0x"):
                return web.json_response(
                    {"error": "Invalid address format"}, status=400
                )

            # Validate address length: "0x" + 40 hex chars = 20 bytes
            addr_hex = address[2:]
            if len(addr_hex) != 40:
                return web.json_response(
                    {"error": "Invalid address length (expected 20 bytes / 40 hex chars)"}, status=400
                )
            try:
                bytes.fromhex(addr_hex)
            except ValueError:
                return web.json_response(
                    {"error": "Invalid address: contains non-hex characters"}, status=400
                )

            # Convert amount from ASF to base units
            amount_base = int(amount * BASE_UNIT) if amount else None

            tx = self.faucet.create_gift_transaction(
                address, amount=amount_base, message=message
            )

            if tx is None:
                can_send, reason = self.faucet.can_gift(address)
                return web.json_response(
                    {"error": reason, "success": False}, status=429
                )

            return web.json_response({
                "success": True,
                "tx_hash": tx.tx_hash_hex,
                "amount": tx.value / BASE_UNIT,
                "recipient": address,
                "message": "Gift sent successfully!",
            })

        except Exception as e:
            logger.warning("gift_send_failed: %s", e)
            return web.json_response(
                {"error": str(e), "success": False}, status=500
            )

    async def handle_check(self, request: web.Request) -> web.Response:
        """Check if an address can receive a gift."""
        address = request.match_info.get("address", "")
        can_send, reason = self.faucet.can_gift(address)
        return web.json_response({
            "address": address,
            "can_receive": can_send,
            "reason": reason,
            "gift_amount": self.faucet.gift_amount / BASE_UNIT,
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        """Get faucet statistics."""
        return web.json_response(self.faucet.get_stats())

    async def handle_history(self, request: web.Request) -> web.Response:
        """Get gift history."""
        limit = int(request.query.get("limit", "50"))
        return web.json_response(self.faucet.get_history(limit))
