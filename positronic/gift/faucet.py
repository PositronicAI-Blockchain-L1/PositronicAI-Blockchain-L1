"""
Positronic - Gift Faucet
Controlled faucet for distributing ASF coins to website customers.
Rate-limited and configurable.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field

from positronic.crypto.keys import KeyPair
from positronic.core.transaction import Transaction, TxType
from positronic.crypto.address import address_from_hex
from positronic.constants import (
    GIFT_DEFAULT_AMOUNT,
    GIFT_MAX_DAILY,
    GIFT_COOLDOWN,
    BASE_UNIT,
    CHAIN_ID,
    TX_BASE_GAS,
)


@dataclass
class GiftRecord:
    """Record of a gift sent to an address."""
    recipient: str
    amount: int
    tx_hash: str
    timestamp: float
    message: str = ""


class GiftFaucet:
    """
    Controlled faucet for gifting ASF coins to customers.

    Features:
    - One claim per IP address, forever
    - One claim per recipient address, forever
    - Daily gift limit (safety cap)
    - Configurable gift amounts
    - Gift history tracking
    """

    def __init__(
        self,
        faucet_keypair: KeyPair,
        gift_amount: int = GIFT_DEFAULT_AMOUNT,
        max_daily: int = GIFT_MAX_DAILY,
        cooldown: int = GIFT_COOLDOWN,
    ):
        self.keypair = faucet_keypair
        self.gift_amount = gift_amount
        self.max_daily = max_daily
        self.cooldown = cooldown

        # Tracking
        self.last_gift_time: Dict[str, float] = {}  # address -> timestamp
        self.claimed_ips: set = set()  # IPs that have already claimed (forever ban)
        self.claimed_addresses: set = set()  # addresses that have already claimed
        self.daily_count: int = 0
        self.daily_reset_time: float = time.time()
        self.history: list = []
        self._nonce: int = 0
        self._state = None  # Optional: link to on-chain state for nonce sync

    def can_gift(self, recipient_address: str, client_ip: str = None) -> tuple:
        """Check if a gift can be sent to this address/IP. One claim per IP, forever."""
        # Check if IP already claimed (permanent, one-time only)
        if client_ip and client_ip != "unknown" and client_ip in self.claimed_ips:
            return False, "Faucet already claimed from this IP address. One claim per IP, forever."

        # Check if address already claimed (permanent, one-time only)
        if recipient_address in self.claimed_addresses:
            return False, "This address has already received faucet tokens."

        # Reset daily counter
        if time.time() - self.daily_reset_time > 86400:
            self.daily_count = 0
            self.daily_reset_time = time.time()

        # Check daily limit (safety cap)
        if self.daily_count >= self.max_daily:
            return False, "Daily gift limit reached. Try again tomorrow."

        return True, "OK"

    def record_claim(self, recipient_address: str, client_ip: str = None):
        """Record that an address/IP has claimed. Called after successful gift."""
        if client_ip and client_ip != "unknown":
            self.claimed_ips.add(client_ip)
        self.claimed_addresses.add(recipient_address)

    def create_gift_transaction(
        self,
        recipient_address: str,
        amount: int = None,
        message: str = "",
        client_ip: str = None,
    ) -> Optional[Transaction]:
        """
        Create a gift transaction.

        Args:
            recipient_address: 0x-prefixed hex address
            amount: Gift amount in base units (default: GIFT_DEFAULT_AMOUNT)
            message: Optional gift message
            client_ip: Client IP for once-per-IP enforcement

        Returns:
            Signed Transaction or None if rate-limited
        """
        can_send, reason = self.can_gift(recipient_address, client_ip)
        if not can_send:
            return None

        gift_value = amount or self.gift_amount
        recipient = address_from_hex(recipient_address)

        # Sync nonce with on-chain state if available, to prevent
        # nonce collisions when multiple faucet instances are running
        # or after node restarts.
        if self._state is not None:
            from positronic.crypto.address import address_from_pubkey
            faucet_addr = address_from_pubkey(self.keypair.public_key_bytes)
            on_chain_nonce = self._state.get_nonce(faucet_addr)
            if on_chain_nonce > self._nonce:
                self._nonce = on_chain_nonce

        tx = Transaction(
            tx_type=TxType.TRANSFER,
            nonce=self._nonce,
            sender=self.keypair.public_key_bytes,
            recipient=recipient,
            value=gift_value,
            gas_price=1,
            gas_limit=TX_BASE_GAS,
            data=message.encode("utf-8")[:256] if message else b"",
            chain_id=CHAIN_ID,
        )
        tx.sign(self.keypair)

        # Update tracking
        self.last_gift_time[recipient_address] = time.time()
        self.daily_count += 1
        self._nonce += 1
        self.record_claim(recipient_address, client_ip)

        self.history.append(GiftRecord(
            recipient=recipient_address,
            amount=gift_value,
            tx_hash=tx.tx_hash_hex,
            timestamp=time.time(),
            message=message,
        ))

        return tx

    def get_stats(self) -> dict:
        return {
            "faucet_address": self.keypair.address_hex,
            "gift_amount_sma": self.gift_amount / BASE_UNIT,
            "daily_sent": self.daily_count,
            "daily_limit": self.max_daily,
            "total_gifts": len(self.history),
            "unique_recipients": len(self.last_gift_time),
        }

    def get_history(self, limit: int = 50) -> list:
        return [
            {
                "recipient": r.recipient,
                "amount": r.amount / BASE_UNIT,
                "tx_hash": r.tx_hash,
                "time": r.timestamp,
                "message": r.message,
            }
            for r in self.history[-limit:]
        ]
