"""
Positronic - Telegram Mini App Integration
Maps Telegram users to blockchain wallets for play-to-mine in Telegram games.
"""

import hashlib
import hmac
import time
import secrets
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, unquote

from positronic.utils.logging import get_logger
from positronic.crypto.keys import KeyPair

logger = get_logger(__name__)


@dataclass
class TelegramUser:
    """A Telegram user linked to a blockchain wallet."""
    telegram_id: int
    username: str = ""
    first_name: str = ""
    wallet_address: bytes = b""
    linked_at: float = 0.0
    total_sessions: int = 0
    total_earned: int = 0


class TelegramBridge:
    """
    Manages Telegram user <-> blockchain wallet mappings.

    Telegram Mini App developers register their bot token hash,
    then their players auto-get wallets when they first play.
    """

    def __init__(self):
        self._users: Dict[int, TelegramUser] = {}  # telegram_id -> TelegramUser
        self._bot_tokens: Dict[str, str] = {}  # game_id -> bot_token_hash (SHA-256)
        self._wallet_to_telegram: Dict[bytes, int] = {}  # wallet -> telegram_id

    def register_bot(self, game_id: str, bot_token_hash: str) -> bool:
        """Register a Telegram bot token hash for a game. Developer stores hash, not raw token."""
        if not game_id or not bot_token_hash or len(bot_token_hash) != 64:
            return False
        self._bot_tokens[game_id] = bot_token_hash
        return True

    def validate_init_data(self, game_id: str, init_data: str, bot_token: str) -> Optional[dict]:
        """
        Validate Telegram WebApp initData using HMAC-SHA256.
        Returns parsed user data if valid, None if invalid.

        The validation follows Telegram's official algorithm:
        1. Parse initData as query string
        2. Sort all params except 'hash' alphabetically
        3. Create data_check_string by joining "key=value" with newlines
        4. secret_key = HMAC-SHA256(bot_token, "WebAppData")
        5. Compare HMAC-SHA256(data_check_string, secret_key) with hash
        """
        try:
            parsed = parse_qs(init_data, keep_blank_values=True)
            received_hash = parsed.get('hash', [''])[0]
            if not received_hash:
                return None

            # Build data check string (all params except hash, sorted)
            check_pairs = []
            for key in sorted(parsed.keys()):
                if key == 'hash':
                    continue
                check_pairs.append(f"{key}={parsed[key][0]}")
            data_check_string = "\n".join(check_pairs)

            # HMAC validation
            secret_key = hmac.new(
                b"WebAppData", bot_token.encode(), hashlib.sha256
            ).digest()
            computed_hash = hmac.new(
                secret_key, data_check_string.encode(), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(computed_hash, received_hash):
                return None

            # Check auth_date freshness (max 24 hours)
            auth_date = int(parsed.get('auth_date', ['0'])[0])
            if time.time() - auth_date > 86400:
                return None

            # Parse user JSON
            import json
            user_str = parsed.get('user', ['{}'])[0]
            user_data = json.loads(unquote(user_str))
            return user_data
        except Exception as e:
            logger.debug("Telegram init data validation failed: %s", e)
            return None

    def get_or_create_wallet(self, telegram_id: int, username: str = "", first_name: str = "") -> Tuple[bytes, bool]:
        """
        Get existing wallet or create new one for a Telegram user.
        Returns (wallet_address, is_new).
        """
        if telegram_id in self._users:
            user = self._users[telegram_id]
            return user.wallet_address, False

        # Generate new keypair for this user
        kp = KeyPair()
        user = TelegramUser(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            wallet_address=kp.address,
            linked_at=time.time(),
        )
        self._users[telegram_id] = user
        self._wallet_to_telegram[kp.address] = telegram_id
        return kp.address, True

    def link_wallet(self, telegram_id: int, wallet_address: bytes, username: str = "") -> bool:
        """Link an existing wallet address to a Telegram user."""
        if len(wallet_address) != 20:
            return False
        if telegram_id in self._users:
            # Update existing
            self._users[telegram_id].wallet_address = wallet_address
        else:
            user = TelegramUser(
                telegram_id=telegram_id,
                username=username,
                wallet_address=wallet_address,
                linked_at=time.time(),
            )
            self._users[telegram_id] = user
        self._wallet_to_telegram[wallet_address] = telegram_id
        return True

    def get_wallet(self, telegram_id: int) -> Optional[bytes]:
        """Get wallet address for a Telegram user."""
        user = self._users.get(telegram_id)
        return user.wallet_address if user else None

    def get_user(self, telegram_id: int) -> Optional[dict]:
        """Get full user info."""
        user = self._users.get(telegram_id)
        if not user:
            return None
        return {
            "telegram_id": user.telegram_id,
            "username": user.username,
            "first_name": user.first_name,
            "wallet_address": "0x" + user.wallet_address.hex(),
            "linked_at": user.linked_at,
            "total_sessions": user.total_sessions,
            "total_earned": user.total_earned,
        }

    def get_stats(self) -> dict:
        """Get Telegram bridge statistics."""
        return {
            "total_users": len(self._users),
            "total_bots": len(self._bot_tokens),
            "total_wallets": len(self._wallet_to_telegram),
        }

    def record_session(self, telegram_id: int, reward: int):
        """Record a completed game session for stats."""
        user = self._users.get(telegram_id)
        if user:
            user.total_sessions += 1
            user.total_earned += reward
