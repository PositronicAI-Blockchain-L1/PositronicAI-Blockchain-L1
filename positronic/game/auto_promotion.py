"""
Positronic - Play-to-Mine Auto-Promotion Manager
Manages the player progression path: Player → Miner → Node → NVN.

Players earn ASF by playing, and when they reach MIN_STAKE (32 ASF),
they can opt in to become a validator node and NVN (Neural Validator Node),
joining the neural network and earning staking rewards + NVN fee shares.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

from positronic.utils.logging import get_logger

logger = get_logger(__name__)

from positronic.constants import MIN_STAKE, BASE_UNIT

if TYPE_CHECKING:
    from positronic.game.play_to_earn import PlayToEarnEngine
    from positronic.consensus.staking import StakingManager
    from positronic.consensus.validator import ValidatorRegistry
    from positronic.consensus.neural_validator import NVNRegistry
    from positronic.core.state import StateManager


# Promotion thresholds
MINER_THRESHOLD = 4 * BASE_UNIT    # 4 ASF → status "miner"
NODE_THRESHOLD = MIN_STAKE          # 32 ASF → auto-stake + validator + NVN


@dataclass
class PromotionEvent:
    """Record of a promotion event."""
    address: bytes
    old_status: str
    new_status: str
    balance: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "old_status": self.old_status,
            "new_status": self.new_status,
            "balance": self.balance,
            "timestamp": self.timestamp,
        }


class PlayerPromotionManager:
    """
    Manages the Play-to-Mine promotion pipeline.

    Promotion Flow:
        1. player  — Anyone playing the game, earning small rewards
        2. miner   — Reached 4+ ASF, halfway to node threshold
        3. node    — Reached 32 ASF, auto-staked, registered as validator
        4. nvn     — Registered as NVN, scoring transactions with AI

    Requirements for node/nvn promotion:
        - Balance >= MIN_STAKE (32 ASF)
        - Player has opted in via opt_in_auto_promotion(address, pubkey)
        - Public key provided for validator registration
    """

    def __init__(
        self,
        game_engine: PlayToEarnEngine,
        staking_manager: Optional[StakingManager] = None,
        validator_registry: Optional[ValidatorRegistry] = None,
        nvn_registry: Optional[NVNRegistry] = None,
        state: Optional[StateManager] = None,
        blockchain=None,
    ):
        self._game = game_engine
        self._staking = staking_manager
        self._validators = validator_registry
        self._nvn = nvn_registry
        self._state = state
        self._blockchain = blockchain  # for recording system TXs on-chain
        self._promotion_history: List[PromotionEvent] = []
        self._promoted_addresses: Dict[bytes, str] = {}  # address → status

    def check_and_promote(self, address: bytes) -> Optional[PromotionEvent]:
        """Check a player's balance and promote if eligible.

        Returns a PromotionEvent if promotion occurred, None otherwise.
        """
        player = self._game.get_player(address)
        if player is None:
            return None

        # Get balance from state if available, else from player profile
        balance = player.total_balance
        if self._state is not None:
            try:
                account = self._state.get_account(address)
                balance = account.effective_balance
            except Exception as e:
                logger.debug("Failed to get account balance for promotion check: %s", e)

        old_status = player.promotion_status

        # Check promotion thresholds
        if balance >= NODE_THRESHOLD and player.opted_in and not player.auto_staked:
            # Promote to node + NVN
            event = self._promote_to_node(address, player.node_pubkey, balance)
            if event:
                return event

        if balance >= MINER_THRESHOLD and old_status == "player":
            return self._promote_to_miner(address, balance)

        return None

    def _promote_to_miner(self, address: bytes, balance: int) -> PromotionEvent:
        """Promote player to miner status (informational only)."""
        player = self._game.get_player(address)
        old_status = player.promotion_status
        player.promotion_status = "miner"
        player.total_balance = balance

        event = PromotionEvent(
            address=address,
            old_status=old_status,
            new_status="miner",
            balance=balance,
        )
        self._promotion_history.append(event)
        self._promoted_addresses[address] = "miner"
        return event

    def _promote_to_node(self, address: bytes, pubkey: bytes,
                         balance: int) -> Optional[PromotionEvent]:
        """Promote player to node (validator + NVN).

        Steps:
        1. Auto-stake MIN_STAKE ASF
        2. Register as validator
        3. Register as NVN
        4. Update player status
        """
        player = self._game.get_player(address)
        if player is None:
            return None

        if not pubkey or len(pubkey) < 32:
            return None

        old_status = player.promotion_status

        # Step 1: Register as validator (includes staking)
        if self._staking is not None and self._validators is not None:
            try:
                # Check if already registered
                existing = self._validators.get(address)
                if existing is None:
                    self._staking.create_validator(
                        pubkey=pubkey,
                        stake=MIN_STAKE,
                        commission_rate=0.05,
                        is_nvn=True,
                    )

                    # Deduct staked amount from balance in state
                    if self._state is not None:
                        self._state.stake(address, MIN_STAKE)

                    # Record on-chain so other nodes see the auto-stake
                    if self._blockchain is not None and hasattr(self._blockchain, '_create_system_tx'):
                        self._blockchain._create_system_tx(
                            3, address, address, MIN_STAKE,
                            b"auto_promotion_stake",
                        )
            except (ValueError, Exception):
                # Staking failed — not enough balance or other issue
                return None

        # Step 2: Register as NVN
        if self._nvn is not None:
            try:
                existing_nvn = self._nvn.nvns.get(address)
                if existing_nvn is None:
                    self._nvn.register(address)
            except Exception as e:
                logger.debug("NVN registration best-effort failed: %s", e)

        # Step 3: Update player profile
        self._game.mark_staked(address)
        player.promotion_status = "nvn"

        event = PromotionEvent(
            address=address,
            old_status=old_status,
            new_status="nvn",
            balance=balance,
        )
        self._promotion_history.append(event)
        self._promoted_addresses[address] = "nvn"
        return event

    def batch_check_promotions(self) -> List[PromotionEvent]:
        """Check all players for promotion eligibility.

        Called during block creation to batch-process promotions.
        Returns list of promotion events that occurred.
        """
        events = []
        for address in list(self._game._players.keys()):
            event = self.check_and_promote(address)
            if event:
                events.append(event)
        return events

    def get_promotion_status(self, address: bytes) -> dict:
        """Get the promotion status for a player."""
        player = self._game.get_player(address)
        if player is None:
            return {
                "address": address.hex(),
                "status": "unknown",
                "opted_in": False,
                "eligible_for_node": False,
            }

        balance = player.total_balance
        if self._state is not None:
            try:
                account = self._state.get_account(address)
                balance = account.effective_balance
            except Exception as e:
                logger.debug("Failed to get account balance for status: %s", e)

        return {
            "address": address.hex(),
            "status": player.promotion_status,
            "opted_in": player.opted_in,
            "auto_staked": player.auto_staked,
            "balance": balance,
            "threshold_node": NODE_THRESHOLD,
            "threshold_miner": MINER_THRESHOLD,
            "eligible_for_node": balance >= NODE_THRESHOLD and player.opted_in,
            "progress_percent": min(100.0, (balance / NODE_THRESHOLD) * 100),
        }

    def get_stats(self) -> dict:
        """Get overall Play-to-Mine statistics."""
        status_counts = {"player": 0, "miner": 0, "node": 0, "nvn": 0}
        for addr, status in self._promoted_addresses.items():
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count non-promoted players
        total_players = self._game.total_players
        promoted = sum(status_counts.values())
        status_counts["player"] = max(0, total_players - promoted)

        return {
            "total_players": total_players,
            "total_promotions": len(self._promotion_history),
            "players_by_status": status_counts,
            "recent_promotions": [
                e.to_dict() for e in self._promotion_history[-10:]
            ],
        }

    @property
    def promotion_history(self) -> List[PromotionEvent]:
        return list(self._promotion_history)
