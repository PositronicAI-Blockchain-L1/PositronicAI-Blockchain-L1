"""
Positronic - Mining Rate Controller
4-layer control system for Play-to-Mine token emission from external games.
Prevents inflation while rewarding legitimate gameplay.
"""

import time
from typing import Dict, Optional

from positronic.constants import BASE_UNIT, GAME_EMISSION_TAX, PLAY_MINING_SUPPLY_CAP


# ---- Emission Constants ----
EXTERNAL_GAME_SUPPLY_CAP = 100_000_000 * BASE_UNIT  # 100M ASF for external games
ONCHAIN_GAME_SUPPLY_CAP = 100_000_000 * BASE_UNIT   # 100M ASF for on-chain games
# Together = 200M PLAY_MINING_SUPPLY_CAP

DEFAULT_GAME_DAILY_CAP = 10_000 * BASE_UNIT    # 10K ASF/day per game
MAX_GAME_DAILY_CAP = 100_000 * BASE_UNIT       # 100K ASF/day max per game
PLAYER_DAILY_CAP = 50 * BASE_UNIT              # 50 ASF/day per player per game
PLAYER_GLOBAL_DAILY_CAP = 200 * BASE_UNIT      # 200 ASF/day per player across ALL games

SESSION_MIN_DURATION = 60          # Minimum 60 seconds to earn
SESSION_MAX_REWARD = 10 * BASE_UNIT  # Max 10 ASF per session

DAILY_RESET_SECONDS = 86_400       # 24 hours


class MiningRateController:
    """
    4-layer emission control:
    1. Global cap — 100M ASF total for external games
    2. Per-game cap — 10K-100K ASF/day (based on trust)
    3. Per-player cap — 50 ASF/day per game, 200 ASF/day global
    4. Per-session — dynamic based on duration, skill, scarcity
    """

    def __init__(self):
        self.total_external_emitted: int = 0
        self._player_daily: Dict[str, _PlayerDailyTracker] = {}
        self._daily_reset_time: float = time.time()

    # ---- Layer 1: Global Cap ----

    @property
    def remaining_supply(self) -> int:
        return max(0, EXTERNAL_GAME_SUPPLY_CAP - self.total_external_emitted)

    @property
    def supply_exhausted(self) -> bool:
        return self.total_external_emitted >= EXTERNAL_GAME_SUPPLY_CAP

    @property
    def scarcity_factor(self) -> float:
        """Reduces rewards as supply depletes (1.0 → 0.1)."""
        if EXTERNAL_GAME_SUPPLY_CAP == 0:
            return 0.0
        pct_remaining = self.remaining_supply / EXTERNAL_GAME_SUPPLY_CAP
        if pct_remaining > 0.5:
            return 1.0
        if pct_remaining > 0.25:
            return 0.75
        if pct_remaining > 0.10:
            return 0.5
        if pct_remaining > 0.01:
            return 0.25
        return 0.1

    # ---- Layer 2: Per-Game Cap ----

    def check_game_daily_cap(self, game) -> bool:
        """Check if game has remaining daily emission budget."""
        self._maybe_reset_game(game)
        return game.daily_emitted < game.daily_emission_cap

    def get_game_remaining_daily(self, game) -> int:
        """How much ASF this game can still emit today."""
        self._maybe_reset_game(game)
        return max(0, game.daily_emission_cap - game.daily_emitted)

    def _maybe_reset_game(self, game):
        now = time.time()
        if now - game.daily_reset_time >= DAILY_RESET_SECONDS:
            game.daily_emitted = 0
            game.daily_reset_time = now

    # ---- Layer 3: Per-Player Cap ----

    def check_player_cap(self, player_hex: str, game_id: str) -> bool:
        """Check if player has remaining daily budget."""
        self._maybe_reset_players()
        tracker = self._player_daily.get(player_hex)
        if not tracker:
            return True
        # Check per-game cap
        game_earned = tracker.per_game.get(game_id, 0)
        if game_earned >= PLAYER_DAILY_CAP:
            return False
        # Check global cap
        if tracker.total >= PLAYER_GLOBAL_DAILY_CAP:
            return False
        return True

    def get_player_remaining(self, player_hex: str, game_id: str) -> int:
        """How much ASF player can still earn today in this game."""
        self._maybe_reset_players()
        tracker = self._player_daily.get(player_hex)
        if not tracker:
            return min(PLAYER_DAILY_CAP, PLAYER_GLOBAL_DAILY_CAP)

        game_remaining = PLAYER_DAILY_CAP - tracker.per_game.get(game_id, 0)
        global_remaining = PLAYER_GLOBAL_DAILY_CAP - tracker.total
        return max(0, min(game_remaining, global_remaining))

    def _maybe_reset_players(self):
        now = time.time()
        if now - self._daily_reset_time >= DAILY_RESET_SECONDS:
            self._player_daily.clear()
            self._daily_reset_time = now

    # ---- Layer 4: Session Reward Calculation ----

    def calculate_reward(
        self,
        game,
        player_hex: str,
        duration: float,
        score: int,
        skill_factor: float = 1.0,
        trust_multiplier: float = 1.0,
    ) -> int:
        """
        Calculate mining reward for a game session.

        reward = base × game_mult × trust_mult × skill × scarcity
        Then clamped by all 4 layers.
        """
        if self.supply_exhausted:
            return 0

        if duration < SESSION_MIN_DURATION:
            return 0

        # Base reward: 0.1 ASF per 10-minute session, scaling with time
        minutes = min(duration / 60, 60)  # Cap at 60 minutes
        base = int(0.1 * BASE_UNIT * (minutes / 10))

        # Score bonus (up to 3x for high scores)
        score_mult = min(1.0 + (score / 10_000), 3.0)

        # Apply multipliers
        game_mult = max(0.1, min(game.reward_multiplier, 2.0))
        trust_mult = max(1.0, min(trust_multiplier, 2.0))
        skill_f = max(0.5, min(skill_factor, 3.0))
        scarcity = self.scarcity_factor

        reward = int(base * score_mult * game_mult * trust_mult * skill_f * scarcity)

        # Clamp to session max
        reward = min(reward, SESSION_MAX_REWARD)

        # Clamp to player remaining
        player_remaining = self.get_player_remaining(player_hex, game.game_id)
        reward = min(reward, player_remaining)

        # Clamp to game remaining
        game_remaining = self.get_game_remaining_daily(game)
        reward = min(reward, game_remaining)

        # Clamp to global remaining
        reward = min(reward, self.remaining_supply)

        return max(0, reward)

    # ---- Record Emission ----

    def record_emission(self, game, player_hex: str, amount: int) -> tuple:
        """Record that ASF was emitted to a player from a game.

        Applies GAME_EMISSION_TAX (5%) split:
          - (1 - tax) goes to the player
          - tax goes to TEAM_ADDRESS (caller handles the transfer)

        Returns (player_amount, team_amount).
        """
        if amount <= 0:
            return (0, 0)

        team_amount = int(amount * GAME_EMISSION_TAX)
        player_amount = amount - team_amount

        # Update global (track gross amount)
        self.total_external_emitted += amount

        # Update game (track gross amount)
        game.total_emitted += amount
        game.daily_emitted += amount

        # Update player tracker
        self._maybe_reset_players()
        if player_hex not in self._player_daily:
            self._player_daily[player_hex] = _PlayerDailyTracker()
        tracker = self._player_daily[player_hex]
        tracker.total += amount
        tracker.per_game[game.game_id] = tracker.per_game.get(game.game_id, 0) + amount

        return (player_amount, team_amount)

    # ---- Stats ----

    def get_stats(self) -> dict:
        return {
            "total_external_emitted": self.total_external_emitted,
            "external_supply_cap": EXTERNAL_GAME_SUPPLY_CAP,
            "remaining_supply": self.remaining_supply,
            "scarcity_factor": self.scarcity_factor,
            "supply_pct_used": round(
                self.total_external_emitted / EXTERNAL_GAME_SUPPLY_CAP * 100, 2
            ) if EXTERNAL_GAME_SUPPLY_CAP else 0,
            "active_players_today": len(self._player_daily),
            "player_daily_cap": PLAYER_DAILY_CAP,
            "player_global_daily_cap": PLAYER_GLOBAL_DAILY_CAP,
        }


class _PlayerDailyTracker:
    """Tracks a player's daily earnings across games."""
    __slots__ = ("total", "per_game")

    def __init__(self):
        self.total: int = 0
        self.per_game: Dict[str, int] = {}
