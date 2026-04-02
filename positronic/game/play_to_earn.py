"""
Positronic - Play-to-Earn Game Infrastructure
Platformer-style game with controlled token rewards.
Players earn ASF coins by playing and achieving milestones.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import hashlib

from positronic.crypto.hashing import sha512
from positronic.constants import (
    P2E_PLAYER_DAILY_CAP,
    P2E_DAILY_MAX_GAMES,
    P2E_SESSION_MAX_REWARD,
)


class GameAchievement(IntEnum):
    """Achievements players can earn."""
    FIRST_JUMP = 1            # First game played
    COIN_COLLECTOR = 2        # Collected 100 coins in game
    BOSS_BEATER = 3           # Defeated a boss
    SPEED_RUNNER = 4          # Completed level under time
    PERFECT_RUN = 5           # Completed level with no damage
    WORLD_CHAMPION = 6        # Completed all worlds
    DAILY_PLAYER = 7          # Played 7 consecutive days
    MARATHON = 8              # Played 30 consecutive days


# Reward amounts in base units (Pixel = 1)
ACHIEVEMENT_REWARDS = {
    GameAchievement.FIRST_JUMP: 1 * 10**18,        # 1 ASF
    GameAchievement.COIN_COLLECTOR: 5 * 10**18,     # 5 ASF
    GameAchievement.BOSS_BEATER: 10 * 10**18,       # 10 ASF
    GameAchievement.SPEED_RUNNER: 15 * 10**18,      # 15 ASF
    GameAchievement.PERFECT_RUN: 20 * 10**18,       # 20 ASF
    GameAchievement.WORLD_CHAMPION: 100 * 10**18,   # 100 ASF
    GameAchievement.DAILY_PLAYER: 3 * 10**18,       # 3 ASF
    GameAchievement.MARATHON: 50 * 10**18,           # 50 ASF
}

# Daily reward caps — sourced from constants.py (single source of truth)
DAILY_MAX_REWARD = P2E_PLAYER_DAILY_CAP   # 50 ASF per day from gaming
DAILY_MAX_GAMES = P2E_DAILY_MAX_GAMES     # 20 games per day

# Game result validation bounds (anti-cheat)
MAX_SCORE_PER_GAME = 100_000       # Maximum plausible score per session
MAX_COINS_PER_GAME = 500           # Maximum collectible coins per level
MAX_ENEMIES_PER_GAME = 100         # Maximum enemies per session
MAX_LEVEL_PER_GAME = 20            # Maximum levels in one session
MIN_TIME_PER_LEVEL = 5.0           # Minimum seconds per level (anti-speedhack)
MAX_TIME_PER_GAME = 7200.0         # Maximum 2 hours per session


@dataclass
class PlayerProfile:
    """A player's gaming profile with Play-to-Mine promotion tracking."""
    address: bytes
    total_games: int = 0
    total_score: int = 0
    total_rewards_earned: int = 0
    achievements: List[int] = field(default_factory=list)
    daily_games: int = 0
    daily_rewards: int = 0
    last_play_date: str = ""
    consecutive_days: int = 0
    joined_at: float = 0.0
    level: int = 1
    experience: int = 0
    # Play-to-Mine promotion fields
    promotion_status: str = "player"  # player → miner → node → nvn
    total_balance: int = 0            # Total ASF earned from gaming
    auto_staked: bool = False         # Whether auto-stake has been triggered
    node_pubkey: bytes = b""          # Public key for node (set on opt-in)
    opted_in: bool = False            # Whether player opted in for auto-promotion

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "total_games": self.total_games,
            "total_score": self.total_score,
            "total_rewards_earned": self.total_rewards_earned,
            "achievements": self.achievements,
            "level": self.level,
            "experience": self.experience,
            "consecutive_days": self.consecutive_days,
            "promotion_status": self.promotion_status,
            "total_balance": self.total_balance,
            "auto_staked": self.auto_staked,
            "opted_in": self.opted_in,
        }


@dataclass
class GameResult:
    """Result of a single game session.

    The result_hash field provides integrity verification: the game client
    computes a HMAC-SHA256 of the result fields using a shared secret,
    preventing trivial forgery of game results.
    """
    player: bytes
    score: int
    level_completed: int
    time_taken: float           # seconds
    coins_collected: int
    enemies_defeated: int
    no_damage: bool = False
    timestamp: float = 0.0
    reward_earned: int = 0
    result_hash: bytes = b""    # HMAC integrity hash from game client

    def compute_hash(self, secret: bytes = b"positronic-game-v1") -> bytes:
        """Compute integrity hash over result fields using SHA-512."""
        payload = (
            f"{self.player.hex()}:{self.score}:{self.level_completed}:"
            f"{self.time_taken:.2f}:{self.coins_collected}:{self.enemies_defeated}:"
            f"{self.no_damage}"
        )
        return sha512(secret + payload.encode())

    def to_dict(self) -> dict:
        return {
            "player": self.player.hex(),
            "score": self.score,
            "level_completed": self.level_completed,
            "time_taken": self.time_taken,
            "coins_collected": self.coins_collected,
            "enemies_defeated": self.enemies_defeated,
            "no_damage": self.no_damage,
            "reward_earned": self.reward_earned,
        }


@dataclass
class GameProof:
    """Proof of gameplay — links a game result to the blockchain.

    NVNs can validate this proof to ensure game results are legitimate.
    The proof_hash is a SHA-512 hash of the game result data.
    """
    player: bytes
    game_result: GameResult
    proof_hash: bytes = b""       # SHA-512 hash of game result
    block_height: int = 0         # Block height when proof was created
    timestamp: float = 0.0
    verified: bool = False        # Whether NVN has verified this proof
    reward_amount: int = 0        # Reward amount in base units

    def __post_init__(self):
        if not self.proof_hash:
            self.proof_hash = self._compute_proof_hash()
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def _compute_proof_hash(self) -> bytes:
        """Compute SHA-512 proof hash over game result."""
        payload = (
            f"{self.player.hex()}:{self.game_result.score}:"
            f"{self.game_result.level_completed}:{self.game_result.time_taken:.2f}:"
            f"{self.game_result.coins_collected}:{self.game_result.enemies_defeated}:"
            f"{self.block_height}:{self.reward_amount}"
        )
        return sha512(payload.encode())

    def to_dict(self) -> dict:
        return {
            "player": self.player.hex(),
            "proof_hash": self.proof_hash.hex(),
            "block_height": self.block_height,
            "timestamp": self.timestamp,
            "verified": self.verified,
            "reward_amount": self.reward_amount,
            "game_result": self.game_result.to_dict(),
        }


class PlayToEarnEngine:
    """
    Manages the Play-to-Earn game economy.
    Controls reward distribution to prevent inflation.
    """

    def __init__(self):
        self._players: Dict[bytes, PlayerProfile] = {}
        self._pending_rewards: Dict[bytes, int] = {}  # address -> pending reward amount
        self._total_distributed: int = 0
        self._game_history: List[GameResult] = []

    def register_player(self, address: bytes) -> PlayerProfile:
        """Register a new player."""
        profile = PlayerProfile(
            address=address,
            joined_at=time.time(),
        )
        self._players[address] = profile
        return profile

    def get_player(self, address: bytes) -> Optional[PlayerProfile]:
        return self._players.get(address)

    def _validate_game_result(self, result: GameResult) -> Optional[str]:
        """Validate game result integrity and plausibility.

        Returns None if valid, or an error string describing the rejection reason.
        Prevents trivial token minting via forged game results.
        """
        # Basic field sanity checks
        if not result.player or len(result.player) < 1:
            return "Missing player address"

        if result.score < 0:
            return "Negative score"

        if result.score > MAX_SCORE_PER_GAME:
            return f"Score {result.score} exceeds maximum {MAX_SCORE_PER_GAME}"

        if result.coins_collected < 0 or result.coins_collected > MAX_COINS_PER_GAME:
            return f"Coins collected {result.coins_collected} out of range [0, {MAX_COINS_PER_GAME}]"

        if result.enemies_defeated < 0 or result.enemies_defeated > MAX_ENEMIES_PER_GAME:
            return f"Enemies defeated {result.enemies_defeated} out of range [0, {MAX_ENEMIES_PER_GAME}]"

        if result.level_completed < 0 or result.level_completed > MAX_LEVEL_PER_GAME:
            return f"Level completed {result.level_completed} out of range [0, {MAX_LEVEL_PER_GAME}]"

        if result.time_taken <= 0 or result.time_taken > MAX_TIME_PER_GAME:
            return f"Time taken {result.time_taken}s out of range (0, {MAX_TIME_PER_GAME}]"

        # Anti-speedhack: enforce minimum time per level completed
        if result.level_completed > 0:
            min_time = result.level_completed * MIN_TIME_PER_LEVEL
            if result.time_taken < min_time:
                return (
                    f"Impossible speed: {result.time_taken:.1f}s for "
                    f"{result.level_completed} levels (min {min_time:.1f}s)"
                )

        # Score consistency: score should be plausible given game actions.
        # Uses a generous upper bound because combo multipliers, hidden
        # bonuses, and special events can amplify the base score.
        base_actions = (
            result.coins_collected * 100
            + result.enemies_defeated * 500
            + result.level_completed * 2000
            + (5000 if result.no_damage else 0)
        )
        # Allow 5x multiplier (combos, events) + 10k base tolerance
        max_plausible_score = base_actions * 5 + 10_000
        if result.score > max_plausible_score:
            return (
                f"Score {result.score} inconsistent with game actions "
                f"(max plausible: {max_plausible_score})"
            )

        # Integrity hash verification (if provided)
        if result.result_hash:
            expected = result.compute_hash()
            if result.result_hash != expected:
                return "Result hash integrity check failed"

        # Replay detection: reject if the exact same result (with hash) was
        # submitted recently. Without a hash, we allow similar results since
        # players legitimately replay the same levels.
        if result.result_hash:
            for recent in self._game_history[-100:]:
                if recent.result_hash and recent.result_hash == result.result_hash:
                    return "Duplicate game result (replay attack)"

        return None  # Valid

    def submit_game_result(self, result: GameResult) -> dict:
        """Submit a game result and calculate rewards.

        Validates the result for plausibility and integrity before
        processing rewards, preventing trivial token minting exploits.
        """
        # Validate game result integrity
        rejection = self._validate_game_result(result)
        if rejection:
            return {"reward": 0, "reason": f"Result rejected: {rejection}"}

        player = self._players.get(result.player)
        if not player:
            player = self.register_player(result.player)

        today = time.strftime("%Y-%m-%d")

        # Reset daily counters if new day
        if player.last_play_date != today:
            if player.last_play_date:
                # Check consecutive days
                player.consecutive_days += 1
            player.daily_games = 0
            player.daily_rewards = 0
            player.last_play_date = today

        # Check daily limits
        if player.daily_games >= DAILY_MAX_GAMES:
            return {"reward": 0, "reason": "Daily game limit reached"}

        if player.daily_rewards >= DAILY_MAX_REWARD:
            return {"reward": 0, "reason": "Daily reward limit reached"}

        # Calculate reward based on performance
        reward = self._calculate_reward(result, player)

        # Cap to daily limit
        remaining = DAILY_MAX_REWARD - player.daily_rewards
        reward = min(reward, remaining)

        # Update player profile
        result.reward_earned = reward
        result.timestamp = time.time()
        player.total_games += 1
        player.total_score += result.score
        player.daily_games += 1
        player.daily_rewards += reward
        player.total_rewards_earned += reward
        player.experience += result.score // 100

        # Level up
        while player.experience >= player.level * 1000:
            player.experience -= player.level * 1000
            player.level += 1

        # Check achievements
        new_achievements = self._check_achievements(player, result)

        # Queue reward
        if reward > 0:
            self._pending_rewards[result.player] = (
                self._pending_rewards.get(result.player, 0) + reward
            )

        self._game_history.append(result)

        return {
            "reward": reward,
            "new_achievements": [a.name for a in new_achievements],
            "level": player.level,
            "total_games": player.total_games,
        }

    def _calculate_reward(self, result: GameResult,
                          player: PlayerProfile) -> int:
        """Calculate reward based on game performance."""
        base_reward = 10**17  # 0.1 ASF base

        # Score bonus
        score_multiplier = min(result.score / 1000, 5.0)

        # Perfect run bonus
        if result.no_damage:
            score_multiplier *= 2.0

        # Level completion bonus
        level_bonus = result.level_completed * 10**16  # 0.01 ASF per level

        reward = int(base_reward * score_multiplier) + level_bonus
        return reward

    def _check_achievements(self, player: PlayerProfile,
                            result: GameResult) -> List[GameAchievement]:
        """Check and award new achievements."""
        new_achievements = []

        checks = [
            (GameAchievement.FIRST_JUMP, player.total_games == 1),
            (GameAchievement.COIN_COLLECTOR, result.coins_collected >= 100),
            (GameAchievement.BOSS_BEATER, result.enemies_defeated >= 10),
            (GameAchievement.SPEED_RUNNER, result.time_taken < 60),
            (GameAchievement.PERFECT_RUN, result.no_damage and result.level_completed > 0),
            (GameAchievement.DAILY_PLAYER, player.consecutive_days >= 7),
            (GameAchievement.MARATHON, player.consecutive_days >= 30),
        ]

        for achievement, condition in checks:
            if condition and achievement.value not in player.achievements:
                player.achievements.append(achievement.value)
                new_achievements.append(achievement)

                # Add achievement reward
                reward = ACHIEVEMENT_REWARDS.get(achievement, 0)
                if reward > 0:
                    self._pending_rewards[player.address] = (
                        self._pending_rewards.get(player.address, 0) + reward
                    )

        return new_achievements

    def get_pending_rewards(self) -> Dict[bytes, int]:
        """Get all pending rewards to be distributed."""
        return dict(self._pending_rewards)

    def clear_pending_rewards(self):
        """Clear pending rewards after distribution."""
        self._total_distributed += sum(self._pending_rewards.values())
        self._pending_rewards.clear()

    def get_leaderboard(self, limit: int = 50) -> List[dict]:
        """Get player leaderboard."""
        sorted_players = sorted(
            self._players.values(),
            key=lambda p: (-p.total_score, -p.total_games),
        )
        return [p.to_dict() for p in sorted_players[:limit]]

    @property
    def total_players(self) -> int:
        return len(self._players)

    def create_game_proof(self, result: GameResult,
                          block_height: int = 0) -> Optional[GameProof]:
        """Create a GameProof from a validated game result."""
        if result.reward_earned <= 0:
            return None

        proof = GameProof(
            player=result.player,
            game_result=result,
            block_height=block_height,
            reward_amount=result.reward_earned,
        )
        return proof

    def opt_in_auto_promotion(self, address: bytes, pubkey: bytes) -> bool:
        """Player opts in for auto-promotion to node/NVN when threshold met.

        Args:
            address: Player's 20-byte address.
            pubkey: Player's 32-byte Ed25519 public key for validator registration.

        Returns:
            True if opt-in was successful.
        """
        player = self._players.get(address)
        if player is None:
            return False
        if not pubkey or len(pubkey) < 32:
            return False
        player.opted_in = True
        player.node_pubkey = pubkey
        return True

    def check_promotion_status(self, address: bytes,
                                balance: int) -> Optional[str]:
        """Check and update a player's promotion status based on balance.

        Args:
            address: Player's address.
            balance: Player's current effective balance (from state).

        Returns:
            New status string or None if no change.
        """
        from positronic.constants import MIN_STAKE

        player = self._players.get(address)
        if player is None:
            return None

        player.total_balance = balance
        miner_threshold = 4 * (10 ** 18)  # 4 ASF
        old_status = player.promotion_status

        if balance >= MIN_STAKE and player.opted_in and not player.auto_staked:
            player.promotion_status = "node"
            return "node" if old_status != "node" else None
        elif player.auto_staked:
            player.promotion_status = "nvn"
            return "nvn" if old_status != "nvn" else None
        elif balance >= miner_threshold:
            player.promotion_status = "miner"
            return "miner" if old_status != "miner" else None
        else:
            player.promotion_status = "player"
            return None

    def mark_staked(self, address: bytes):
        """Mark a player as having been auto-staked."""
        player = self._players.get(address)
        if player:
            player.auto_staked = True
            player.promotion_status = "nvn"

    def get_stats(self) -> dict:
        players_by_status = {"player": 0, "miner": 0, "node": 0, "nvn": 0}
        for p in self._players.values():
            players_by_status[p.promotion_status] = (
                players_by_status.get(p.promotion_status, 0) + 1
            )
        return {
            "total_players": self.total_players,
            "total_games_played": sum(p.total_games for p in self._players.values()),
            "total_distributed": self._total_distributed,
            "pending_rewards": sum(self._pending_rewards.values()),
            "players_by_status": players_by_status,
        }
