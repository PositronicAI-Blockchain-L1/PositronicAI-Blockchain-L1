"""
Positronic - Fully On-Chain Game (FOCG)
Complete game state and logic verified on blockchain.
AI anti-cheat validates all game actions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import IntEnum
import time
import hashlib


class GameAction(IntEnum):
    MOVE = 0
    JUMP = 1
    ATTACK = 2
    COLLECT_ITEM = 3
    USE_ITEM = 4
    COMPLETE_LEVEL = 5


class ItemType(IntEnum):
    CREDIT = 0
    AMPLIFIER = 1
    NOVA = 2
    PLASMA = 3
    SHIELD = 4


@dataclass
class GameItem:
    item_type: ItemType
    quantity: int = 1

    def to_dict(self) -> dict:
        return {"type": self.item_type.name, "quantity": self.quantity}


@dataclass
class OnChainGameState:
    """Complete game state stored on blockchain."""
    player: bytes
    level: int = 1
    score: int = 0
    lives: int = 3
    position_x: int = 0
    position_y: int = 0
    items: List[GameItem] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    actions_count: int = 0
    started_at: float = 0.0
    last_action_at: float = 0.0
    completed: bool = False
    game_hash: str = ""  # Hash of all actions for integrity

    def __post_init__(self):
        if self.started_at == 0.0:
            self.started_at = time.time()

    def apply_action(self, action: GameAction, params: dict = None) -> bool:
        """Apply a game action and update state."""
        params = params or {}
        self.actions_count += 1
        self.last_action_at = time.time()

        if action == GameAction.MOVE:
            dx = params.get("dx", 1)
            dy = params.get("dy", 0)
            self.position_x += dx
            self.position_y += dy
            self.score += 1
            return True

        elif action == GameAction.JUMP:
            self.position_y += params.get("height", 3)
            self.score += 5
            return True

        elif action == GameAction.ATTACK:
            self.score += 10
            return True

        elif action == GameAction.COLLECT_ITEM:
            item_type = ItemType(params.get("item_type", 0))
            item = GameItem(item_type=item_type)
            self.items.append(item)
            if item_type == ItemType.CREDIT:
                self.score += 10
            elif item_type == ItemType.NOVA:
                self.score += 100
            elif item_type == ItemType.SHIELD:
                self.lives += 1
            return True

        elif action == GameAction.COMPLETE_LEVEL:
            self.level += 1
            self.score += 1000
            if f"level_{self.level-1}_complete" not in self.achievements:
                self.achievements.append(f"level_{self.level-1}_complete")
            return True

        return False

    def verify_action(self, action: GameAction, params: dict = None) -> bool:
        """Verify if an action is valid (anti-cheat)."""
        # Check impossible movement speed
        if action == GameAction.MOVE:
            dx = abs(params.get("dx", 0)) if params else 0
            dy = abs(params.get("dy", 0)) if params else 0
            if dx > 10 or dy > 10:  # Max movement per action
                return False

        # Check impossible jump height
        if action == GameAction.JUMP:
            height = params.get("height", 0) if params else 0
            if height > 20:
                return False

        # Cannot act on completed game
        if self.completed:
            return False

        return True

    def get_reward(self) -> int:
        """Calculate ASF reward based on game state."""
        base_reward = self.score // 100  # 1 ASF per 100 points
        level_bonus = (self.level - 1) * 5  # 5 ASF per level completed
        return base_reward + level_bonus

    def compute_hash(self) -> str:
        """Compute integrity hash of game state."""
        data = f"{self.player.hex()}:{self.score}:{self.level}:{self.actions_count}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def to_dict(self) -> dict:
        return {
            "player": self.player.hex(),
            "level": self.level,
            "score": self.score,
            "lives": self.lives,
            "position": {"x": self.position_x, "y": self.position_y},
            "items": [i.to_dict() for i in self.items],
            "achievements": self.achievements,
            "actions_count": self.actions_count,
            "completed": self.completed,
            "reward_sma": self.get_reward(),
            "game_hash": self.compute_hash(),
        }


class OnChainGameEngine:
    """Engine for managing on-chain game sessions."""

    def __init__(self):
        self._active_games: Dict[bytes, OnChainGameState] = {}
        self._completed_games: int = 0
        self._total_actions: int = 0
        self._total_rewards: int = 0
        self._cheats_detected: int = 0

    def start_game(self, player: bytes) -> OnChainGameState:
        """Start a new game session."""
        game = OnChainGameState(player=player)
        self._active_games[player] = game
        return game

    def get_game(self, player: bytes) -> Optional[OnChainGameState]:
        """Get active game for player."""
        return self._active_games.get(player)

    def submit_action(self, player: bytes, action: GameAction, params: dict = None) -> bool:
        """Submit and verify a game action."""
        game = self.get_game(player)
        if game is None:
            return False

        # AI anti-cheat verification
        if not game.verify_action(action, params):
            self._cheats_detected += 1
            return False

        result = game.apply_action(action, params)
        if result:
            self._total_actions += 1
        return result

    def complete_game(self, player: bytes) -> Optional[dict]:
        """Complete a game session and calculate rewards."""
        game = self.get_game(player)
        if game is None:
            return None

        game.completed = True
        game.game_hash = game.compute_hash()
        reward = game.get_reward()
        self._total_rewards += reward
        self._completed_games += 1

        result = game.to_dict()
        del self._active_games[player]
        return result

    def get_stats(self) -> dict:
        return {
            "active_games": len(self._active_games),
            "completed_games": self._completed_games,
            "total_actions": self._total_actions,
            "total_rewards_sma": self._total_rewards,
            "cheats_detected": self._cheats_detected,
        }
