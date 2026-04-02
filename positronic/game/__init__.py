"""
Positronic - Game Module
Play-to-Mine: Play games, earn coins, become a node operator.
Game Bridge: Connect external games to the blockchain.
"""

from positronic.game.play_to_earn import (
    PlayToEarnEngine,
    GameResult,
    GameProof,
    PlayerProfile,
    GameAchievement,
)
from positronic.game.auto_promotion import PlayerPromotionManager, PromotionEvent
from positronic.game.game_registry import GameRegistry, GameInfo, GameType, GameStatus
from positronic.game.mining_controller import MiningRateController
from positronic.game.session_validator import (
    GameSessionValidator, GameSession, GameEvent, ProofType, SessionStatus,
)
from positronic.game.bridge_api import GameBridgeAPI

__all__ = [
    "PlayToEarnEngine",
    "GameResult",
    "GameProof",
    "PlayerProfile",
    "GameAchievement",
    "PlayerPromotionManager",
    "PromotionEvent",
    "GameRegistry",
    "GameInfo",
    "GameType",
    "GameStatus",
    "MiningRateController",
    "GameSessionValidator",
    "GameSession",
    "GameEvent",
    "ProofType",
    "SessionStatus",
    "GameBridgeAPI",
]
