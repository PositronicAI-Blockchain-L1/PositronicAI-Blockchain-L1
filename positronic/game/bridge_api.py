"""
Positronic - Game Bridge API
RPC method handlers connecting external games to the blockchain.
Routes API calls to GameRegistry, MiningRateController, and GameSessionValidator.
Persists session and emission data to SQLite via GameDatabase.
"""

import time
from typing import Any, Dict, List, Optional

from positronic.utils.logging import get_logger

logger = get_logger("game_bridge")

from positronic.game.game_registry import (
    GameRegistry, GameInfo, GameType, GameStatus,
)
from positronic.game.mining_controller import MiningRateController
from positronic.game.session_validator import (
    GameSessionValidator, GameEvent, GameSession, ProofType,
)
from positronic.constants import (
    GAME_API_RATE_LIMIT,
    GAME_API_RATE_WINDOW,
    GAME_TRUST_DECAY_BASE,
    GAME_TRUST_DECAY_EXPONENT,
)


class GameBridgeAPI:
    """
    Handles 15+ RPC methods for the external Game Bridge.

    Methods:
    - positronic_registerGame            — Register a new game
    - positronic_getGameInfo             — Get game details
    - positronic_listRegisteredGames     — List all active games
    - positronic_getGameBridgeStats      — Global bridge stats
    - positronic_startGameSession        — Start a play session
    - positronic_addGameEvent            — Add event during session
    - positronic_submitGameSession       — Submit session for validation
    - positronic_getSessionStatus        — Check session status
    - positronic_getPlayerGameHistory    — Player history in a game
    - positronic_getGameMiningRate       — Current mining rate for a game
    - positronic_getGameEmission         — Emission stats for a game
    - positronic_getGlobalPlayMineStats  — Global play-to-mine stats
    - positronic_generateGameAPIKey      — Regenerate game API key
    - positronic_testGameSession         — Test session (no real reward)
    - positronic_getGameSDKConfig        — SDK configuration for a game
    """

    def __init__(
        self,
        registry: Optional[GameRegistry] = None,
        mining_controller: Optional[MiningRateController] = None,
        session_validator: Optional[GameSessionValidator] = None,
        game_db=None,
        data_dir: Optional[str] = None,
    ):
        # If no registry provided, create one.  When data_dir is given,
        # pass db_path so GameRegistry persists to SQLite.
        if registry is not None:
            self.registry = registry
        elif data_dir is not None:
            import os
            self.registry = GameRegistry(db_path=os.path.join(data_dir, "game_registry.db"))
        else:
            self.registry = GameRegistry()
        self.mining = mining_controller or MiningRateController()
        self.validator = session_validator or GameSessionValidator()
        self._db = game_db  # Optional GameDatabase for persistence

        # Phase 15: API rate limiting
        self._rate_limits: Dict[str, List[float]] = {}  # api_key_hash → [timestamps]

        # On-chain reward settlement — flushed every N blocks by blockchain
        self._pending_rewards: Dict[bytes, int] = {}  # player_address → pending ASF

    # ---- Phase 15: Rate Limiting ----

    def _check_rate_limit(self, api_key_hash: str) -> bool:
        """Returns True if request is allowed, False if rate limited."""
        now = time.time()
        timestamps = self._rate_limits.get(api_key_hash, [])
        # Remove old entries outside window
        timestamps = [t for t in timestamps if now - t < GAME_API_RATE_WINDOW]
        if len(timestamps) >= GAME_API_RATE_LIMIT:
            return False
        timestamps.append(now)
        self._rate_limits[api_key_hash] = timestamps
        return True

    def _auth_and_rate_limit(self, api_key: str) -> tuple:
        """Authenticate API key and check rate limit. Returns (game, error)."""
        import hashlib
        game = self.registry.authenticate(api_key)
        if not game:
            return None, "Invalid API key or game not active"
        key_hash = hashlib.sha512(api_key.encode()).hexdigest()
        if not self._check_rate_limit(key_hash):
            return None, "Rate limit exceeded (max 100 req/min)"
        return game, None

    # ---- 1. positronic_registerGame ----

    def register_game(self, params: list) -> dict:
        """Register a new external game.

        params: [{developer, name, game_type, description?, daily_cap?}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing game registration data"}
        data = params[0]

        try:
            developer = bytes.fromhex(data["developer"])
            name = str(data["name"])
            game_type = GameType[data.get("game_type", "SDK_INTEGRATED")]
        except (KeyError, ValueError) as e:
            return {"error": f"Invalid parameter: {e}"}

        description = data.get("description", "")
        daily_cap = int(data.get("daily_cap", 0)) or None

        kwargs = {}
        if daily_cap:
            kwargs["daily_cap"] = daily_cap

        game, api_key, fee = self.registry.register_game(
            developer=developer,
            name=name,
            game_type=game_type,
            description=description,
            **kwargs,
        )
        return {
            "game_id": game.game_id,
            "status": game.status.name,
            "api_key": api_key,
            "registration_fee": fee,
            "message": "Game registered. Awaiting AI review and council vote.",
        }

    # ---- 2. positronic_getGameInfo ----

    def get_game_info(self, params: list) -> Optional[dict]:
        """Get information about a registered game.

        params: [game_id]
        """
        if not params:
            return None
        game = self.registry.get_game(str(params[0]))
        return game.to_dict() if game else None

    # ---- 3. positronic_listRegisteredGames ----

    def list_registered_games(self, params: list) -> list:
        """List all active games (or all if include_all=true).

        params: [{include_all?: bool}] or []
        """
        include_all = False
        if params and isinstance(params[0], dict):
            include_all = bool(params[0].get("include_all", False))

        if include_all:
            games = self.registry.list_all_games()
        else:
            games = self.registry.list_active_games()
        return [g.to_dict() for g in games]

    # ---- 4. positronic_getGameBridgeStats ----

    def get_game_bridge_stats(self, params: list) -> dict:
        """Get combined stats from registry, mining, and validator."""
        return {
            "registry": self.registry.get_stats(),
            "mining": self.mining.get_stats(),
            "validator": self.validator.get_stats(),
        }

    # ---- 5. positronic_startGameSession ----

    def start_game_session(self, params: list) -> dict:
        """Start a new game session.

        params: [{api_key, player, proof_type?}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing session data"}
        data = params[0]

        api_key = data.get("api_key", "")
        game, err = self._auth_and_rate_limit(api_key)
        if err:
            return {"error": err}

        try:
            player = bytes.fromhex(data["player"])
        except (KeyError, ValueError):
            return {"error": "Invalid player address"}

        player_hex = player.hex()

        # Check player daily cap
        if not self.mining.check_player_cap(player_hex, game.game_id):
            return {"error": "Player daily mining cap reached"}

        # Check game daily cap
        if not self.mining.check_game_daily_cap(game):
            return {"error": "Game daily emission cap reached"}

        proof_type_name = data.get("proof_type", "LAUNCHER_TRACKED")
        try:
            proof_type = ProofType[proof_type_name]
        except KeyError:
            proof_type = ProofType.LAUNCHER_TRACKED

        session = self.validator.start_session(
            game_id=game.game_id,
            player=player,
            proof_type=proof_type,
        )
        return {
            "session_id": session.session_id,
            "game_id": game.game_id,
            "status": session.status.name,
        }

    # ---- 6. positronic_addGameEvent ----

    def add_game_event(self, params: list) -> dict:
        """Add an event to an active session.

        params: [{session_id, event_type, timestamp?, data?}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing event data"}
        data = params[0]

        import time
        event = GameEvent(
            event_type=str(data.get("event_type", "unknown")),
            timestamp=float(data.get("timestamp", time.time())),
            data=data.get("data", {}),
        )

        ok = self.validator.add_event(str(data.get("session_id", "")), event)
        return {"success": ok}

    # ---- 7. positronic_submitGameSession ----

    def submit_game_session(self, params: list) -> dict:
        """Submit a completed session for validation and reward.

        params: [{api_key, session_id, metrics, proof_signature?, proof_public_key?}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing submission data"}
        data = params[0]

        api_key = data.get("api_key", "")
        game, err = self._auth_and_rate_limit(api_key)
        if err:
            return {"error": err}

        session_id = str(data.get("session_id", ""))
        metrics = data.get("metrics", {})

        proof_sig = b""
        proof_pk = b""
        if data.get("proof_signature"):
            try:
                proof_sig = bytes.fromhex(data["proof_signature"])
            except ValueError:
                pass
        if data.get("proof_public_key"):
            try:
                proof_pk = bytes.fromhex(data["proof_public_key"])
            except ValueError:
                pass

        # Submit session
        session = self.validator.submit_session(
            session_id=session_id,
            metrics=metrics,
            proof_signature=proof_sig,
            proof_public_key=proof_pk,
        )
        if not session:
            return {"error": "Session not found or already submitted"}

        # Validate (AI anti-cheat) — pass game's server pubkey for SERVER_SIGNED proof
        game_pubkey = game.server_pubkey if hasattr(game, 'server_pubkey') else b""
        session = self.validator.validate_session(session, game_pubkey=game_pubkey)

        # Calculate reward if accepted or reduced
        reward = 0
        if session.status.name in ("ACCEPTED", "REDUCED"):
            score = metrics.get("score", 0)
            trust_mult = game.trust_score / 100  # 0-200 → 0.0-2.0
            skill_factor = min(1.0 + (score / 10_000), 3.0)

            reward = self.mining.calculate_reward(
                game=game,
                player_hex=session.player.hex(),
                duration=session.duration,
                score=score,
                skill_factor=skill_factor,
                trust_multiplier=trust_mult,
            )

            # Reduce reward for suspicious sessions
            if session.status.name == "REDUCED":
                reward = reward // 2

            # Record emission + queue for on-chain settlement
            if reward > 0:
                self.mining.record_emission(game, session.player.hex(), reward)
                game.total_sessions += 1
                # Queue reward for on-chain settlement (flushed by blockchain)
                self._pending_rewards[session.player] = (
                    self._pending_rewards.get(session.player, 0) + reward
                )

            session.reward = reward

        # Update game cheat stats
        if session.status.name == "REJECTED":
            game.cheat_detections += 1
            # Phase 15: Exponential trust decay (replaces linear -5)
            if game.cheat_detections % 5 == 0:
                cheat_tier = min(game.cheat_detections // 5, 8)
                trust_loss = GAME_TRUST_DECAY_BASE * (
                    GAME_TRUST_DECAY_EXPONENT ** cheat_tier
                )
                self.registry.adjust_trust(game.game_id, -trust_loss)

        # Persist session to database
        if self._db is not None:
            try:
                self._db.save_session({
                    "session_id": session.session_id,
                    "game_id": game.game_id,
                    "player_hex": session.player.hex(),
                    "status": session.status.name,
                    "started_at": session.started_at,
                    "ended_at": getattr(session, 'ended_at', 0),
                    "duration": session.duration,
                    "reward": reward,
                    "ai_cheat_score": session.ai_cheat_score,
                    "metrics": metrics,
                })
                if reward > 0:
                    self._db.record_emission(
                        game.game_id, session.player.hex(), reward
                    )
            except Exception as e:
                logger.error("session_persist_failed", exc_info=e)

        return {
            "session_id": session.session_id,
            "status": session.status.name,
            "ai_cheat_score": round(session.ai_cheat_score, 4),
            "reward": reward,
            "validation_notes": session.validation_notes,
        }

    # ---- 8. positronic_getSessionStatus ----

    def get_session_status(self, params: list) -> Optional[dict]:
        """Get session status.

        params: [session_id]
        """
        if not params:
            return None
        session = self.validator.get_session(str(params[0]))
        return session.to_dict() if session else None

    # ---- 9. positronic_getPlayerGameHistory ----

    def get_player_game_history(self, params: list) -> list:
        """Get player's session history.

        params: [player_hex, limit?]
        """
        if not params:
            return []
        player_hex = str(params[0])
        limit = int(params[1]) if len(params) > 1 else 20
        return self.validator.get_player_sessions(player_hex, limit)

    # ---- 10. positronic_getGameMiningRate ----

    def get_game_mining_rate(self, params: list) -> Optional[dict]:
        """Get current mining rate information for a game.

        params: [game_id]
        """
        if not params:
            return None
        game = self.registry.get_game(str(params[0]))
        if not game:
            return None

        return {
            "game_id": game.game_id,
            "reward_multiplier": game.reward_multiplier,
            "daily_emission_cap": game.daily_emission_cap,
            "daily_emitted": game.daily_emitted,
            "daily_remaining": max(0, game.daily_emission_cap - game.daily_emitted),
            "trust_score": game.trust_score,
            "scarcity_factor": self.mining.scarcity_factor,
        }

    # ---- 11. positronic_getGameEmission ----

    def get_game_emission(self, params: list) -> Optional[dict]:
        """Get emission statistics for a game.

        params: [game_id]
        """
        if not params:
            return None
        game = self.registry.get_game(str(params[0]))
        if not game:
            return None

        return {
            "game_id": game.game_id,
            "total_emitted": game.total_emitted,
            "total_sessions": game.total_sessions,
            "active_players": game.active_players,
            "total_players": game.total_players,
            "cheat_detections": game.cheat_detections,
        }

    # ---- 12. positronic_getGlobalPlayMineStats ----

    def get_global_play_mine_stats(self, params: list) -> dict:
        """Get global play-to-mine statistics."""
        return self.mining.get_stats()

    # ---- 13. positronic_generateGameAPIKey ----

    def generate_game_api_key(self, params: list) -> dict:
        """Regenerate API key for a game (developer only).

        params: [{game_id, developer}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing parameters"}
        data = params[0]

        try:
            game_id = str(data["game_id"])
            developer = bytes.fromhex(data["developer"])
        except (KeyError, ValueError) as e:
            return {"error": f"Invalid parameter: {e}"}

        new_key = self.registry.regenerate_api_key(game_id, developer)
        if not new_key:
            return {"error": "Game not found or not authorized"}

        return {"api_key": new_key, "message": "API key regenerated"}

    # ---- 14. positronic_testGameSession ----

    def test_game_session(self, params: list) -> dict:
        """Test session validation without real reward.

        params: [{api_key, duration, score, events_count, proof_type?}]
        """
        if not params or not isinstance(params[0], dict):
            return {"error": "Missing test data"}
        data = params[0]

        api_key = data.get("api_key", "")
        game, err = self._auth_and_rate_limit(api_key)
        if err:
            return {"error": err}

        import time as _time
        dummy_player = b"\x00" * 20

        proof_type_name = data.get("proof_type", "LAUNCHER_TRACKED")
        try:
            proof_type = ProofType[proof_type_name]
        except KeyError:
            proof_type = ProofType.LAUNCHER_TRACKED

        # Create test session
        session = self.validator.start_session(
            game_id=game.game_id,
            player=dummy_player,
            proof_type=proof_type,
        )

        # Add fake events
        events_count = int(data.get("events_count", 20))
        now = _time.time()
        import random
        for i in range(min(events_count, 100)):
            event = GameEvent(
                event_type=random.choice([
                    "kill_enemy", "collect_item", "complete_level",
                    "take_damage", "use_skill",
                ]),
                timestamp=now + i * random.uniform(1.0, 10.0),
            )
            self.validator.add_event(session.session_id, event)

        # Submit with test metrics
        duration = float(data.get("duration", 600))
        score = int(data.get("score", 1000))

        # Override started_at so duration calculation works
        session.started_at = _time.time() - duration

        session = self.validator.submit_session(
            session_id=session.session_id,
            metrics={"score": score},
            proof_signature=b"\x01" * 32 if proof_type != ProofType.LAUNCHER_TRACKED else b"",
        )
        if not session:
            return {"error": "Test session submission failed"}

        game_pubkey = game.server_pubkey if hasattr(game, 'server_pubkey') else b""
        session = self.validator.validate_session(session, game_pubkey=game_pubkey)

        # Calculate hypothetical reward (not actually emitted)
        hypothetical_reward = self.mining.calculate_reward(
            game=game,
            player_hex=dummy_player.hex(),
            duration=duration,
            score=score,
        )

        return {
            "test": True,
            "session_id": session.session_id,
            "status": session.status.name,
            "ai_cheat_score": round(session.ai_cheat_score, 4),
            "hypothetical_reward": hypothetical_reward,
            "validation_notes": session.validation_notes,
            "message": "Test session — no real reward emitted",
        }

    # ---- 15. positronic_getGameSDKConfig ----

    def get_game_sdk_config(self, params: list) -> Optional[dict]:
        """Get SDK configuration for a game.

        params: [game_id]
        """
        if not params:
            return None
        game = self.registry.get_game(str(params[0]))
        if not game:
            return None

        return {
            "game_id": game.game_id,
            "name": game.name,
            "game_type": game.game_type.name,
            "status": game.status.name,
            "reward_multiplier": game.reward_multiplier,
            "player_daily_cap": game.player_daily_cap,
            "daily_emission_cap": game.daily_emission_cap,
            "proof_types": [pt.name for pt in ProofType],
            "event_types": [
                "kill_enemy", "collect_item", "complete_level",
                "take_damage", "use_skill", "craft_item",
                "trade", "discover", "achievement",
            ],
            "session_max_duration": 14400,
            "session_min_duration": 60,
            "scarcity_factor": self.mining.scarcity_factor,
        }

    # ---- On-chain reward settlement ----

    def get_pending_rewards(self) -> Dict[bytes, int]:
        """Get all pending game bridge rewards to be settled on-chain."""
        return dict(self._pending_rewards)

    def clear_pending_rewards(self):
        """Clear pending rewards after on-chain settlement."""
        self._pending_rewards.clear()
