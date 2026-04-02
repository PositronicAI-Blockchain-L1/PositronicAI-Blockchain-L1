"""
Positronic - Game Registry
Register, approve, and manage external games that can mine ASF tokens.
Games must pass AI risk review + governance council vote before activation.
"""

import json
import sqlite3
import time
import hashlib
import secrets
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from positronic.constants import BASE_UNIT, GAME_HEARTBEAT_GRACE, GAME_REGISTRATION_FEE


class GameType(IntEnum):
    """How the game connects to the blockchain."""
    PLUGIN = 0          # Game-specific plugin (e.g., Minecraft mod)
    SERVER_ORACLE = 1   # Game server submits signed results
    LAUNCHER = 2        # External launcher monitors gameplay
    SDK_INTEGRATED = 3  # Game uses Positronic SDK natively


class GameStatus(IntEnum):
    """Lifecycle status of a registered game."""
    PENDING = 0         # Awaiting AI review
    AI_REVIEWING = 1    # AI risk assessment in progress
    AI_APPROVED = 2     # Passed AI review, awaiting council vote
    AI_REJECTED = 3     # Failed AI review
    COUNCIL_VOTING = 4  # Council members voting
    APPROVED = 5        # Approved but not yet active
    ACTIVE = 6          # Live and mining-enabled
    SUSPENDED = 7       # Temporarily disabled (cheat spike, etc.)
    REVOKED = 8         # Permanently removed


@dataclass
class GameInfo:
    """Complete information about a registered game."""
    game_id: str
    name: str
    developer: bytes            # Developer's blockchain address
    game_type: GameType
    status: GameStatus = GameStatus.PENDING
    description: str = ""

    # Authentication
    api_key_hash: str = ""      # SHA-512 hash of the API key
    server_pubkey: bytes = b""  # Game server's public key for SERVER_SIGNED proof

    # Mining parameters
    daily_emission_cap: int = 10_000 * BASE_UNIT   # Default 10K ASF/day
    reward_multiplier: float = 1.0                  # 0.1x to 2.0x
    player_daily_cap: int = 50 * BASE_UNIT          # 50 ASF/day per player

    # Statistics
    total_emitted: int = 0
    total_sessions: int = 0
    active_players: int = 0
    total_players: int = 0
    cheat_detections: int = 0

    # Trust & reputation
    trust_score: int = 100      # Starts at 100, adjusted by behavior
    ai_risk_score: float = 0.0  # AI assessment (0=safe, 1=risky)

    # Governance
    votes_for: int = 0
    votes_against: int = 0
    voters: List[str] = field(default_factory=list)

    # Timestamps
    registered_at: float = 0.0
    approved_at: float = 0.0
    last_session_at: float = 0.0

    # Daily tracking (reset every 24h)
    daily_emitted: int = 0
    daily_reset_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "name": self.name,
            "developer": self.developer.hex(),
            "game_type": self.game_type.name,
            "status": self.status.name,
            "description": self.description,
            "daily_emission_cap": self.daily_emission_cap,
            "reward_multiplier": self.reward_multiplier,
            "player_daily_cap": self.player_daily_cap,
            "total_emitted": self.total_emitted,
            "total_sessions": self.total_sessions,
            "active_players": self.active_players,
            "total_players": self.total_players,
            "cheat_detections": self.cheat_detections,
            "trust_score": self.trust_score,
            "ai_risk_score": self.ai_risk_score,
            "registered_at": self.registered_at,
        }


# ---- Constants ----
AI_MAX_RISK_SCORE = 0.6          # Stricter than token governance (0.7)
COUNCIL_MIN_VOTES = 3
COUNCIL_APPROVAL_PCT = 0.6       # 60% approval
MIN_GAME_TRUST = 10              # Below this, game is auto-suspended
DEFAULT_DAILY_CAP = 10_000 * BASE_UNIT
MAX_DAILY_CAP = 100_000 * BASE_UNIT
DAILY_RESET_SECONDS = 86_400     # 24 hours


class GameRegistry:
    """
    Manages registration and lifecycle of external games.

    Flow: Register → AI Review → Council Vote → Activate → Monitor

    If ``db_path`` is provided, state is persisted to SQLite and restored
    on init.  Without ``db_path`` the registry is purely in-memory
    (backward-compatible default).
    """

    def __init__(self, db_path: Optional[str] = None):
        self._games: Dict[str, GameInfo] = {}
        self._api_keys: Dict[str, str] = {}  # api_key_hash → game_id
        self._council_members: set = set()
        self._game_counter: int = 0
        self._db_path = db_path
        self._db: Optional[sqlite3.Connection] = None

        if db_path:
            self._init_db()
            self._load_from_db()

    # ---- SQLite persistence ----

    def _init_db(self):
        self._db = sqlite3.connect(self._db_path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS game_registry_games (
                game_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                developer_hex TEXT NOT NULL,
                game_type INTEGER NOT NULL,
                status INTEGER NOT NULL,
                description TEXT DEFAULT '',
                api_key_hash TEXT DEFAULT '',
                daily_emission_cap INTEGER DEFAULT 0,
                reward_multiplier REAL DEFAULT 1.0,
                player_daily_cap INTEGER DEFAULT 0,
                total_emitted INTEGER DEFAULT 0,
                total_sessions INTEGER DEFAULT 0,
                active_players INTEGER DEFAULT 0,
                total_players INTEGER DEFAULT 0,
                cheat_detections INTEGER DEFAULT 0,
                trust_score INTEGER DEFAULT 100,
                ai_risk_score REAL DEFAULT 0.0,
                votes_for INTEGER DEFAULT 0,
                votes_against INTEGER DEFAULT 0,
                voters_json TEXT DEFAULT '[]',
                registered_at REAL DEFAULT 0.0,
                approved_at REAL DEFAULT 0.0,
                last_session_at REAL DEFAULT 0.0,
                daily_emitted INTEGER DEFAULT 0,
                daily_reset_time REAL DEFAULT 0.0
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS game_registry_council (
                address_hex TEXT PRIMARY KEY
            )
        """)
        self._db.commit()

    def _load_from_db(self):
        if not self._db:
            return
        cur = self._db.execute("SELECT * FROM game_registry_games")
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall():
            r = dict(zip(cols, row))
            game = GameInfo(
                game_id=r["game_id"],
                name=r["name"],
                developer=bytes.fromhex(r["developer_hex"]),
                game_type=GameType(r["game_type"]),
                status=GameStatus(r["status"]),
                description=r["description"],
                api_key_hash=r["api_key_hash"],
                daily_emission_cap=r["daily_emission_cap"],
                reward_multiplier=r["reward_multiplier"],
                player_daily_cap=r["player_daily_cap"],
                total_emitted=r["total_emitted"],
                total_sessions=r["total_sessions"],
                active_players=r["active_players"],
                total_players=r["total_players"],
                cheat_detections=r["cheat_detections"],
                trust_score=r["trust_score"],
                ai_risk_score=r["ai_risk_score"],
                votes_for=r["votes_for"],
                votes_against=r["votes_against"],
                voters=json.loads(r["voters_json"]),
                registered_at=r["registered_at"],
                approved_at=r["approved_at"],
                last_session_at=r["last_session_at"],
                daily_emitted=r["daily_emitted"],
                daily_reset_time=r["daily_reset_time"],
            )
            self._games[game.game_id] = game
            if game.api_key_hash:
                self._api_keys[game.api_key_hash] = game.game_id
        # Restore counter
        if self._games:
            self._game_counter = len(self._games)
        # Council
        for row in self._db.execute("SELECT address_hex FROM game_registry_council"):
            self._council_members.add(bytes.fromhex(row[0]))

    def _persist_game(self, game: GameInfo):
        if not self._db:
            return
        self._db.execute("""
            INSERT OR REPLACE INTO game_registry_games (
                game_id, name, developer_hex, game_type, status, description,
                api_key_hash, daily_emission_cap, reward_multiplier, player_daily_cap,
                total_emitted, total_sessions, active_players, total_players,
                cheat_detections, trust_score, ai_risk_score,
                votes_for, votes_against, voters_json,
                registered_at, approved_at, last_session_at,
                daily_emitted, daily_reset_time
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game.game_id, game.name, game.developer.hex(),
            int(game.game_type), int(game.status), game.description,
            game.api_key_hash, game.daily_emission_cap, game.reward_multiplier,
            game.player_daily_cap, game.total_emitted, game.total_sessions,
            game.active_players, game.total_players, game.cheat_detections,
            game.trust_score, game.ai_risk_score,
            game.votes_for, game.votes_against, json.dumps(game.voters),
            game.registered_at, game.approved_at, game.last_session_at,
            game.daily_emitted, game.daily_reset_time,
        ))
        self._db.commit()

    def _persist_council(self):
        if not self._db:
            return
        self._db.execute("DELETE FROM game_registry_council")
        for addr in self._council_members:
            self._db.execute(
                "INSERT INTO game_registry_council (address_hex) VALUES (?)",
                (addr.hex(),),
            )
        self._db.commit()

    def close(self):
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None

    def add_council_member(self, address: bytes):
        self._council_members.add(address)
        self._persist_council()

    def remove_council_member(self, address: bytes):
        self._council_members.discard(address)
        self._persist_council()

    # ---- Registration ----

    def register_game(
        self,
        developer: bytes,
        name: str,
        game_type: GameType,
        description: str = "",
        daily_cap: int = DEFAULT_DAILY_CAP,
    ) -> tuple:
        """Register a new game. Returns (GameInfo, api_key, registration_fee).

        The caller (RPC handler / executor) is responsible for deducting
        GAME_REGISTRATION_FEE from the developer's account and crediting
        TEAM_ADDRESS before committing the registration.
        """
        self._game_counter += 1
        game_id = f"GAME-{int(time.time())}-{self._game_counter:04d}"

        # Generate API key
        api_key = f"gk_{secrets.token_hex(32)}"
        api_key_hash = hashlib.sha512(api_key.encode()).hexdigest()

        # Clamp daily cap
        capped = max(BASE_UNIT, min(daily_cap, MAX_DAILY_CAP))

        game = GameInfo(
            game_id=game_id,
            name=name,
            developer=developer,
            game_type=game_type,
            description=description,
            status=GameStatus.PENDING,
            api_key_hash=api_key_hash,
            daily_emission_cap=capped,
            registered_at=time.time(),
            daily_reset_time=time.time(),
        )

        self._games[game_id] = game
        self._api_keys[api_key_hash] = game_id
        self._persist_game(game)

        return game, api_key, GAME_REGISTRATION_FEE

    # ---- AI Review ----

    def ai_review(self, game_id: str, risk_score: float, notes: str = "") -> Optional[GameInfo]:
        """AI reviews a game registration."""
        game = self._games.get(game_id)
        if not game or game.status != GameStatus.PENDING:
            return None

        game.ai_risk_score = risk_score
        game.status = GameStatus.AI_REVIEWING

        if risk_score <= AI_MAX_RISK_SCORE:
            game.status = GameStatus.AI_APPROVED
        else:
            game.status = GameStatus.AI_REJECTED

        self._persist_game(game)
        return game

    # ---- Council Vote ----

    def council_vote(self, game_id: str, voter: bytes, approve: bool) -> Optional[GameInfo]:
        """Council member votes on a game."""
        game = self._games.get(game_id)
        if not game:
            return None
        if game.status not in (GameStatus.AI_APPROVED, GameStatus.COUNCIL_VOTING):
            return None
        if voter not in self._council_members:
            return None

        voter_hex = voter.hex()
        if voter_hex in game.voters:
            return None  # Already voted

        game.status = GameStatus.COUNCIL_VOTING
        game.voters.append(voter_hex)

        if approve:
            game.votes_for += 1
        else:
            game.votes_against += 1

        total = game.votes_for + game.votes_against
        if total >= COUNCIL_MIN_VOTES:
            ratio = game.votes_for / total
            if ratio >= COUNCIL_APPROVAL_PCT:
                game.status = GameStatus.APPROVED
                game.approved_at = time.time()
            elif total >= len(self._council_members):
                game.status = GameStatus.AI_REJECTED

        self._persist_game(game)
        return game

    def activate_game(self, game_id: str) -> Optional[GameInfo]:
        """Activate an approved game for mining."""
        game = self._games.get(game_id)
        if not game or game.status != GameStatus.APPROVED:
            return None
        game.status = GameStatus.ACTIVE
        self._persist_game(game)
        return game

    # ---- Authentication ----

    def authenticate(self, api_key: str) -> Optional[GameInfo]:
        """Verify API key and return game info."""
        key_hash = hashlib.sha512(api_key.encode()).hexdigest()
        game_id = self._api_keys.get(key_hash)
        if not game_id:
            return None
        game = self._games.get(game_id)
        if not game or game.status != GameStatus.ACTIVE:
            return None
        return game

    def regenerate_api_key(self, game_id: str, developer: bytes) -> Optional[str]:
        """Regenerate API key (only by developer)."""
        game = self._games.get(game_id)
        if not game or game.developer != developer:
            return None

        # Remove old key
        old_hash = game.api_key_hash
        self._api_keys.pop(old_hash, None)

        # Generate new key
        new_key = f"gk_{secrets.token_hex(32)}"
        new_hash = hashlib.sha512(new_key.encode()).hexdigest()
        game.api_key_hash = new_hash
        self._api_keys[new_hash] = game_id
        self._persist_game(game)

        return new_key

    # ---- Daily Reset ----

    def check_daily_reset(self, game: GameInfo):
        """Reset daily emission counter if 24h passed."""
        now = time.time()
        if now - game.daily_reset_time >= DAILY_RESET_SECONDS:
            game.daily_emitted = 0
            game.daily_reset_time = now

    # ---- Trust Management ----

    def adjust_trust(self, game_id: str, delta: int):
        """Adjust a game's trust score."""
        game = self._games.get(game_id)
        if not game:
            return
        game.trust_score = max(0, game.trust_score + delta)

        # Auto-suspend if trust too low
        if game.trust_score < MIN_GAME_TRUST and game.status == GameStatus.ACTIVE:
            game.status = GameStatus.SUSPENDED
        self._persist_game(game)

    # ---- Phase 15: Heartbeat & Emergency Pause ----

    def heartbeat(self, game_id: str, api_key: str) -> bool:
        """Game server sends periodic heartbeat to prove liveness."""
        game = self.authenticate(api_key)
        if not game or game.game_id != game_id:
            return False
        game.last_heartbeat = time.time()
        return True

    def check_heartbeats(self) -> List[str]:
        """Suspend games with no heartbeat beyond grace period."""
        now = time.time()
        suspended = []
        for game in self._games.values():
            if game.status == GameStatus.ACTIVE:
                last_hb = getattr(game, 'last_heartbeat', 0)
                if last_hb > 0 and now - last_hb > GAME_HEARTBEAT_GRACE:
                    game.status = GameStatus.SUSPENDED
                    suspended.append(game.game_id)
        return suspended

    def emergency_pause(self, game_id: str) -> bool:
        """Pause a single game without affecting others."""
        game = self._games.get(game_id)
        if game and game.status == GameStatus.ACTIVE:
            game.status = GameStatus.SUSPENDED
            return True
        return False

    def suspend_game(self, game_id: str) -> Optional[GameInfo]:
        game = self._games.get(game_id)
        if game and game.status == GameStatus.ACTIVE:
            game.status = GameStatus.SUSPENDED
            self._persist_game(game)
        return game

    def revoke_game(self, game_id: str) -> Optional[GameInfo]:
        game = self._games.get(game_id)
        if game:
            game.status = GameStatus.REVOKED
            self._api_keys.pop(game.api_key_hash, None)
            self._persist_game(game)
        return game

    # ---- Queries ----

    def get_game(self, game_id: str) -> Optional[GameInfo]:
        return self._games.get(game_id)

    def list_active_games(self) -> List[GameInfo]:
        return [g for g in self._games.values() if g.status == GameStatus.ACTIVE]

    def list_all_games(self) -> List[GameInfo]:
        return list(self._games.values())

    def get_stats(self) -> dict:
        games = list(self._games.values())
        active = [g for g in games if g.status == GameStatus.ACTIVE]
        return {
            "total_registered": len(games),
            "active_games": len(active),
            "total_emitted_all_games": sum(g.total_emitted for g in games),
            "total_sessions_all_games": sum(g.total_sessions for g in games),
            "total_players_all_games": sum(g.total_players for g in games),
            "council_members": len(self._council_members),
        }
