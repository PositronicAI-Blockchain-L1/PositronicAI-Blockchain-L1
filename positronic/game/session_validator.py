"""
Positronic - Game Session Validator
Validates external game sessions with Proof-of-Play verification
and AI-powered anti-cheat detection.
"""

import time
import hashlib
import hmac
import statistics
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.constants import BASE_UNIT, GAME_SCORE_MAX_PER_MINUTE
from positronic.types import PlayerSessionRecord, GameSessionStats


class ProofType(IntEnum):
    """How the game session was verified."""
    SERVER_SIGNED = 0     # Game server signed the result
    PLUGIN_VERIFIED = 1   # Game plugin captured events
    LAUNCHER_TRACKED = 2  # External launcher monitored gameplay


class SessionStatus(IntEnum):
    """Status of a game session."""
    STARTED = 0
    SUBMITTED = 1
    VALIDATING = 2
    ACCEPTED = 3
    REDUCED = 4       # Accepted but with reduced reward (mild suspicion)
    REJECTED = 5
    EXPIRED = 6


@dataclass
class GameEvent:
    """A single gameplay event captured during a session."""
    event_type: str       # e.g., "kill_enemy", "collect_item", "complete_level"
    timestamp: float
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class GameSession:
    """A complete game session from an external game."""
    session_id: str
    game_id: str
    player: bytes
    started_at: float
    proof_type: ProofType

    # Filled during gameplay
    events: List[GameEvent] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)  # score, kills, items, etc.

    # Filled on submission
    ended_at: float = 0.0
    duration: float = 0.0
    proof_signature: bytes = b""     # Digital signature from game server/plugin
    proof_public_key: bytes = b""    # Public key of signer

    # Filled during validation
    status: SessionStatus = SessionStatus.STARTED
    ai_cheat_score: float = 0.0
    reward: int = 0
    validation_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "game_id": self.game_id,
            "player": self.player.hex(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": self.duration,
            "proof_type": self.proof_type.name,
            "status": self.status.name,
            "ai_cheat_score": self.ai_cheat_score,
            "reward": self.reward,
            "events_count": len(self.events),
            "metrics": self.metrics,
        }


# ---- Cheat Detection Thresholds ----
CHEAT_ACCEPT = 0.3       # < 0.3 = clean
CHEAT_REDUCE = 0.7       # 0.3-0.7 = reduced reward
# > 0.7 = rejected

SESSION_MAX_DURATION = 14_400   # 4 hours max
SESSION_MIN_DURATION = 60       # 1 minute min
SESSION_MAX_EVENTS = 50_000     # Max events per session
SESSION_EXPIRE_TIME = 86_400    # Sessions expire after 24h if not submitted

# Weights for cheat score components
W_TIMING = 0.25
W_SCORE = 0.25
W_PATTERN = 0.20
W_CROSS_GAME = 0.15
W_STATISTICAL = 0.15


class GameSessionValidator:
    """
    Validates game sessions and detects cheating using AI-inspired analysis.

    Checks:
    1. Proof verification — is the signature valid?
    2. Timing analysis — are action timings natural?
    3. Score consistency — does score match gameplay metrics?
    4. Pattern analysis — does behavior look like a bot?
    5. Cross-game analysis — is player mining multiple games simultaneously?
    6. Statistical outlier — are results abnormally high?
    """

    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}
        self._active_sessions: Dict[str, str] = {}  # player_hex → session_id
        self._player_history: Dict[str, List[dict]] = {}  # player_hex → recent sessions
        self._game_averages: Dict[str, dict] = {}  # game_id → average metrics
        self._session_counter: int = 0

    # ---- Session Lifecycle ----

    def start_session(
        self,
        game_id: str,
        player: bytes,
        proof_type: ProofType,
    ) -> GameSession:
        """Start a new game session."""
        self._session_counter += 1
        session_id = f"SES-{int(time.time())}-{self._session_counter:06d}"

        session = GameSession(
            session_id=session_id,
            game_id=game_id,
            player=player,
            started_at=time.time(),
            proof_type=proof_type,
        )

        self._sessions[session_id] = session
        self._active_sessions[player.hex()] = session_id
        return session

    def add_event(self, session_id: str, event: GameEvent) -> bool:
        """Add a gameplay event to an active session."""
        session = self._sessions.get(session_id)
        if not session or session.status != SessionStatus.STARTED:
            return False
        if len(session.events) >= SESSION_MAX_EVENTS:
            return False
        session.events.append(event)
        return True

    def submit_session(
        self,
        session_id: str,
        metrics: dict,
        proof_signature: bytes = b"",
        proof_public_key: bytes = b"",
    ) -> Optional[GameSession]:
        """Submit a completed session for validation."""
        session = self._sessions.get(session_id)
        if not session or session.status != SessionStatus.STARTED:
            return None

        now = time.time()
        session.ended_at = now
        session.duration = now - session.started_at
        session.metrics = metrics
        session.proof_signature = proof_signature
        session.proof_public_key = proof_public_key
        session.status = SessionStatus.SUBMITTED

        # Clean up active session
        self._active_sessions.pop(session.player.hex(), None)

        return session

    # ---- Validation Pipeline ----

    def verify_proof_signature(
        self, session: GameSession, game_pubkey: bytes = b""
    ) -> bool:
        """
        Phase 15: Verify proof signature against game's registered public key.
        SERVER_SIGNED requires valid cryptographic proof.
        PLUGIN_VERIFIED requires non-empty signature.
        LAUNCHER_TRACKED relies on anti-cheat AI (no proof needed).
        """
        if session.proof_type == ProofType.SERVER_SIGNED:
            if not session.proof_signature or not game_pubkey:
                return False
            # Construct signed payload: session_id + player + score + duration
            payload = (
                session.session_id
                + session.player.hex()
                + str(session.metrics.get("score", 0))
                + str(int(session.duration))
            )
            payload_hash = hashlib.sha512(payload.encode()).digest()
            # Verify HMAC-based signature (game server uses shared secret)
            expected = hmac.new(game_pubkey, payload_hash, hashlib.sha256).digest()
            return hmac.compare_digest(
                session.proof_signature[:32], expected[:32]
            )
        elif session.proof_type == ProofType.PLUGIN_VERIFIED:
            if not session.proof_signature:
                return False
            return len(session.proof_signature) >= 32
        else:
            # LAUNCHER_TRACKED — no server-side proof, rely on anti-cheat
            return True

    def validate_session(self, session: GameSession, game_pubkey: bytes = b"") -> GameSession:
        """Run full validation pipeline on a submitted session."""
        session.status = SessionStatus.VALIDATING

        # Basic checks
        if not self._check_basic_validity(session):
            session.status = SessionStatus.REJECTED
            session.validation_notes = "Failed basic validity checks"
            return session

        # Phase 15: Score bounds validation
        duration_min = max(session.duration / 60, 0.01)
        max_allowed_score = GAME_SCORE_MAX_PER_MINUTE * duration_min
        actual_score = session.metrics.get("score", 0)
        if actual_score > max_allowed_score:
            session.status = SessionStatus.REJECTED
            session.ai_cheat_score = 0.95
            session.validation_notes = (
                f"Score {actual_score} exceeds max {int(max_allowed_score)} "
                f"for {duration_min:.1f}min session"
            )
            self._record_history(session)
            return session

        # Phase 15: Proof signature verification
        if not self.verify_proof_signature(session, game_pubkey):
            session.status = SessionStatus.REJECTED
            session.ai_cheat_score = 0.90
            session.validation_notes = "Invalid proof signature"
            self._record_history(session)
            return session

        # AI cheat detection
        cheat_score = self._compute_cheat_score(session)
        session.ai_cheat_score = cheat_score

        if cheat_score < CHEAT_ACCEPT:
            session.status = SessionStatus.ACCEPTED
            session.validation_notes = "Clean session"
        elif cheat_score < CHEAT_REDUCE:
            session.status = SessionStatus.REDUCED
            session.validation_notes = f"Mild suspicion (score={cheat_score:.2f}), reward reduced"
        else:
            session.status = SessionStatus.REJECTED
            session.validation_notes = f"Cheat detected (score={cheat_score:.2f})"

        # Record in player history
        self._record_history(session)

        return session

    def _check_basic_validity(self, session: GameSession) -> bool:
        """Basic sanity checks before AI analysis."""
        # Duration bounds
        if session.duration < SESSION_MIN_DURATION:
            return False
        if session.duration > SESSION_MAX_DURATION:
            return False

        # Must have some events or metrics
        if not session.events and not session.metrics:
            return False

        # Score must be non-negative
        score = session.metrics.get("score", 0)
        if score < 0:
            return False

        # Proof signature required for SERVER_SIGNED and PLUGIN_VERIFIED
        if session.proof_type in (ProofType.SERVER_SIGNED, ProofType.PLUGIN_VERIFIED):
            if not session.proof_signature:
                return False

        return True

    # ---- AI Cheat Detection ----

    def _compute_cheat_score(self, session: GameSession) -> float:
        """
        Compute composite cheat score (0=clean, 1=cheat).
        Uses weighted combination of 5 analysis modules.
        """
        timing = self._analyze_timing(session)
        score_c = self._analyze_score_consistency(session)
        pattern = self._analyze_patterns(session)
        cross = self._analyze_cross_game(session)
        outlier = self._analyze_statistical_outlier(session)

        composite = (
            W_TIMING * timing +
            W_SCORE * score_c +
            W_PATTERN * pattern +
            W_CROSS_GAME * cross +
            W_STATISTICAL * outlier
        )

        return min(1.0, max(0.0, composite))

    def _analyze_timing(self, session: GameSession) -> float:
        """
        Check if event timings are suspiciously uniform (bot-like).
        Humans have natural variance; bots are too regular.
        """
        if len(session.events) < 5:
            return 0.1  # Too few events to judge

        timestamps = [e.timestamp for e in session.events]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        if not intervals:
            return 0.1

        # Filter out negative/zero intervals
        intervals = [i for i in intervals if i > 0]
        if len(intervals) < 3:
            return 0.1

        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.9  # All at same time = bot

        # Coefficient of variation: std_dev / mean
        try:
            std_dev = statistics.stdev(intervals)
            cv = std_dev / mean_interval if mean_interval > 0 else 0
        except statistics.StatisticsError:
            return 0.1

        # Humans: CV typically 0.3-1.5
        # Bots: CV typically < 0.1 (too uniform)
        if cv < 0.05:
            return 0.95  # Extremely uniform = bot
        if cv < 0.15:
            return 0.7
        if cv < 0.25:
            return 0.4
        return 0.1  # Natural variance = human

    def _analyze_score_consistency(self, session: GameSession) -> float:
        """Check if score is consistent with gameplay duration and events."""
        score = session.metrics.get("score", 0)
        duration_min = session.duration / 60

        if duration_min <= 0:
            return 0.9

        # Score per minute rate
        score_rate = score / duration_min

        # Events per minute
        epm = len(session.events) / duration_min if duration_min > 0 else 0

        suspicion = 0.0

        # Extremely high score rate
        if score_rate > 5000:
            suspicion += 0.4
        elif score_rate > 2000:
            suspicion += 0.2

        # Score with no events
        if score > 1000 and len(session.events) == 0:
            suspicion += 0.5

        # Impossibly high events per minute (>120 = 2/sec sustained)
        if epm > 120:
            suspicion += 0.3
        elif epm > 60:
            suspicion += 0.15

        return min(1.0, suspicion)

    def _analyze_patterns(self, session: GameSession) -> float:
        """Detect repetitive patterns typical of bots."""
        if len(session.events) < 10:
            return 0.1

        # Check for repeating event sequences
        types = [e.event_type for e in session.events]

        # Count unique event types
        unique_types = set(types)
        type_diversity = len(unique_types) / max(len(types), 1)

        # Very low diversity = repetitive bot
        if type_diversity < 0.02:
            return 0.9
        if type_diversity < 0.05:
            return 0.6

        # Check for exact repeating subsequences (length 3-5)
        repeat_score = 0.0
        for seq_len in [3, 4, 5]:
            sequences = []
            for i in range(len(types) - seq_len):
                seq = tuple(types[i:i+seq_len])
                sequences.append(seq)
            if sequences:
                unique_seqs = set(sequences)
                repeat_ratio = 1 - (len(unique_seqs) / len(sequences))
                if repeat_ratio > 0.8:
                    repeat_score = max(repeat_score, 0.8)
                elif repeat_ratio > 0.5:
                    repeat_score = max(repeat_score, 0.4)

        return min(1.0, max(type_diversity < 0.1 and 0.3 or 0.0, repeat_score))

    def _analyze_cross_game(self, session: GameSession) -> float:
        """Check if player has overlapping sessions in other games."""
        player_hex = session.player.hex()

        # Check for other active sessions
        active_id = self._active_sessions.get(player_hex)
        if active_id and active_id != session.session_id:
            other = self._sessions.get(active_id)
            if other and other.game_id != session.game_id:
                # Playing two games simultaneously
                return 0.8

        # Check recent history for impossibly fast game-hopping
        history = self._player_history.get(player_hex, [])
        if len(history) >= 2:
            recent = history[-1]
            gap = session.started_at - recent.get("ended_at", 0)
            if gap < 30 and recent.get("game_id") != session.game_id:
                # Less than 30s between different games
                return 0.6

        return 0.0

    def _analyze_statistical_outlier(self, session: GameSession) -> float:
        """Check if results are statistical outliers vs game average."""
        game_id = session.game_id
        avg = self._game_averages.get(game_id)
        if not avg or avg.get("count", 0) < 10:
            return 0.1  # Not enough data

        score = session.metrics.get("score", 0)
        avg_score = avg.get("avg_score", 0)
        std_score = avg.get("std_score", 1)

        if std_score <= 0:
            return 0.1

        # Z-score: how many standard deviations from mean
        z = abs(score - avg_score) / std_score

        if z > 4:
            return 0.8  # Extreme outlier
        if z > 3:
            return 0.5
        if z > 2:
            return 0.2
        return 0.0

    # ---- History & Statistics ----

    def _record_history(self, session: GameSession) -> None:
        """Record session in player history and update game averages."""
        player_hex = session.player.hex()

        # Player history (keep last 50)
        if player_hex not in self._player_history:
            self._player_history[player_hex] = []
        self._player_history[player_hex].append({
            "session_id": session.session_id,
            "game_id": session.game_id,
            "score": session.metrics.get("score", 0),
            "duration": session.duration,
            "cheat_score": session.ai_cheat_score,
            "ended_at": session.ended_at,
        })
        if len(self._player_history[player_hex]) > 50:
            self._player_history[player_hex] = self._player_history[player_hex][-50:]

        # Update game averages (running mean/std)
        if session.status in (SessionStatus.ACCEPTED, SessionStatus.REDUCED):
            self._update_game_average(session)

    def _update_game_average(self, session: GameSession) -> None:
        """Update running average for a game."""
        game_id = session.game_id
        score = session.metrics.get("score", 0)

        if game_id not in self._game_averages:
            self._game_averages[game_id] = {
                "count": 0, "total_score": 0,
                "scores": [], "avg_score": 0, "std_score": 1,
            }

        avg = self._game_averages[game_id]
        avg["count"] += 1
        avg["total_score"] += score
        avg["scores"].append(score)

        # Keep last 1000 scores for std calculation
        if len(avg["scores"]) > 1000:
            avg["scores"] = avg["scores"][-1000:]

        avg["avg_score"] = avg["total_score"] / avg["count"]
        if len(avg["scores"]) >= 2:
            try:
                avg["std_score"] = statistics.stdev(avg["scores"])
            except statistics.StatisticsError:
                avg["std_score"] = 1

    # ---- Queries ----

    def get_session(self, session_id: str) -> Optional[GameSession]:
        return self._sessions.get(session_id)

    def get_player_sessions(self, player_hex: str, limit: int = 20) -> List[PlayerSessionRecord]:
        history = self._player_history.get(player_hex, [])
        return history[-limit:]

    def get_stats(self) -> GameSessionStats:
        sessions = list(self._sessions.values())
        accepted = sum(1 for s in sessions if s.status == SessionStatus.ACCEPTED)
        rejected = sum(1 for s in sessions if s.status == SessionStatus.REJECTED)
        reduced = sum(1 for s in sessions if s.status == SessionStatus.REDUCED)
        return {
            "total_sessions": len(sessions),
            "accepted": accepted,
            "rejected": rejected,
            "reduced": reduced,
            "active_sessions": len(self._active_sessions),
            "games_tracked": len(self._game_averages),
            "players_tracked": len(self._player_history),
        }
