"""
Positronic - TRUST System (Soulbound Token)
Non-transferable reputation token tied to on-chain behavior.
Higher TRUST = higher mining rewards (up to 2.0x multiplier).

Audit fix: Added daily rate limit (TRUST_DAILY_CAP = 50) to prevent
bot farming. A miner at 3s blocks could earn 28,800 TRUST/day without
this cap, trivializing the reputation system.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time

from positronic.constants import TRUST_DAILY_CAP


class TrustLevel(IntEnum):
    """Trust levels based on accumulated score."""
    NEWCOMER   = 0   # score 0-99      — 0.5x  (new validator, limited weight)
    APPRENTICE = 1   # score 100-499   — 0.75x (building track record)
    TRUSTED    = 2   # score 500-1999  — 1.0x  (baseline, fully trusted)
    VETERAN    = 3   # score 2000-9999 — 1.5x  (proven long-term validator)
    LEGEND     = 4   # score 10000+    — 2.0x  (elite, double mining weight)


# Trust level thresholds
TRUST_THRESHOLDS = {
    TrustLevel.NEWCOMER:   0,
    TrustLevel.APPRENTICE: 100,
    TrustLevel.TRUSTED:    500,
    TrustLevel.VETERAN:    2000,
    TrustLevel.LEGEND:     10000,
}

# Mining weight multiplier per level
# NEWCOMER starts at 0.5x — earns full weight only after proving reputation
# Path: 0.5x → 0.75x → 1.0x → 1.5x → 2.0x (purely through on-chain behavior)
TRUST_MULTIPLIERS = {
    TrustLevel.NEWCOMER:   0.5,
    TrustLevel.APPRENTICE: 0.75,
    TrustLevel.TRUSTED:    1.0,
    TrustLevel.VETERAN:    1.5,
    TrustLevel.LEGEND:     2.0,
}

# Backward-compat alias (MEMBER was renamed to APPRENTICE)
TrustLevel.MEMBER = TrustLevel.APPRENTICE  # type: ignore[attr-defined]

# Score adjustments for actions
TRUST_REWARDS = {
    "block_mined": 1,
    "game_completed": 1,
    "dao_vote": 2,
    "token_created": 3,
    "nft_created": 2,
    "consistent_uptime": 1,  # per epoch of uptime
}

TRUST_PENALTIES = {
    "suspicious_tx": -20,
    "quarantined_tx": -10,
    "false_evidence": -100,
    "slashed": -50,
    "spam_detected": -5,
}

# Inactivity decay: lose 1 point per 30 days of inactivity
INACTIVITY_DECAY_INTERVAL = 30 * 24 * 3600  # 30 days in seconds
INACTIVITY_DECAY_AMOUNT = 1


@dataclass
class TrustProfile:
    """Soulbound trust profile for an address."""
    address: bytes
    score: int = 0
    level: TrustLevel = TrustLevel.NEWCOMER
    blocks_mined: int = 0
    games_completed: int = 0
    dao_votes: int = 0
    tokens_created: int = 0
    penalties_received: int = 0
    last_active: float = 0.0
    created_at: float = 0.0

    @property
    def mining_multiplier(self) -> float:
        """Get mining reward multiplier based on trust level."""
        return TRUST_MULTIPLIERS.get(self.level, 1.0)

    def _recalculate_level(self):
        """Recalculate trust level from score."""
        for level in reversed(TrustLevel):
            if self.score >= TRUST_THRESHOLDS[level]:
                self.level = level
                return

    def add_score(self, amount: int, reason: str = "") -> int:
        """Add to trust score. Score cannot go below 0."""
        self.score = max(0, self.score + amount)
        self.last_active = time.time()
        self._recalculate_level()
        return self.score

    def apply_decay(self, current_time: float) -> int:
        """Apply inactivity decay if applicable."""
        if self.last_active <= 0:
            return self.score
        elapsed = current_time - self.last_active
        decay_periods = int(elapsed / INACTIVITY_DECAY_INTERVAL)
        if decay_periods > 0:
            penalty = decay_periods * INACTIVITY_DECAY_AMOUNT
            self.score = max(0, self.score - penalty)
            self._recalculate_level()
        return self.score

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "score": self.score,
            "level": self.level.name,
            "level_value": int(self.level),
            "mining_multiplier": self.mining_multiplier,
            "blocks_mined": self.blocks_mined,
            "games_completed": self.games_completed,
            "dao_votes": self.dao_votes,
            "tokens_created": self.tokens_created,
            "penalties_received": self.penalties_received,
            "last_active": self.last_active,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrustProfile":
        return cls(
            address=bytes.fromhex(d["address"].removeprefix("0x")),
            score=d.get("score", 0),
            level=TrustLevel(d.get("level_value", 0)),
            blocks_mined=d.get("blocks_mined", 0),
            games_completed=d.get("games_completed", 0),
            dao_votes=d.get("dao_votes", 0),
            tokens_created=d.get("tokens_created", 0),
            penalties_received=d.get("penalties_received", 0),
            last_active=d.get("last_active", 0.0),
            created_at=d.get("created_at", 0.0),
        )


def _utc_day_key() -> int:
    """Return an integer key for the current UTC day (days since epoch)."""
    return int(time.time()) // 86400


class TrustManager:
    """
    Manages soulbound TRUST tokens for all addresses.

    Audit fix: Daily cap enforced — each address can earn at most
    TRUST_DAILY_CAP (50) points per UTC day. Penalties are NOT capped.
    """

    def __init__(self):
        self._profiles: Dict[bytes, TrustProfile] = {}
        self._total_score_awarded: int = 0
        self._total_penalties: int = 0
        # Daily gain tracking: address -> (day_key, total_gained_today)
        self._daily_gains: Dict[bytes, Tuple[int, int]] = {}

    def _check_daily_cap(self, address: bytes, amount: int) -> int:
        """
        Check and enforce daily TRUST cap.
        Returns the actual amount that can be awarded (may be reduced or 0).
        Penalties bypass the daily cap.
        """
        if amount <= 0:
            return amount  # Penalties are never capped

        today = _utc_day_key()
        day_key, gained_today = self._daily_gains.get(address, (0, 0))

        # Reset if new day
        if day_key != today:
            day_key = today
            gained_today = 0

        remaining = max(0, TRUST_DAILY_CAP - gained_today)
        actual = min(amount, remaining)

        if actual > 0:
            self._daily_gains[address] = (day_key, gained_today + actual)

        return actual

    def get_profile(self, address: bytes) -> TrustProfile:
        """Get or create trust profile for address."""
        if address not in self._profiles:
            self._profiles[address] = TrustProfile(
                address=address,
                created_at=time.time(),
                last_active=time.time(),
            )
        return self._profiles[address]

    def get_score(self, address: bytes) -> int:
        """Get current trust score for address."""
        return self.get_profile(address).score

    def get_level(self, address: bytes) -> TrustLevel:
        """Get current trust level for address."""
        return self.get_profile(address).level

    def get_mining_multiplier(self, address: bytes) -> float:
        """Get mining reward multiplier based on trust."""
        return self.get_profile(address).mining_multiplier

    def _award(self, address: bytes, base_amount: int, reason: str):
        """Internal: award TRUST with daily cap enforcement."""
        actual = self._check_daily_cap(address, base_amount)
        if actual > 0:
            profile = self.get_profile(address)
            profile.add_score(actual, reason)
            self._total_score_awarded += actual

    def on_block_mined(self, miner_address: bytes):
        """Called when a validator mines a block."""
        profile = self.get_profile(miner_address)
        profile.blocks_mined += 1
        self._award(miner_address, TRUST_REWARDS["block_mined"], "block_mined")

    def on_game_completed(self, player_address: bytes):
        """Called when a player completes a game."""
        profile = self.get_profile(player_address)
        profile.games_completed += 1
        self._award(player_address, TRUST_REWARDS["game_completed"], "game_completed")

    def on_dao_vote(self, voter_address: bytes):
        """Called when an address participates in DAO voting."""
        profile = self.get_profile(voter_address)
        profile.dao_votes += 1
        self._award(voter_address, TRUST_REWARDS["dao_vote"], "dao_vote")

    def on_token_created(self, creator_address: bytes):
        """Called when an address creates a new token."""
        profile = self.get_profile(creator_address)
        profile.tokens_created += 1
        self._award(creator_address, TRUST_REWARDS["token_created"], "token_created")

    def on_suspicious_tx(self, address: bytes):
        """Called when a suspicious transaction is detected. Penalties bypass daily cap."""
        profile = self.get_profile(address)
        profile.penalties_received += 1
        penalty = TRUST_PENALTIES["suspicious_tx"]
        profile.add_score(penalty, "suspicious_tx")
        self._total_penalties += abs(penalty)

    def on_quarantined_tx(self, address: bytes):
        """Called when a transaction is quarantined by AI. Penalties bypass daily cap."""
        profile = self.get_profile(address)
        profile.penalties_received += 1
        penalty = TRUST_PENALTIES["quarantined_tx"]
        profile.add_score(penalty, "quarantined_tx")
        self._total_penalties += abs(penalty)

    def on_slashed(self, address: bytes):
        """Called when a validator is slashed. Penalties bypass daily cap."""
        profile = self.get_profile(address)
        profile.penalties_received += 1
        penalty = TRUST_PENALTIES["slashed"]
        profile.add_score(penalty, "slashed")
        self._total_penalties += abs(penalty)

    def on_spam_detected(self, address: bytes):
        """Called when spam is detected from an address. Penalties bypass daily cap."""
        profile = self.get_profile(address)
        profile.penalties_received += 1
        penalty = TRUST_PENALTIES["spam_detected"]
        profile.add_score(penalty, "spam_detected")
        self._total_penalties += abs(penalty)

    def get_daily_gains(self, address: bytes) -> int:
        """Get how much TRUST an address has earned today."""
        today = _utc_day_key()
        day_key, gained = self._daily_gains.get(address, (0, 0))
        if day_key != today:
            return 0
        return gained

    def get_leaderboard(self, top_n: int = 20) -> List[dict]:
        """Get top trust scores."""
        sorted_profiles = sorted(
            self._profiles.values(),
            key=lambda p: p.score,
            reverse=True,
        )
        return [p.to_dict() for p in sorted_profiles[:top_n]]

    def get_stats(self) -> dict:
        """Get TRUST system statistics."""
        total_profiles = len(self._profiles)
        level_counts = {}
        for level in TrustLevel:
            level_counts[level.name] = sum(
                1 for p in self._profiles.values() if p.level == level
            )

        avg_score = 0
        if total_profiles > 0:
            avg_score = sum(p.score for p in self._profiles.values()) / total_profiles

        return {
            "total_profiles": total_profiles,
            "total_score_awarded": self._total_score_awarded,
            "total_penalties": self._total_penalties,
            "average_score": round(avg_score, 2),
            "level_distribution": level_counts,
            "daily_cap": TRUST_DAILY_CAP,
            "multiplier_range": {
                "min": TRUST_MULTIPLIERS[TrustLevel.NEWCOMER],
                "max": TRUST_MULTIPLIERS[TrustLevel.LEGEND],
            },
        }
