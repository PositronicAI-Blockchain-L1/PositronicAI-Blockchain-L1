"""
Positronic - AI Military Rank System
AI validators earn ranks based on experience and accuracy.
Ranks: E-1 Private through O-7 General (8 levels).
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


class AIRank(IntEnum):
    """Military-style ranks for AI validators."""
    E1_PRIVATE = 1        # سرباز - New AI, learning phase
    E2_CORPORAL = 2       # سرجوخه - Basic competence proven
    E3_SERGEANT = 3       # گروهبان - Reliable performer
    E4_STAFF_SGT = 4      # استوار - Experienced veteran
    O1_LIEUTENANT = 5     # ستوان - Junior officer, leadership
    O3_CAPTAIN = 6        # سروان - Proven leader
    O5_COLONEL = 7        # سرهنگ - Senior authority
    O7_GENERAL = 8        # ژنرال - Supreme commander


@dataclass
class RankRequirements:
    """Requirements to achieve a rank."""
    min_transactions_scored: int
    min_accuracy: float
    min_uptime_hours: int
    min_days_active: int


# Rank progression requirements
RANK_REQUIREMENTS = {
    AIRank.E1_PRIVATE: RankRequirements(0, 0.0, 0, 0),
    AIRank.E2_CORPORAL: RankRequirements(1_000, 0.70, 24, 7),
    AIRank.E3_SERGEANT: RankRequirements(10_000, 0.80, 168, 30),
    AIRank.E4_STAFF_SGT: RankRequirements(50_000, 0.85, 720, 90),
    AIRank.O1_LIEUTENANT: RankRequirements(200_000, 0.90, 2160, 180),
    AIRank.O3_CAPTAIN: RankRequirements(500_000, 0.92, 4320, 365),
    AIRank.O5_COLONEL: RankRequirements(1_000_000, 0.95, 8640, 730),
    AIRank.O7_GENERAL: RankRequirements(5_000_000, 0.97, 17520, 1460),
}

# Reward multipliers by rank
RANK_REWARD_MULTIPLIER = {
    AIRank.E1_PRIVATE: 1.0,
    AIRank.E2_CORPORAL: 1.2,
    AIRank.E3_SERGEANT: 1.5,
    AIRank.E4_STAFF_SGT: 2.0,
    AIRank.O1_LIEUTENANT: 2.5,
    AIRank.O3_CAPTAIN: 3.0,
    AIRank.O5_COLONEL: 4.0,
    AIRank.O7_GENERAL: 5.0,
}

# Rank names in Persian
RANK_NAMES_FA = {
    AIRank.E1_PRIVATE: "سرباز",
    AIRank.E2_CORPORAL: "سرجوخه",
    AIRank.E3_SERGEANT: "گروهبان",
    AIRank.E4_STAFF_SGT: "استوار",
    AIRank.O1_LIEUTENANT: "ستوان",
    AIRank.O3_CAPTAIN: "سروان",
    AIRank.O5_COLONEL: "سرهنگ",
    AIRank.O7_GENERAL: "ژنرال",
}


@dataclass
class AIValidatorProfile:
    """Profile tracking an AI validator's rank progression."""
    address: bytes
    rank: AIRank = AIRank.E1_PRIVATE
    total_scored: int = 0
    accurate_scores: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    uptime_hours: float = 0.0
    first_active: float = 0.0
    last_active: float = 0.0
    promotions: int = 0
    demotions: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.accurate_scores / self.total_scored

    @property
    def days_active(self) -> int:
        if self.first_active == 0:
            return 0
        return int((time.time() - self.first_active) / 86400)

    @property
    def rank_name(self) -> str:
        return self.rank.name

    @property
    def rank_name_fa(self) -> str:
        return RANK_NAMES_FA.get(self.rank, "نامشخص")

    @property
    def reward_multiplier(self) -> float:
        return RANK_REWARD_MULTIPLIER.get(self.rank, 1.0)

    def to_dict(self) -> dict:
        return {
            "address": self.address.hex(),
            "rank": int(self.rank),
            "rank_name": self.rank_name,
            "rank_name_fa": self.rank_name_fa,
            "total_scored": self.total_scored,
            "accuracy": self.accuracy,
            "uptime_hours": self.uptime_hours,
            "days_active": self.days_active,
            "reward_multiplier": self.reward_multiplier,
            "promotions": self.promotions,
            "demotions": self.demotions,
        }


class AIRankManager:
    """Manages AI validator rank progression."""

    def __init__(self):
        self._profiles: Dict[bytes, AIValidatorProfile] = {}

    def register(self, address: bytes) -> AIValidatorProfile:
        """Register a new AI validator at E1 rank."""
        profile = AIValidatorProfile(
            address=address,
            rank=AIRank.E1_PRIVATE,
            first_active=time.time(),
            last_active=time.time(),
        )
        self._profiles[address] = profile
        return profile

    def get_profile(self, address: bytes) -> Optional[AIValidatorProfile]:
        return self._profiles.get(address)

    def record_score(self, address: bytes, was_accurate: bool):
        """Record a scoring event and check for promotion."""
        profile = self._profiles.get(address)
        if not profile:
            return

        profile.total_scored += 1
        if was_accurate:
            profile.accurate_scores += 1
        profile.last_active = time.time()

        # Check for rank promotion
        self._check_promotion(profile)

    def update_uptime(self, address: bytes, hours: float):
        """Update uptime hours for a validator."""
        profile = self._profiles.get(address)
        if profile:
            profile.uptime_hours = hours
            self._check_promotion(profile)

    def _check_promotion(self, profile: AIValidatorProfile):
        """Check if validator qualifies for promotion."""
        current_rank = profile.rank
        if current_rank >= AIRank.O7_GENERAL:
            return  # Already max rank

        next_rank = AIRank(current_rank + 1)
        reqs = RANK_REQUIREMENTS[next_rank]

        if (profile.total_scored >= reqs.min_transactions_scored
            and profile.accuracy >= reqs.min_accuracy
            and profile.uptime_hours >= reqs.min_uptime_hours
            and profile.days_active >= reqs.min_days_active):
            profile.rank = next_rank
            profile.promotions += 1

    def check_demotion(self, address: bytes):
        """Check if validator should be demoted (accuracy dropped)."""
        profile = self._profiles.get(address)
        if not profile or profile.rank <= AIRank.E1_PRIVATE:
            return

        current_reqs = RANK_REQUIREMENTS[profile.rank]
        if profile.accuracy < current_reqs.min_accuracy * 0.9:  # 10% below requirement
            profile.rank = AIRank(profile.rank - 1)
            profile.demotions += 1

    @property
    def total_validators(self) -> int:
        return len(self._profiles)

    def get_by_rank(self, rank: AIRank) -> list:
        return [p for p in self._profiles.values() if p.rank == rank]

    def get_generals(self) -> list:
        return self.get_by_rank(AIRank.O7_GENERAL)

    def get_stats(self) -> dict:
        rank_distribution = {}
        for rank in AIRank:
            rank_distribution[rank.name] = len(self.get_by_rank(rank))
        return {
            "total_validators": self.total_validators,
            "rank_distribution": rank_distribution,
        }
