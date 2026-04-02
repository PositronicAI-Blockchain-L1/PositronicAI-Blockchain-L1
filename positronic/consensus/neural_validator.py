"""
Positronic - Neural Validator Node (NVN)
Specialized validator node that runs AI scoring models.
NVNs receive 30% of transaction fees for their AI computation work.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger
from positronic.ai.meta_model import AIValidationGate, ValidationResult

logger = get_logger(__name__)
from positronic.ai.trainer import AITrainer
from positronic.core.transaction import Transaction, TxStatus
from positronic.constants import (
    AI_ACCEPT_THRESHOLD,
    AI_QUARANTINE_THRESHOLD,
    MIN_STAKE,
)


@dataclass
class NVNScore:
    """A score submitted by an NVN for a transaction."""
    tx_hash: bytes
    score: float
    nvn_address: bytes
    model_version: int
    timestamp: float = field(default_factory=time.time)
    signature: bytes = b""


@dataclass
class NVNStats:
    """Statistics for a Neural Validator Node."""
    address: bytes
    total_scored: int = 0
    accurate_scores: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    avg_score_time_ms: float = 0.0
    total_rewards: int = 0
    uptime_start: float = field(default_factory=time.time)

    @property
    def confirmed_count(self) -> int:
        return self.accurate_scores + self.false_positives + self.false_negatives

    @property
    def accuracy(self) -> float:
        if self.confirmed_count == 0:
            return 1.0  # No confirmed scores yet, assume good
        return self.accurate_scores / self.confirmed_count

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.uptime_start) / 3600


class NeuralValidator:
    """
    Neural Validator Node (NVN) logic.

    NVNs are specialized validators that run the AI Validation Gate
    to score transactions. They receive 30% of fees for this work.

    Requirements:
    - Must stake at least NVN_MIN_STAKE ASF
    - Must run the approved AI model version
    - Scores must agree with consensus within tolerance
    - Slashed for consistently inaccurate scores
    """

    def __init__(self, address: bytes, ai_gate: AIValidationGate = None):
        self.address = address
        self.ai_gate = ai_gate or AIValidationGate()
        self.trainer = AITrainer(
            self.ai_gate.anomaly_detector,
            self.ai_gate.feature_extractor,
        )
        self.stats = NVNStats(address=address)
        self.pending_scores: Dict[bytes, NVNScore] = {}
        self._active = True

    @property
    def model_version(self) -> int:
        return self.ai_gate.model_version

    @property
    def is_active(self) -> bool:
        return self._active

    def score_transaction(self, tx: Transaction) -> NVNScore:
        """Score a transaction using the AI gate."""
        start = time.time()

        result = self.ai_gate.validate_transaction(tx)

        elapsed_ms = (time.time() - start) * 1000

        score = NVNScore(
            tx_hash=tx.tx_hash,
            score=result.final_score,
            nvn_address=self.address,
            model_version=self.model_version,
            timestamp=time.time(),
        )

        # Update stats
        self.stats.total_scored += 1
        n = self.stats.total_scored
        self.stats.avg_score_time_ms = (
            (self.stats.avg_score_time_ms * (n - 1) + elapsed_ms) / n
        )

        self.pending_scores[tx.tx_hash] = score
        return score

    def score_batch(self, transactions: List[Transaction]) -> List[NVNScore]:
        """Score a batch of transactions."""
        return [self.score_transaction(tx) for tx in transactions]

    def on_score_confirmed(self, tx_hash: bytes, consensus_score: float):
        """Called when a score is confirmed by consensus."""
        if tx_hash not in self.pending_scores:
            return

        our_score = self.pending_scores.pop(tx_hash)
        tolerance = 0.15

        if abs(our_score.score - consensus_score) <= tolerance:
            self.stats.accurate_scores += 1
        else:
            # Determine type of inaccuracy
            if our_score.score > consensus_score:
                self.stats.false_positives += 1
            else:
                self.stats.false_negatives += 1

    def record_reward(self, amount: int):
        """Record a reward received."""
        self.stats.total_rewards += amount

    def train_on_feedback(self, tx: Transaction, confirmed_status: TxStatus):
        """Update AI models based on confirmed transaction outcomes."""
        try:
            features = self.ai_gate.feature_extractor.extract(tx)
            self.trainer.add_training_data([tx])
        except Exception as e:
            logger.debug("Failed to train on feedback: %s", e)

    def validate_game_proof(self, proof) -> dict:
        """Validate a game proof using AI heuristics.

        Checks if game result is plausible by analyzing patterns:
        - Score-to-time ratio
        - Coins-to-level ratio
        - Statistical deviation from player history

        Args:
            proof: GameProof object with game_result attribute.

        Returns:
            dict with 'valid', 'risk_score', and 'nvn_address'.
        """
        result = proof.game_result
        risk_score = 0.0

        # Check score-to-time ratio (too high score in too little time)
        if result.time_taken > 0:
            score_per_second = result.score / result.time_taken
            if score_per_second > 500:  # Suspiciously fast scoring
                risk_score += 0.3

        # Check coins-to-level ratio
        if result.level_completed > 0:
            coins_per_level = result.coins_collected / result.level_completed
            if coins_per_level > 100:  # Unrealistic coin collection
                risk_score += 0.2

        # Check if no_damage with high score and many enemies
        if result.no_damage and result.enemies_defeated > 50:
            risk_score += 0.15  # Suspicious: no damage + many enemies

        # Cap risk score
        risk_score = min(risk_score, 1.0)

        return {
            "valid": risk_score < 0.85,
            "risk_score": round(risk_score, 4),
            "nvn_address": self.address.hex(),
        }

    def deactivate(self):
        """Deactivate this NVN."""
        self._active = False

    def activate(self):
        """Activate this NVN."""
        self._active = True

    def get_status(self) -> dict:
        return {
            "address": self.address.hex(),
            "active": self._active,
            "model_version": self.model_version,
            "total_scored": self.stats.total_scored,
            "accuracy": round(self.stats.accuracy, 4),
            "avg_score_time_ms": round(self.stats.avg_score_time_ms, 2),
            "false_positive_rate": (
                self.stats.false_positives / max(1, self.stats.total_scored)
            ),
            "total_rewards": self.stats.total_rewards,
            "uptime_hours": round(self.stats.uptime_hours, 2),
        }


class NVNRegistry:
    """
    Registry of all Neural Validator Nodes in the network.
    Manages NVN registration, score aggregation, and reward distribution.
    """

    def __init__(self):
        self.nvns: Dict[bytes, NeuralValidator] = {}
        self.score_buffer: Dict[bytes, List[NVNScore]] = {}
        self.min_nvn_scores = 1  # Minimum NVN scores needed for consensus

    def register(self, address: bytes, ai_gate: AIValidationGate = None) -> NeuralValidator:
        """Register a new NVN."""
        nvn = NeuralValidator(address, ai_gate)
        self.nvns[address] = nvn
        return nvn

    def unregister(self, address: bytes):
        """Remove an NVN from the registry."""
        self.nvns.pop(address, None)

    def get_active_nvns(self) -> List[NeuralValidator]:
        """Get all active NVNs."""
        return [nvn for nvn in self.nvns.values() if nvn.is_active]

    def submit_score(self, score: NVNScore):
        """Submit an NVN score for aggregation."""
        if score.tx_hash not in self.score_buffer:
            self.score_buffer[score.tx_hash] = []
        self.score_buffer[score.tx_hash].append(score)

    def get_consensus_score(self, tx_hash: bytes) -> Optional[float]:
        """
        Get the consensus AI score for a transaction.
        Uses weighted median of NVN scores.
        """
        scores = self.score_buffer.get(tx_hash, [])
        if len(scores) < self.min_nvn_scores:
            return None

        # Weighted median: NVNs with higher accuracy get more weight
        weighted_scores = []
        for s in scores:
            nvn = self.nvns.get(s.nvn_address)
            weight = nvn.stats.accuracy if nvn else 0.5
            weighted_scores.append((s.score, weight))

        weighted_scores.sort(key=lambda x: x[0])
        total_weight = sum(w for _, w in weighted_scores)
        if total_weight == 0:
            return scores[0].score

        cumulative = 0.0
        for score, weight in weighted_scores:
            cumulative += weight / total_weight
            if cumulative >= 0.5:
                return score

        return weighted_scores[-1][0]

    def finalize_scores(self, tx_hash: bytes):
        """Finalize scores and notify NVNs of the consensus result."""
        consensus = self.get_consensus_score(tx_hash)
        if consensus is None:
            return

        scores = self.score_buffer.pop(tx_hash, [])
        for s in scores:
            nvn = self.nvns.get(s.nvn_address)
            if nvn:
                nvn.on_score_confirmed(tx_hash, consensus)

    def distribute_rewards(self, total_nvn_reward: int):
        """Distribute NVN rewards proportional to scores submitted."""
        active = self.get_active_nvns()
        if not active:
            return {}

        # Weight by accuracy and number of scores
        weights = {}
        total_weight = 0
        for nvn in active:
            w = nvn.stats.accuracy * max(1, nvn.stats.total_scored)
            weights[nvn.address] = w
            total_weight += w

        if total_weight == 0:
            return {}

        rewards = {}
        for addr, w in weights.items():
            share = int(total_nvn_reward * w / total_weight)
            if share > 0:
                rewards[addr] = share
                self.nvns[addr].record_reward(share)

        return rewards

    @property
    def active_count(self) -> int:
        return len(self.get_active_nvns())

    @property
    def total_count(self) -> int:
        return len(self.nvns)
