"""
AI Judge — Consensus-level AI score verification.

Judge validators re-run AI scoring on block transactions and compare
results with the proposer's scores. Significant disagreement triggers
a NIL prevote in the BFT consensus round.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Divergence threshold in basis points (200bp = 2.0%)
AI_DIVERGENCE_THRESHOLD = 200

# If more than 1/3 of transactions have score disputes, vote NIL
DISPUTE_RATIO_THRESHOLD = 1 / 3


@dataclass
class AIScoreDispute:
    """Evidence of AI score disagreement between judge and proposer."""
    tx_hash: bytes
    local_score: int       # quantized basis points (0-10000)
    block_score: int       # quantized basis points from proposer
    deviation: int         # absolute difference in basis points
    validator_address: bytes
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'tx_hash': self.tx_hash.hex(),
            'local_score': self.local_score,
            'block_score': self.block_score,
            'deviation': self.deviation,
            'validator': self.validator_address.hex(),
            'timestamp': self.timestamp,
        }


class AIJudge:
    """Judge validators re-run AI scoring to verify block proposals.

    During the prevote step of Tendermint BFT, judges independently
    score all transactions in the proposed block. If scores diverge
    beyond AI_DIVERGENCE_THRESHOLD for too many transactions, the
    judge votes NIL.
    """

    def __init__(self, ai_gate, my_address: bytes):
        self._ai_gate = ai_gate
        self._my_address = my_address
        self._dispute_history: List[AIScoreDispute] = []

    def verify_block_ai_scores(
        self,
        block,
        block_ai_scores: Dict[bytes, int],
    ) -> Tuple[bool, List[AIScoreDispute]]:
        """Re-run AI scoring on all block transactions.

        Args:
            block: The proposed Block
            block_ai_scores: Dict mapping tx_hash -> quantized score from proposer

        Returns:
            (all_ok, disputes) -- True if all scores within threshold
        """
        disputes = []

        for tx in block.transactions:
            # Skip system transactions (rewards, treasury)
            if hasattr(tx, 'tx_type') and tx.tx_type in (0x06, 0x07):  # REWARD, AI_TREASURY
                continue

            block_score = block_ai_scores.get(tx.tx_hash, 0)

            # Re-score with deterministic context
            try:
                local_result = self._ai_gate.validate_transaction(tx)
                local_score = local_result.score_quantized
            except Exception as e:
                logger.debug("AI Judge: scoring failed for tx %s: %s",
                             tx.tx_hash.hex()[:16], e)
                local_score = 0  # Default to safe if scoring fails

            if not self.compare_scores(local_score, block_score):
                dispute = AIScoreDispute(
                    tx_hash=tx.tx_hash,
                    local_score=local_score,
                    block_score=block_score,
                    deviation=abs(local_score - block_score),
                    validator_address=self._my_address,
                )
                disputes.append(dispute)
                self._dispute_history.append(dispute)

        all_ok = len(disputes) == 0
        if not all_ok:
            logger.info(
                "AI Judge: %d/%d transactions have score disputes",
                len(disputes), len(block.transactions),
            )

        return all_ok, disputes

    def compare_scores(self, local_score: int, block_score: int) -> bool:
        """True if scores are within acceptable divergence threshold."""
        return abs(local_score - block_score) <= AI_DIVERGENCE_THRESHOLD

    def should_vote_nil(
        self,
        disputes: List[AIScoreDispute],
        total_txs: int,
    ) -> bool:
        """Vote NIL if too many transactions have score disputes.

        Threshold: >1/3 of transactions disputed -> NIL vote.
        """
        if total_txs == 0:
            return False
        return len(disputes) > total_txs * DISPUTE_RATIO_THRESHOLD

    def generate_dispute(
        self,
        tx_hash: bytes,
        local_score: int,
        block_score: int,
    ) -> AIScoreDispute:
        """Create dispute evidence for a specific transaction."""
        return AIScoreDispute(
            tx_hash=tx_hash,
            local_score=local_score,
            block_score=block_score,
            deviation=abs(local_score - block_score),
            validator_address=self._my_address,
        )

    @property
    def recent_disputes(self) -> List[AIScoreDispute]:
        """Last 100 disputes for monitoring."""
        return self._dispute_history[-100:]

    @property
    def dispute_rate(self) -> float:
        """Fraction of recent verifications with disputes."""
        if not self._dispute_history:
            return 0.0
        recent = self._dispute_history[-100:]
        return sum(1 for d in recent if d.deviation > AI_DIVERGENCE_THRESHOLD) / len(recent)
