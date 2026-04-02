"""
Positronic - Fallback Validator (Phase 15)
Heuristic-based validation when AI models are degraded or disabled.
Provides a safety net so the network is never completely unprotected.

Rules are simple, deterministic, and consensus-safe — no neural models
or floating-point indeterminacy.
"""

import time
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from positronic.constants import (
    FALLBACK_GAS_SPIKE_MULTIPLIER,
    FALLBACK_BALANCE_RATIO_LIMIT,
    FALLBACK_SENDER_BURST_LIMIT,
    FALLBACK_BURST_WINDOW_BLOCKS,
    FALLBACK_GAS_HISTORY_SIZE,
    AI_ACCEPT_THRESHOLD,
    AI_QUARANTINE_THRESHOLD,
)


@dataclass
class FallbackResult:
    """Result from the fallback validator."""
    score: float
    reasons: List[str] = field(default_factory=list)
    tier: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "reasons": self.reasons,
            "tier": self.tier,
        }


class FallbackValidator:
    """
    Heuristic-based validation when AI is degraded or disabled.

    Provides 5 simple rules that catch obvious attacks without
    requiring trained neural models:

    1. Gas spike detection    — reject gas > 10x median
    2. Balance drain          — quarantine >80% of balance
    3. New account + high val — quarantine nonce=0 + big value
    4. Sender burst           — reject >5 TXs in 10 blocks
    5. MEV pattern            — quarantine same-recipient + high gas
    """

    def __init__(self):
        self._gas_history: List[int] = []
        self._median_gas: int = 21_000
        self._sender_activity: Dict[bytes, List[int]] = {}  # sender → [block_heights]
        self._recent_txs: List[dict] = []  # Last 100 TXs for MEV detection
        self._max_recent = 100

        # Stats
        self.total_validated: int = 0
        self.flagged: int = 0

    def validate(
        self,
        sender: bytes,
        value: int,
        gas_price: int,
        gas_limit: int,
        nonce: int,
        balance: int,
        recipient: bytes,
        is_contract: bool,
        block_height: int,
    ) -> FallbackResult:
        """
        Score a transaction using simple, deterministic rules.

        Returns FallbackResult with score in [0, 1].
        """
        score = 0.0
        reasons = []

        # Rule 1: Gas spike — gas price > 10x rolling median
        if self._median_gas > 0 and gas_price > self._median_gas * FALLBACK_GAS_SPIKE_MULTIPLIER:
            score = max(score, 0.90)
            reasons.append("gas_spike")

        # Rule 2: Balance drain — sending >80% of balance
        if balance > 0:
            ratio = value / balance
            if ratio > FALLBACK_BALANCE_RATIO_LIMIT:
                score = max(score, 0.86)
                reasons.append("balance_drain")

        # Rule 3: New account + high value
        if nonce == 0 and balance > 0 and value > balance * 0.3:
            score = max(score, 0.86)
            reasons.append("new_account_high_value")

        # Rule 4: Sender burst — too many TXs from same sender recently
        recent_count = self._count_recent_activity(sender, block_height)
        if recent_count >= FALLBACK_SENDER_BURST_LIMIT:
            score = max(score, 0.90)
            reasons.append("sender_burst")

        # Rule 5: MEV pattern — same recipient + significantly higher gas
        mev_score = self._detect_mev_pattern(sender, recipient, gas_price)
        if mev_score > 0:
            score = max(score, mev_score)
            reasons.append("mev_pattern")

        self.total_validated += 1
        if score >= AI_ACCEPT_THRESHOLD:
            self.flagged += 1

        return FallbackResult(score=min(score, 1.0), reasons=reasons)

    def _count_recent_activity(self, sender: bytes, block_height: int) -> int:
        """Count sender's transactions in the recent block window."""
        history = self._sender_activity.get(sender, [])
        cutoff = block_height - FALLBACK_BURST_WINDOW_BLOCKS
        return sum(1 for h in history if h >= cutoff)

    def _detect_mev_pattern(
        self, sender: bytes, recipient: bytes, gas_price: int
    ) -> float:
        """Detect MEV/front-running: same recipient + much higher gas."""
        if not self._recent_txs:
            return 0.0

        for recent in reversed(self._recent_txs[-50:]):
            if recent["recipient"] == recipient and recent["sender"] != sender:
                if gas_price > 0 and recent["gas_price"] > 0:
                    ratio = gas_price / recent["gas_price"]
                    if ratio > 5.0:
                        return 0.90
                    if ratio > 2.0:
                        return 0.87
        return 0.0

    def record_transaction(
        self,
        sender: bytes,
        recipient: bytes,
        gas_price: int,
        block_height: int,
    ):
        """Record a transaction for future pattern analysis."""
        # Update gas history
        self._gas_history.append(gas_price)
        if len(self._gas_history) > FALLBACK_GAS_HISTORY_SIZE:
            self._gas_history = self._gas_history[-FALLBACK_GAS_HISTORY_SIZE:]
        self._update_median_gas()

        # Update sender activity
        if sender not in self._sender_activity:
            self._sender_activity[sender] = []
        self._sender_activity[sender].append(block_height)
        # Keep only recent blocks
        cutoff = block_height - FALLBACK_BURST_WINDOW_BLOCKS * 2
        self._sender_activity[sender] = [
            h for h in self._sender_activity[sender] if h >= cutoff
        ]

        # LRU eviction for sender tracking
        if len(self._sender_activity) > 10_000:
            oldest = next(iter(self._sender_activity))
            del self._sender_activity[oldest]

        # Record recent TXs for MEV detection
        self._recent_txs.append({
            "sender": sender,
            "recipient": recipient,
            "gas_price": gas_price,
            "block_height": block_height,
        })
        if len(self._recent_txs) > self._max_recent:
            self._recent_txs = self._recent_txs[-self._max_recent:]

    def _update_median_gas(self):
        """Update rolling median gas price."""
        if not self._gas_history:
            return
        sorted_gas = sorted(self._gas_history)
        mid = len(sorted_gas) // 2
        if len(sorted_gas) % 2 == 0:
            self._median_gas = (sorted_gas[mid - 1] + sorted_gas[mid]) // 2
        else:
            self._median_gas = sorted_gas[mid]

    def get_stats(self) -> dict:
        return {
            "total_validated": self.total_validated,
            "flagged": self.flagged,
            "flagged_rate": self.flagged / max(self.total_validated, 1),
            "median_gas": self._median_gas,
            "tracked_senders": len(self._sender_activity),
            "gas_history_size": len(self._gas_history),
        }
