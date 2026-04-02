"""
Positronic - Meta Model (AI Validation Gate)
Combines scores from all 4 AI models into a single risk score.
This is the heart of Proof of Neural Consensus (PoNC).

Decision flow (quantized integer scoring for cross-node determinism):
  score_q < 8450  -> ACCEPTED  (clearly safe)
  8450-9550       -> QUARANTINED (suspicious or in dead zone)
  >= 9550         -> REJECTED (clearly malicious)

All consensus-critical decisions use integer basis points (0-10000)
to eliminate IEEE 754 floating-point divergence across CPU architectures.
Neural models still use float64 internally, but the final decision boundary
is quantized with a ±50bp dead zone to absorb cross-platform float errors.
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from positronic.utils.logging import get_logger
from positronic.core.transaction import Transaction, TxType, TxStatus

logger = get_logger(__name__)
from positronic.core.account import Account
from positronic.ai.feature_extractor import FeatureExtractor, TransactionFeatures
from positronic.ai.anomaly_detector import Autoencoder
from positronic.ai.mev_detector import MEVDetector
from positronic.ai.contract_analyzer import ContractAnalyzer
from positronic.ai.stability_guardian import StabilityGuardian
from positronic.ai.engine.random import DeterministicContext
from positronic.constants import (
    AI_ACCEPT_THRESHOLD,
    AI_QUARANTINE_THRESHOLD,
    AI_KILL_SWITCH_FP_RATE,
    AI_MODEL_VERSION,
    AI_CONSENSUS_SEED,
    AI_MAX_LATENCY_MS,
    AI_TIERED_LIGHT_VALUE,
    AI_TIERED_LIGHT_REPUTATION,
    AI_TIERED_MEDIUM_VALUE,
    KS_LEVEL_0_MAX_FP,
    KS_LEVEL_1_MAX_FP,
    KS_LEVEL_2_MAX_FP,
    KS_LEVEL_1_ACCEPT,
    KS_LEVEL_2_ACCEPT,
    MODEL_VETO_THRESHOLD,
    MODEL_VETO_MIN_SCORE,
    HARD_NEGATIVE_MIN_SCORE,
    HARD_NEGATIVE_MAX_SCORE,
    XAI_MAX_EXPLANATION_LEN,
    XAI_ATTENTION_THRESHOLD,
    # Quantized integer scoring constants
    AI_SCORE_SCALE,
    AI_SCORE_DEAD_ZONE,
    AI_ACCEPT_THRESHOLD_Q,
    AI_QUARANTINE_THRESHOLD_Q,
    AI_WEIGHT_TAD_Q,
    AI_WEIGHT_MSAD_Q,
    AI_WEIGHT_SCRA_Q,
    AI_WEIGHT_ESG_Q,
    KS_LEVEL_1_ACCEPT_Q,
    KS_LEVEL_2_ACCEPT_Q,
    MODEL_VETO_THRESHOLD_Q,
    MODEL_VETO_MIN_SCORE_Q,
)
from positronic.ai.fallback_validator import FallbackValidator, FallbackResult


# ===== Quantized Scoring Layer (Cross-Node Determinism) =====
# Neural models produce float64 scores that may differ by ~1e-4 to ~1e-7
# across CPU architectures (Intel vs AMD vs ARM) due to SIMD, FMA, and
# BLAS library differences. The quantization layer converts float scores
# to integer basis points BEFORE any consensus-critical comparison.

# Integer weights for weighted sum (sum = 100)
_WEIGHTS_Q = {"tad": AI_WEIGHT_TAD_Q, "msad": AI_WEIGHT_MSAD_Q,
              "scra": AI_WEIGHT_SCRA_Q, "esg": AI_WEIGHT_ESG_Q}


def quantize_score(score: float) -> int:
    """Quantize a 0.0-1.0 float score to integer basis points (0-10000).

    Uses int() truncation which is deterministic on all platforms.
    The maximum float error (~1e-4) is absorbed by the dead zone (±50bp).
    """
    if score <= 0.0:
        return 0
    if score >= 1.0:
        return AI_SCORE_SCALE
    return int(score * AI_SCORE_SCALE)


def quantized_weighted_sum(scores: Dict[str, float]) -> int:
    """Compute weighted sum using integer arithmetic only.

    Each float score is first quantized to int, then multiplied by its
    integer weight. Division by weight sum (100) uses integer division.
    This eliminates all float arithmetic from the consensus path.
    """
    total = 0
    for key, weight in _WEIGHTS_Q.items():
        total += quantize_score(scores.get(key, 0.0)) * weight
    return total // 100  # Integer division by sum of weights


def classify_with_dead_zone(
    score_q: int,
    accept_q: int,
    quarantine_q: int,
    dead_zone: int = AI_SCORE_DEAD_ZONE,
) -> TxStatus:
    """Classify score using integer thresholds with dead zone.

    If the quantized score falls within ±dead_zone of either threshold,
    the conservative path (QUARANTINED) is chosen. This prevents forks
    caused by different CPUs producing slightly different float scores.

    Cost: ~1% of borderline transactions get conservatively quarantined.
    Benefit: Zero consensus forks from float divergence.
    """
    if score_q < accept_q - dead_zone:
        return TxStatus.ACCEPTED
    if score_q >= quarantine_q + dead_zone:
        return TxStatus.REJECTED
    return TxStatus.QUARANTINED


@dataclass
class ValidationResult:
    """Result of AI validation for a transaction."""
    tx_hash: bytes
    final_score: float
    status: TxStatus
    model_version: int
    component_scores: Dict[str, float]
    timestamp: float
    explanation: str = ""
    score_quantized: int = 0  # Integer basis points (0-10000) used for consensus

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash.hex(),
            "final_score": self.final_score,
            "score_quantized": self.score_quantized,
            "status": self.status.name,
            "model_version": self.model_version,
            "component_scores": self.component_scores,
            "timestamp": self.timestamp,
            "explanation": self.explanation,
        }


class AIValidationGate:
    """
    The main AI Validation Gate for Positronic.
    Orchestrates all 4 AI models and produces a final risk score.

    Model weights (dynamically adjusted):
    - TAD (Anomaly Detector):    30%
    - MSAD (MEV Detector):       25%
    - SCRA (Contract Analyzer):  25%
    - ESG (Stability Guardian):  20%

    Note: Consensus-critical scoring uses quantized integer thresholds
    (AI_ACCEPT_THRESHOLD_Q, AI_QUARANTINE_THRESHOLD_Q) from constants.py
    for cross-node determinism. Optional: when running with NodeConfig,
    pass config.ai.accept_threshold and config.ai.quarantine_threshold
    into the gate for non-consensus use (e.g., RPC queries, logging).
    """

    def __init__(self, ai_config=None):
        """
        Initialize the AI Validation Gate.

        Args:
            ai_config: Optional AIConfig instance from NodeConfig. When provided,
                       accept_threshold and quarantine_threshold override the display
                       values used for RPC queries and logging. Consensus-critical
                       decisions ALWAYS use quantized integer constants from constants.py.
        """
        self.feature_extractor = FeatureExtractor()
        self.anomaly_detector = Autoencoder()
        self.mev_detector = MEVDetector()
        self.contract_analyzer = ContractAnalyzer()
        self.stability_guardian = StabilityGuardian()

        # Display thresholds (non-consensus — used for RPC queries, logging, dashboards).
        # When ai_config provides overrides, use them; otherwise fall back to constants.
        if ai_config is not None and ai_config.accept_threshold is not None:
            self.display_accept_threshold = ai_config.accept_threshold
        else:
            self.display_accept_threshold = AI_ACCEPT_THRESHOLD

        if ai_config is not None and ai_config.quarantine_threshold is not None:
            self.display_quarantine_threshold = ai_config.quarantine_threshold
        else:
            self.display_quarantine_threshold = AI_QUARANTINE_THRESHOLD

        # Model weights (sum to 1.0)
        self.weights = {
            "tad": 0.30,   # Transaction Anomaly Detector
            "msad": 0.25,  # MEV/Sandwich Attack Detector
            "scra": 0.25,  # Smart Contract Risk Analyzer
            "esg": 0.20,   # Economic Stability Guardian
        }

        # Performance tracking
        self.total_scored: int = 0
        self.accepted: int = 0
        self.quarantined: int = 0
        self.rejected: int = 0
        self.false_positives: int = 0
        self.model_version: int = AI_MODEL_VERSION

        # Tiered scoring stats
        self.tier_counts: Dict[str, int] = {"light": 0, "medium": 0, "full": 0}

        # Score history for metrics dashboard (last 1000 scores)
        self._recent_scores: List[float] = []
        self._max_recent_scores = 1000

        # Kill switch state (Phase 15: graduated 0-3 levels)
        self.ai_enabled: bool = True
        self.kill_switch_triggered: bool = False
        self.kill_switch_level: int = 0  # 0=normal, 1=elevated, 2=degraded, 3=fallback
        self._kill_switch_disabled_at: float = 0.0  # Security fix: cooldown timer
        self._kill_switch_cooldown: float = 300.0  # 5 minutes cooldown before re-enable

        # Fallback validator (Phase 15: heuristic rules when AI disabled)
        self.fallback_validator = FallbackValidator()

        # Neural meta-ensemble (initialized under deterministic seed so
        # all validator nodes get identical initial weights)
        try:
            from positronic.ai.models.meta_ensemble import MetaEnsemble
            with DeterministicContext(seed=AI_CONSENSUS_SEED):
                self._meta_ensemble = MetaEnsemble()
        except ImportError:
            self._meta_ensemble = None

        # Phase 16: Online learner reference (set externally via set_online_learner)
        self._online_learner = None

    def bootstrap(self, seed: int = 42, count: int = 500) -> None:
        """Bootstrap-train AI models with synthetic normal transaction data.

        Call during node startup to provide baseline anomaly detection
        before real transaction data is available. After bootstrap, the TAD
        autoencoder can detect anomalous patterns via reconstruction error.
        """
        from positronic.ai.training.bootstrap import bootstrap_gate
        bootstrap_gate(self, seed=seed, count=count)

    def validate_transaction(
        self,
        tx: Transaction,
        sender_account: Optional[Account] = None,
        pending_txs: List[Transaction] = None,
        mempool_size: int = 0,
    ) -> ValidationResult:
        """
        Run a transaction through the full AI validation pipeline.
        Returns a ValidationResult with final score and status.
        """
        # Phase 15: Graduated kill-switch — Level 3 uses fallback heuristics
        if not self.ai_enabled or self.kill_switch_level >= 3:
            fb_result = self.fallback_validator.validate(
                sender=tx.sender,
                value=tx.value,
                gas_price=tx.gas_price,
                gas_limit=tx.gas_limit,
                nonce=tx.nonce if hasattr(tx, 'nonce') else 0,
                balance=sender_account.balance if sender_account else 0,
                recipient=tx.recipient,
                is_contract=tx.tx_type in (TxType.CONTRACT_CREATE, TxType.CONTRACT_CALL),
                block_height=0,
            )
            fb_score = fb_result.score
            fb_score_q = quantize_score(fb_score)
            fb_status = classify_with_dead_zone(
                fb_score_q, AI_ACCEPT_THRESHOLD_Q, AI_QUARANTINE_THRESHOLD_Q
            )
            tx.ai_score = fb_score
            tx.status = fb_status
            self.total_scored += 1
            if fb_status == TxStatus.ACCEPTED:
                self.accepted += 1
            elif fb_status == TxStatus.QUARANTINED:
                self.quarantined += 1
            else:
                self.rejected += 1
            return ValidationResult(
                tx_hash=tx.tx_hash,
                final_score=fb_score,
                score_quantized=fb_score_q,
                status=fb_status,
                model_version=self.model_version,
                component_scores={"fallback": True, "reasons": fb_result.reasons},
                timestamp=time.time(),
                explanation=f"Fallback validation (KS level {self.kill_switch_level}): {', '.join(fb_result.reasons) or 'clean'}",
            )

        # Reject obviously invalid transactions before AI scoring
        if tx.value < 0:
            tx.ai_score = 1.0
            tx.status = TxStatus.REJECTED
            self.total_scored += 1
            self.rejected += 1
            return ValidationResult(
                tx_hash=tx.tx_hash, final_score=1.0,
                status=TxStatus.REJECTED, model_version=self.model_version,
                component_scores={"rejection_reason": "negative_value"},
                timestamp=time.time(),
                explanation="REJECTED: Negative transaction value (attack attempt)",
            )

        # Reject transactions where sender has insufficient balance
        if sender_account is not None and tx.value > 0:
            sender_balance = getattr(sender_account, 'balance', 0)
            if tx.value > sender_balance and sender_balance >= 0:
                # Value exceeds balance — likely attack or error
                overdraft_ratio = tx.value / max(sender_balance, 1)
                if overdraft_ratio > 10:
                    # Extreme overdraft (>10x balance) — reject
                    tx.ai_score = 0.95
                    tx.status = TxStatus.REJECTED
                    self.total_scored += 1
                    self.rejected += 1
                    return ValidationResult(
                        tx_hash=tx.tx_hash, final_score=0.95,
                        status=TxStatus.REJECTED, model_version=self.model_version,
                        component_scores={"rejection_reason": "extreme_overdraft",
                                          "overdraft_ratio": overdraft_ratio},
                        timestamp=time.time(),
                        explanation=f"REJECTED: Value {tx.value} exceeds balance {sender_balance} by {overdraft_ratio:.0f}x",
                    )
                elif overdraft_ratio > 1:
                    # Moderate overdraft — quarantine for review
                    tx.ai_score = 0.55
                    tx.status = TxStatus.QUARANTINED
                    self.total_scored += 1
                    self.quarantined += 1
                    return ValidationResult(
                        tx_hash=tx.tx_hash, final_score=0.55,
                        status=TxStatus.QUARANTINED, model_version=self.model_version,
                        component_scores={"rejection_reason": "overdraft",
                                          "overdraft_ratio": overdraft_ratio},
                        timestamp=time.time(),
                        explanation=f"QUARANTINED: Value {tx.value} exceeds balance {sender_balance} by {overdraft_ratio:.1f}x",
                    )

        # System transactions always pass
        if tx.tx_type in (TxType.REWARD, TxType.AI_TREASURY):
            return ValidationResult(
                tx_hash=tx.tx_hash,
                final_score=0.0,
                status=TxStatus.ACCEPTED,
                model_version=self.model_version,
                component_scores={"system_tx": True},
                timestamp=time.time(),
                explanation="System transaction - auto-accepted",
            )

        # Extract features
        start_time = time.time()
        features = self.feature_extractor.extract(
            tx, sender_account, mempool_size
        )

        # Determine scoring tier based on transaction risk profile.
        # Light:  small value + trusted sender    → TAD only (fastest)
        # Medium: moderate value or new sender    → TAD + ESG
        # Full:   large value, contracts, or risky → all 4 models
        tier = self._determine_tier(tx, features)
        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1

        # Phase 15: At kill-switch Level 2, force TAD+ESG only (skip MSAD/SCRA)
        ks_degraded = self.kill_switch_level >= 2

        scores = {}

        # 1. TAD - Anomaly Detection (always runs)
        scores["tad"] = self.anomaly_detector.compute_anomaly_score(features)

        if tier in ("medium", "full"):
            # 4. ESG - Network Stability
            scores["esg"] = self.stability_guardian.get_transaction_risk_modifier()
        else:
            scores["esg"] = 0.0

        if tier == "full" and not ks_degraded:
            # 2. MSAD - MEV Detection
            scores["msad"] = self.mev_detector.analyze_transaction(
                tx, pending_txs or []
            )
            # 3. SCRA - Contract Risk
            scores["scra"] = self.contract_analyzer.analyze_transaction(tx)
        else:
            scores["msad"] = 0.0
            scores["scra"] = 0.0

        # If TAD alone flags high risk in light/medium tier, escalate to full
        if tier != "full" and not ks_degraded and scores["tad"] > AI_ACCEPT_THRESHOLD:
            scores["msad"] = self.mev_detector.analyze_transaction(
                tx, pending_txs or []
            )
            scores["scra"] = self.contract_analyzer.analyze_transaction(tx)
            if tier == "light":
                scores["esg"] = self.stability_guardian.get_transaction_risk_modifier()
            tier = "full"  # escalated

        # Try meta-ensemble first, fall back to static weights.
        # Wrapped in DeterministicContext so all nodes produce identical
        # scores for the same transaction (consensus-critical).
        if self._meta_ensemble is not None and self._meta_ensemble.is_ready:
            tx_context = {
                'tx_type': int(tx.tx_type),
                'value_log': features.value_log,
                'sender_nonce': features.sender_nonce,
                'sender_reputation': features.sender_reputation,
                'is_contract': features.is_contract_interaction,
                'gas_price_ratio': features.gas_price_ratio,
            }
            try:
                with DeterministicContext(seed=AI_CONSENSUS_SEED):
                    final_score = self._meta_ensemble.score(scores, tx_context)
            except Exception as e:
                # Fall back to static weights if meta-ensemble inference fails.
                # This prevents a single model error from crashing the entire
                # validation pipeline and ensures block production continues.
                logger.debug("Meta-ensemble inference failed, falling back to static weights: %s", e)
                final_score = sum(
                    scores[key] * self.weights[key] for key in self.weights
                )
        else:
            # Static weighted combination
            final_score = sum(
                scores[key] * self.weights[key] for key in self.weights
            )

        # Latency safety: if scoring took too long, trigger kill switch
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > AI_MAX_LATENCY_MS:
            self._check_kill_switch()

        # Apply dynamic adjustments (float — for display/logging)
        final_score = self._apply_adjustments(final_score, tx, features)

        # Clamp to [0, 1]
        final_score = min(max(final_score, 0.0), 1.0)

        # ===== QUANTIZED CONSENSUS DECISION (integer arithmetic only) =====
        # All nodes MUST agree on the same status. Float scores may differ
        # by ~1e-4 across CPUs, so we quantize to integer basis points
        # and use a dead zone around thresholds for deterministic consensus.

        # 1. Quantize the weighted sum from float scores
        score_q = quantized_weighted_sum(scores)

        # 2. Apply adjustments in integer arithmetic
        score_q = self._apply_adjustments_q(score_q, features)

        # 3. Model veto in integer arithmetic
        for key in ("tad", "msad", "scra", "esg"):
            if quantize_score(scores.get(key, 0.0)) > MODEL_VETO_THRESHOLD_Q:
                score_q = max(score_q, MODEL_VETO_MIN_SCORE_Q)
                break

        # 3b. Nonce gap detection: tx.nonce far from account nonce is
        #     a strong replay/sequence attack signal
        if sender_account is not None and hasattr(tx, 'nonce'):
            nonce_gap = abs(tx.nonce - sender_account.nonce)
            if nonce_gap > 10:
                score_q = max(score_q, MODEL_VETO_MIN_SCORE_Q)

        # 4. Kill-switch adjusts accept threshold (integer)
        accept_q = AI_ACCEPT_THRESHOLD_Q
        if self.kill_switch_level == 1:
            accept_q = KS_LEVEL_1_ACCEPT_Q
        elif self.kill_switch_level == 2:
            accept_q = KS_LEVEL_2_ACCEPT_Q

        # 5. Classify with dead zone (DETERMINISTIC — no floats)
        status = classify_with_dead_zone(
            score_q, accept_q, AI_QUARANTINE_THRESHOLD_Q
        )

        if status == TxStatus.ACCEPTED:
            self.accepted += 1
        elif status == TxStatus.QUARANTINED:
            self.quarantined += 1
        else:
            self.rejected += 1

        self.total_scored += 1

        # Track recent scores for metrics dashboard
        self._recent_scores.append(final_score)
        if len(self._recent_scores) > self._max_recent_scores:
            self._recent_scores = self._recent_scores[-self._max_recent_scores:]

        # Update transaction
        tx.ai_score = final_score
        tx.ai_model_version = self.model_version
        tx.status = status

        # Update feature extractor stats
        self.feature_extractor.update_stats(tx)

        # Phase 16: Feed account data to contract analyzer for GAT
        if sender_account:
            self.contract_analyzer.update_account_data(
                address=tx.sender,
                nonce=sender_account.nonce if hasattr(sender_account, 'nonce') else 0,
                balance=sender_account.balance if hasattr(sender_account, 'balance') else 0,
                code_size=len(sender_account.code) if hasattr(sender_account, 'code') and sender_account.code else 0,
            )

        # Phase 16: Feed hard negatives to online learner
        if (self._online_learner is not None
                and HARD_NEGATIVE_MIN_SCORE <= final_score <= HARD_NEGATIVE_MAX_SCORE):
            try:
                import numpy as np
                fv = np.array(features.to_vector(), dtype=np.float32)
                self._online_learner.record_hard_negative(fv, final_score)
            except Exception as e:
                logger.debug("Hard negative logging failed (non-critical): %s", e)

        # Check kill switch conditions
        self._check_kill_switch()

        # Generate explanation
        explanation = self._generate_explanation(scores, final_score, status)

        return ValidationResult(
            tx_hash=tx.tx_hash,
            final_score=final_score,
            score_quantized=score_q,
            status=status,
            model_version=self.model_version,
            component_scores=scores,
            timestamp=time.time(),
            explanation=explanation,
        )

    def _apply_adjustments(
        self,
        score: float,
        tx: Transaction,
        features: TransactionFeatures,
    ) -> float:
        """Apply dynamic adjustments to the final score."""
        adjusted = score

        # Reputation bonus: trusted senders get lower scores
        if features.sender_reputation > 0.9 and features.sender_nonce > 50:
            adjusted *= 0.8  # 20% reduction for trusted senders

        # New account penalty
        if features.sender_nonce == 0:
            adjusted = min(adjusted + 0.05, 1.0)

        # Large value transactions get extra scrutiny
        if features.sender_balance_ratio > 0.8:
            adjusted = min(adjusted + 0.1, 1.0)

        return adjusted

    def _apply_adjustments_q(
        self,
        score_q: int,
        features: "TransactionFeatures",
    ) -> int:
        """Apply dynamic adjustments using integer arithmetic only.

        Integer equivalents of _apply_adjustments():
          ×0.8  → ×80 // 100
          +0.05 → +500 basis points
          +0.1  → +1000 basis points
        """
        adjusted = score_q

        # Reputation bonus: trusted senders get lower scores
        # sender_reputation > 0.9 → quantized > 9000
        if (quantize_score(features.sender_reputation) > 9000
                and features.sender_nonce > 50):
            adjusted = adjusted * 80 // 100  # 20% reduction (integer division)

        # New account penalty
        if features.sender_nonce == 0:
            adjusted = min(adjusted + 500, AI_SCORE_SCALE)  # +0.05

        # Large value transactions get extra scrutiny
        # sender_balance_ratio > 0.8 → quantized > 8000
        if quantize_score(features.sender_balance_ratio) > 8000:
            adjusted = min(adjusted + 1000, AI_SCORE_SCALE)  # +0.1

        return adjusted

    def _determine_tier(
        self,
        tx: Transaction,
        features: TransactionFeatures,
    ) -> str:
        """
        Determine scoring tier for a transaction.

        Returns:
            "light"  - TAD only (small value + trusted sender)
            "medium" - TAD + ESG (moderate value or newer sender)
            "full"   - All 4 models (large value, contracts, or new accounts)
        """
        # Full scoring for contract interactions (always risky)
        if features.is_contract_interaction:
            return "full"

        # Full scoring for new accounts (nonce 0)
        if features.sender_nonce == 0:
            return "full"

        # Full scoring for large transactions
        if tx.value >= AI_TIERED_MEDIUM_VALUE:
            return "full"

        # Light scoring for small value + trusted sender
        if (tx.value < AI_TIERED_LIGHT_VALUE
                and features.sender_reputation >= AI_TIERED_LIGHT_REPUTATION
                and features.sender_nonce > 10):
            return "light"

        # Everything else gets medium
        return "medium"

    def _generate_explanation(
        self,
        scores: Dict[str, float],
        final_score: float,
        status: TxStatus,
    ) -> str:
        """Generate a human-readable explanation (Phase 21: XAI).

        Includes per-model scores with weights, top signal identification,
        and cross-attention correlations when available.
        """
        model_labels = {
            "tad": "TAD(anomaly)",
            "msad": "MSAD(MEV)",
            "scra": "SCRA(contract)",
            "esg": "ESG(stability)",
        }

        # Per-model breakdown with weights
        model_parts = []
        for key in ["tad", "msad", "scra", "esg"]:
            raw = scores.get(key, 0.0)
            w = self.weights[key]
            model_parts.append(f"{key.upper()}:{raw:.2f}({int(w*100)}%)")

        # Top signal by weighted contribution
        top_key = max(
            ["tad", "msad", "scra", "esg"],
            key=lambda k: scores.get(k, 0) * self.weights.get(k, 0),
        )
        top_label = model_labels.get(top_key, top_key)
        top_score = scores.get(top_key, 0)

        explanation = f"Risk {final_score:.2f} ({status.name}). {', '.join(model_parts)}."
        if top_score > 0.1:
            explanation += f" Top signal: {top_label} at {top_score:.2f}."

        # Cross-attention correlations
        if self._meta_ensemble is not None:
            try:
                correlations = self._meta_ensemble.get_attention_correlations()
                if correlations:
                    top_corrs = [
                        (pair, val) for pair, val in correlations.items()
                        if val > XAI_ATTENTION_THRESHOLD
                    ]
                    if top_corrs:
                        top_corrs.sort(key=lambda x: x[1], reverse=True)
                        corr_str = ", ".join(f"{p} {v:.2f}" for p, v in top_corrs[:2])
                        explanation += f" Attention: {corr_str}."
            except Exception as e:
                logger.debug("Attention correlation XAI failed: %s", e)

        return explanation[:XAI_MAX_EXPLANATION_LEN]

    def _check_kill_switch(self):
        """
        Phase 15: Graduated kill-switch with 4 levels.

        Level 0 (Normal):   FP < 3%  — full AI scoring
        Level 1 (Elevated): FP 3-5%  — raise accept threshold to 0.90
        Level 2 (Degraded): FP 5-8%  — only TAD+ESG, threshold 0.92
        Level 3 (Fallback): FP > 8%  — disable AI, use heuristic rules
        """
        if self.total_scored < 100:
            return  # Not enough data yet

        total = max(self.total_scored, 1)
        fp_rate = self.false_positives / total

        # Determine level using graduated thresholds
        if fp_rate < KS_LEVEL_0_MAX_FP:
            new_level = 0
        elif fp_rate < KS_LEVEL_1_MAX_FP:
            new_level = 1
        elif fp_rate < KS_LEVEL_2_MAX_FP:
            new_level = 2
        else:
            new_level = 3

        self.kill_switch_level = new_level

        # Only fully disable AI at level 3
        if new_level >= 3:
            self.kill_switch_triggered = True
            self.ai_enabled = False
            self._kill_switch_disabled_at = time.time()  # Security fix: record for cooldown
        else:
            # Ensure AI stays enabled at levels 0-2
            if not self.kill_switch_triggered:
                self.ai_enabled = True

    def set_online_learner(self, learner):
        """Phase 16: Set the online learner reference for hard negative feedback.

        Args:
            learner: OnlineLearner instance.
        """
        self._online_learner = learner

    def get_thresholds(self) -> dict:
        """Return current threshold configuration for RPC and dashboard display.

        Returns display thresholds (from config or constants fallback) and
        the immutable quantized consensus thresholds.
        """
        return {
            "accept_threshold": self.display_accept_threshold,
            "quarantine_threshold": self.display_quarantine_threshold,
            "consensus_accept_q": AI_ACCEPT_THRESHOLD_Q,
            "consensus_quarantine_q": AI_QUARANTINE_THRESHOLD_Q,
            "score_scale": AI_SCORE_SCALE,
            "dead_zone": AI_SCORE_DEAD_ZONE,
        }

    def report_false_positive(self, tx_hash: bytes):
        """Report that a quarantined/rejected TX was actually legitimate."""
        self.false_positives += 1
        self._check_kill_switch()

    def train_models(self, confirmed_txs: List[Transaction]):
        """
        Train AI models on confirmed (accepted) transactions.
        Called periodically with recently confirmed transactions.
        """
        if not confirmed_txs:
            return

        features_batch = []
        for tx in confirmed_txs:
            features = self.feature_extractor.extract(tx)
            features_batch.append(features.to_vector())

        self.anomaly_detector.train_step(features_batch)

    def reset_kill_switch(self) -> bool:
        """Manually reset the kill switch (requires governance vote).

        Security fix: enforces a cooldown period after the kill switch
        was triggered, preventing rapid on/off oscillation attacks.
        Returns True if reset succeeded, False if cooldown is active.
        """
        if self._kill_switch_disabled_at > 0:
            elapsed = time.time() - self._kill_switch_disabled_at
            if elapsed < self._kill_switch_cooldown:
                return False  # Cooldown not expired

        self.kill_switch_triggered = False
        self.ai_enabled = True
        self.kill_switch_level = 0
        self.false_positives = 0
        self._kill_switch_disabled_at = 0.0
        return True

    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights (requires governance vote).

        Security fix: rejects negative weights that could invert model scores.
        """
        if any(w < 0 for w in new_weights.values()):
            raise ValueError("All weights must be non-negative")
        total = sum(new_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        self.weights.update(new_weights)

    def _harmonic_weighted_mean(
        self, scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """
        Phase 15: Harmonic mean is more sensitive to high outlier scores.
        Used in FULL tier for better attack detection.
        """
        adjusted = {k: max(v, 0.001) for k, v in scores.items()}
        numerator = sum(weights.values())
        denominator = sum(
            weights[k] / adjusted[k] for k in weights if k in adjusted
        )
        return numerator / denominator if denominator > 0 else 0.0

    def get_kill_switch_status(self) -> dict:
        """Get detailed kill-switch status."""
        total = max(self.total_scored, 1)
        return {
            "level": self.kill_switch_level,
            "level_name": ["normal", "elevated", "degraded", "fallback"][
                min(self.kill_switch_level, 3)
            ],
            "ai_enabled": self.ai_enabled,
            "kill_switch_triggered": self.kill_switch_triggered,
            "false_positive_rate": self.false_positives / total,
            "false_positives": self.false_positives,
            "total_scored": self.total_scored,
            "fallback_stats": self.fallback_validator.get_stats(),
        }

    def get_stats(self) -> dict:
        # Compute score histogram bins for dashboard
        histogram = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
                     "0.6-0.8": 0, "0.8-1.0": 0}
        avg_score = 0.0
        if self._recent_scores:
            avg_score = sum(self._recent_scores) / len(self._recent_scores)
            for s in self._recent_scores:
                if s < 0.2:
                    histogram["0.0-0.2"] += 1
                elif s < 0.4:
                    histogram["0.2-0.4"] += 1
                elif s < 0.6:
                    histogram["0.4-0.6"] += 1
                elif s < 0.8:
                    histogram["0.6-0.8"] += 1
                else:
                    histogram["0.8-1.0"] += 1

        return {
            "model_version": self.model_version,
            "ai_enabled": self.ai_enabled,
            "kill_switch_triggered": self.kill_switch_triggered,
            "kill_switch_level": self.kill_switch_level,
            "total_scored": self.total_scored,
            "accepted": self.accepted,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "false_positives": self.false_positives,
            "false_positive_rate": (
                self.false_positives / max(self.total_scored, 1)
            ),
            "avg_score": round(avg_score, 4),
            "score_histogram": histogram,
            "recent_scores_count": len(self._recent_scores),
            "weights": self.weights,
            "thresholds": self.get_thresholds(),
            "tier_counts": self.tier_counts,
            "anomaly_detector": self.anomaly_detector.get_stats(),
            "mev_detector": self.mev_detector.get_stats(),
            "contract_analyzer": self.contract_analyzer.get_stats(),
            "stability_guardian": self.stability_guardian.get_stats(),
            "neural_engine": {
                "meta_ensemble_ready": self._meta_ensemble.is_ready if self._meta_ensemble else False,
                "meta_ensemble_steps": self._meta_ensemble._training_steps if self._meta_ensemble else 0,
            },
        }
