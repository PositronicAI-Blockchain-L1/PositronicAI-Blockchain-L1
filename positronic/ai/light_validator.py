"""
Positronic - Light Validator Module

A lightweight AI scoring module for desktop app users who have staked ASF.
Runs only the TAD (Transaction Anomaly Detector) model instead of all 4 models,
providing attestation capability and anomaly reporting with minimal resource usage.

Modes:
- Full Validator (server): 4 AI models, block production, full scoring (~200MB RAM)
- Light Validator (desktop + stake): 1 AI model (TAD), attestation, light scoring (~50MB RAM)
- Sync Node (desktop, no stake): just sync and relay

Light Validators:
- DO attest blocks (confirms they're valid)
- DO run TAD scoring on incoming transactions
- DO report anomalies to peers
- DO receive attestation rewards (30% share, pro-rata)
- DO NOT produce blocks
"""

import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnomalyReport:
    """An anomaly detected by a Light Validator."""
    tx_hash: str
    score: float
    reason: str
    timestamp: float = field(default_factory=time.time)
    reporter_address: str = ""


class LightValidator:
    """
    Lightweight AI scoring for desktop stakers.

    Uses only the TAD (Transaction Anomaly Detector) from the full AI gate,
    skipping the MSAD, SCRA, and ESG models to reduce RAM/CPU usage.

    The Light Validator scores incoming transactions and reports anomalies
    but never produces blocks.
    """

    # Anomaly threshold: scores above this are reported to peers
    ANOMALY_THRESHOLD = 0.65

    # Maximum pending anomaly reports before oldest are discarded
    MAX_PENDING_REPORTS = 500

    # How often to run light validation on pending transactions (seconds)
    SCORING_INTERVAL = 2.0

    def __init__(self, ai_gate=None):
        """
        Initialize the Light Validator.

        Args:
            ai_gate: The full AIValidationGate instance from the blockchain.
                     We only use its anomaly_detector (TAD) and feature_extractor.
                     If None, creates standalone TAD components.
        """
        if ai_gate is not None:
            # Reuse the TAD and feature extractor from the full gate
            self.anomaly_detector = ai_gate.anomaly_detector
            self.feature_extractor = ai_gate.feature_extractor
            self._shared_gate = True
        else:
            # Standalone mode: create only TAD components (minimal RAM)
            from positronic.ai.anomaly_detector import Autoencoder
            from positronic.ai.feature_extractor import FeatureExtractor
            self.anomaly_detector = Autoencoder()
            self.feature_extractor = FeatureExtractor()
            self._shared_gate = False

        # Anomaly reports pending broadcast to peers
        self._pending_reports: List[AnomalyReport] = []

        # Stats
        self.total_scored: int = 0
        self.anomalies_detected: int = 0
        self.blocks_attested: int = 0
        self.active: bool = True

        logger.info(
            "Light Validator initialized (TAD-only, shared_gate=%s)",
            self._shared_gate,
        )

    def score_transaction(
        self,
        tx,
        sender_account=None,
    ) -> Tuple[float, bool]:
        """
        Score a transaction using only the TAD model.

        Args:
            tx: Transaction object to score.
            sender_account: The sender's account state (for balance/nonce checks).

        Returns:
            (score, is_anomaly) where score is 0.0-1.0 and is_anomaly is True
            if the score exceeds the anomaly threshold.
        """
        if not self.active:
            return 0.0, False

        try:
            # Extract features (same as full gate)
            features = self.feature_extractor.extract(tx, sender_account)

            # Score using TAD only (skip MSAD, SCRA, ESG)
            score = self.anomaly_detector.compute_anomaly_score(features)
            score = min(max(score, 0.0), 1.0)

            self.total_scored += 1
            is_anomaly = score >= self.ANOMALY_THRESHOLD

            if is_anomaly:
                self.anomalies_detected += 1
                logger.info(
                    "Light Validator anomaly: tx=%s score=%.3f",
                    tx.hash_hex[:16] if hasattr(tx, 'hash_hex') else "?",
                    score,
                )

            return score, is_anomaly

        except Exception as e:
            logger.debug("Light Validator scoring error: %s", e)
            return 0.0, False

    def report_anomaly(
        self,
        tx_hash: str,
        score: float,
        reason: str,
        reporter_address: str = "",
    ) -> Optional[AnomalyReport]:
        """
        Create an anomaly report for broadcast to peers.

        Args:
            tx_hash: Hash of the anomalous transaction.
            score: The TAD anomaly score (0.0-1.0).
            reason: Human-readable reason for the anomaly flag.
            reporter_address: This light validator's address.

        Returns:
            The AnomalyReport, or None if reporting is disabled.
        """
        if not self.active:
            return None

        report = AnomalyReport(
            tx_hash=tx_hash,
            score=score,
            reason=reason,
            reporter_address=reporter_address,
        )

        self._pending_reports.append(report)

        # Cap pending reports to avoid unbounded memory growth
        if len(self._pending_reports) > self.MAX_PENDING_REPORTS:
            self._pending_reports = self._pending_reports[-self.MAX_PENDING_REPORTS:]

        logger.debug(
            "Anomaly report queued: tx=%s score=%.3f reason=%s",
            tx_hash[:16], score, reason,
        )
        return report

    def get_pending_reports(self) -> List[AnomalyReport]:
        """Get and clear all pending anomaly reports for broadcast."""
        reports = list(self._pending_reports)
        self._pending_reports.clear()
        return reports

    def score_block_transactions(
        self,
        block,
        state_manager=None,
    ) -> Tuple[int, int, List[AnomalyReport]]:
        """
        Score all transactions in a block using light (TAD-only) validation.

        Args:
            block: The block whose transactions to score.
            state_manager: State manager for looking up sender accounts.

        Returns:
            (total_scored, anomalies_found, reports) tuple.
        """
        scored = 0
        anomalies = 0
        reports: List[AnomalyReport] = []

        for tx in block.transactions:
            sender_account = None
            if state_manager and hasattr(tx, 'sender'):
                try:
                    sender_account = state_manager.get_account(tx.sender)
                except Exception:
                    pass

            score, is_anomaly = self.score_transaction(tx, sender_account)
            scored += 1

            if is_anomaly:
                anomalies += 1
                tx_hash = tx.hash_hex if hasattr(tx, 'hash_hex') else tx.hash.hex()
                reason = self._classify_anomaly(score, tx, sender_account)
                report = self.report_anomaly(
                    tx_hash=tx_hash,
                    score=score,
                    reason=reason,
                )
                if report:
                    reports.append(report)

        return scored, anomalies, reports

    def _classify_anomaly(self, score: float, tx, sender_account) -> str:
        """Generate a human-readable reason for the anomaly."""
        reasons = []

        if sender_account:
            balance = sender_account.balance
            if balance > 0 and tx.value / balance > 0.9:
                reasons.append("high-value-ratio")
            if sender_account.nonce < 3 and tx.value > 0:
                reasons.append("new-account-large-tx")
            if sender_account.ai_reputation < 0.5:
                reasons.append("low-reputation")

        if hasattr(tx, 'data') and tx.data and len(tx.data) > 5000:
            reasons.append("large-payload")

        if score >= 0.85:
            reasons.append("high-tad-score")
        elif score >= 0.65:
            reasons.append("moderate-tad-score")

        return ",".join(reasons) if reasons else "tad-anomaly"

    def on_block_attested(self):
        """Record that we attested a block."""
        self.blocks_attested += 1

    def get_stats(self) -> Dict:
        """Get Light Validator statistics."""
        return {
            "mode": "light_validator",
            "active": self.active,
            "total_scored": self.total_scored,
            "anomalies_detected": self.anomalies_detected,
            "blocks_attested": self.blocks_attested,
            "pending_reports": len(self._pending_reports),
            "shared_gate": self._shared_gate,
            "anomaly_threshold": self.ANOMALY_THRESHOLD,
        }
