"""
Positronic - Forensic Reporting System
Generates detailed forensic reports for legal/regulatory purposes.
Provides court-admissible evidence from the blockchain.
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import IntEnum

from positronic.utils.logging import get_logger

logger = get_logger(__name__)


class ReportType(IntEnum):
    """Types of forensic reports."""
    TRANSACTION_TRACE = 1     # Full trace of a transaction
    WALLET_HISTORY = 2        # Complete wallet activity history
    FUND_FLOW = 3             # Track flow of funds between addresses
    SUSPICIOUS_ACTIVITY = 4   # SAR (Suspicious Activity Report)
    COMPLIANCE_AUDIT = 5      # Full compliance audit
    INCIDENT_REPORT = 6       # Security incident report


@dataclass
class ForensicEvidence:
    """A piece of forensic evidence."""
    evidence_type: str
    timestamp: float
    block_height: int
    tx_hash: str
    description: str
    data: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type,
            "timestamp": self.timestamp,
            "block_height": self.block_height,
            "tx_hash": self.tx_hash,
            "description": self.description,
            "data": self.data,
        }


@dataclass
class ForensicReport:
    """A complete forensic report for legal/regulatory use."""
    report_id: str
    report_type: ReportType
    created_at: float
    subject_address: str           # Address being investigated
    investigator: str = ""         # Who requested the report
    summary: str = ""
    evidence: List[ForensicEvidence] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    risk_score: float = 0.0        # Overall risk assessment (0-1)
    total_value_traced: int = 0    # Total value involved
    related_addresses: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_evidence(self, evidence: ForensicEvidence):
        """Add evidence to the report."""
        self.evidence.append(evidence)

    # Digital signatures for court-admissible evidence
    report_hash: bytes = b""
    signature: bytes = b""
    signer_pubkey: bytes = b""

    def add_finding(self, finding: str):
        """Add a finding to the report."""
        self.findings.append(finding)

    def compute_hash(self) -> bytes:
        """Compute SHA-512 hash of report content for signing."""
        import hashlib
        content = json.dumps({
            "report_id": self.report_id,
            "report_type": self.report_type.name,
            "created_at": self.created_at,
            "subject_address": self.subject_address,
            "evidence_count": len(self.evidence),
            "findings": self.findings,
            "risk_score": self.risk_score,
            "total_value_traced": self.total_value_traced,
        }, sort_keys=True, default=str)
        self.report_hash = hashlib.sha512(content.encode()).digest()
        return self.report_hash

    def sign_report(self, keypair) -> bytes:
        """Sign the report hash with Ed25519 for court-admissible evidence."""
        if not self.report_hash:
            self.compute_hash()
        self.signature = keypair.sign(self.report_hash)
        self.signer_pubkey = keypair.public_key_bytes
        return self.signature

    def verify_report_signature(self, pubkey: bytes = b"") -> bool:
        """Verify the digital signature on this report."""
        if not self.signature or not self.report_hash:
            return False
        check_key = pubkey if pubkey else self.signer_pubkey
        if not check_key:
            return False
        try:
            from positronic.crypto.keys import KeyPair
            return KeyPair.verify(check_key, self.signature, self.report_hash)
        except Exception as e:
            logger.debug("Forensic report signature verification failed: %s", e)
            return False

    def export_pdf_ready(self) -> dict:
        """Export structured dict with all proofs for PDF generation."""
        return {
            **self.to_dict(),
            "report_hash": self.report_hash.hex() if self.report_hash else "",
            "signature": self.signature.hex() if self.signature else "",
            "signer_pubkey": self.signer_pubkey.hex() if self.signer_pubkey else "",
            "signature_valid": self.verify_report_signature(),
        }

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.name,
            "created_at": self.created_at,
            "subject_address": self.subject_address,
            "investigator": self.investigator,
            "summary": self.summary,
            "evidence": [e.to_dict() for e in self.evidence],
            "findings": self.findings,
            "risk_score": self.risk_score,
            "total_value_traced": self.total_value_traced,
            "related_addresses": self.related_addresses,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Export report as JSON (for court submission)."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ForensicReporter:
    """
    Generates forensic reports for legal and regulatory compliance.
    Can trace transactions, analyze wallet histories, and produce
    court-admissible evidence from the blockchain.
    """

    def __init__(self):
        self._reports: Dict[str, ForensicReport] = {}
        self._report_counter: int = 0

    def _generate_report_id(self) -> str:
        self._report_counter += 1
        return f"POSITRONIC-FR-{int(time.time())}-{self._report_counter:04d}"

    def create_transaction_trace(self, tx_hash: str,
                                  sender: str, recipient: str,
                                  value: int, block_height: int,
                                  ai_score: float) -> ForensicReport:
        """Create a transaction trace report."""
        report = ForensicReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.TRANSACTION_TRACE,
            created_at=time.time(),
            subject_address=sender,
            summary=f"Transaction trace for {tx_hash[:16]}...",
        )

        evidence = ForensicEvidence(
            evidence_type="transaction",
            timestamp=time.time(),
            block_height=block_height,
            tx_hash=tx_hash,
            description=f"Transfer of {value} base units from {sender[:16]}... to {recipient[:16]}...",
            data={
                "sender": sender,
                "recipient": recipient,
                "value": value,
                "ai_score": ai_score,
                "block_height": block_height,
            },
        )
        report.add_evidence(evidence)
        report.total_value_traced = value
        report.related_addresses = [sender, recipient]
        report.risk_score = ai_score

        self._reports[report.report_id] = report
        return report

    def create_wallet_history(self, address: str,
                               transactions: List[dict]) -> ForensicReport:
        """Create a wallet history report."""
        report = ForensicReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.WALLET_HISTORY,
            created_at=time.time(),
            subject_address=address,
            summary=f"Complete wallet history for {address[:16]}...",
        )

        total_value = 0
        related = set()

        for tx in transactions:
            evidence = ForensicEvidence(
                evidence_type="transaction",
                timestamp=tx.get("timestamp", 0),
                block_height=tx.get("block_height", 0),
                tx_hash=tx.get("tx_hash", ""),
                description=f"TX: {tx.get('tx_type', 'unknown')}",
                data=tx,
            )
            report.add_evidence(evidence)
            total_value += tx.get("value", 0)

            if tx.get("sender"):
                related.add(tx["sender"])
            if tx.get("recipient"):
                related.add(tx["recipient"])

        report.total_value_traced = total_value
        report.related_addresses = list(related)
        report.add_finding(f"Total {len(transactions)} transactions found")
        report.add_finding(f"Total value: {total_value} base units")

        self._reports[report.report_id] = report
        return report

    def create_suspicious_activity_report(self, address: str,
                                           reason: str,
                                           ai_scores: List[float],
                                           related_txs: List[dict]) -> ForensicReport:
        """Create a Suspicious Activity Report (SAR)."""
        report = ForensicReport(
            report_id=self._generate_report_id(),
            report_type=ReportType.SUSPICIOUS_ACTIVITY,
            created_at=time.time(),
            subject_address=address,
            summary=f"Suspicious activity detected: {reason}",
        )

        avg_score = sum(ai_scores) / len(ai_scores) if ai_scores else 0
        report.risk_score = avg_score

        for tx in related_txs:
            evidence = ForensicEvidence(
                evidence_type="suspicious_transaction",
                timestamp=tx.get("timestamp", 0),
                block_height=tx.get("block_height", 0),
                tx_hash=tx.get("tx_hash", ""),
                description=f"Suspicious: {reason}",
                data=tx,
            )
            report.add_evidence(evidence)

        report.add_finding(f"Reason: {reason}")
        report.add_finding(f"Average AI risk score: {avg_score:.3f}")
        report.add_finding(f"Number of flagged transactions: {len(related_txs)}")

        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> Optional[ForensicReport]:
        return self._reports.get(report_id)

    def get_reports_for_address(self, address: str) -> List[ForensicReport]:
        return [r for r in self._reports.values()
                if r.subject_address == address]

    @property
    def total_reports(self) -> int:
        return len(self._reports)

    def get_stats(self) -> dict:
        type_counts = {}
        for r in self._reports.values():
            name = r.report_type.name
            type_counts[name] = type_counts.get(name, 0) + 1
        return {
            "total_reports": self.total_reports,
            "report_types": type_counts,
        }
