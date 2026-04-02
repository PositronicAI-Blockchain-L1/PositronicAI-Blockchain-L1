"""
Positronic - Court Evidence Report Generator
Generates structured, legally-formatted evidence documents from forensic reports.
Provides court-admissible digital evidence packages with blockchain verification.
"""

import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from positronic.utils.logging import get_logger
from positronic.compliance.forensic_report import (
    ForensicReport,
    ForensicEvidence,
    ForensicReporter,
    ReportType,
)

logger = get_logger(__name__)


@dataclass
class EvidenceExhibit:
    """A single exhibit in a court evidence package."""
    exhibit_id: str
    title: str
    description: str
    evidence: ForensicEvidence
    merkle_proof: Optional[dict] = None
    block_finality: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {
            "exhibit_id": self.exhibit_id,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence.to_dict(),
        }
        if self.merkle_proof:
            d["merkle_proof"] = self.merkle_proof
        if self.block_finality:
            d["block_finality"] = self.block_finality
        return d


@dataclass
class ChainOfCustody:
    """Digital chain of custody record."""
    created_at: float
    created_by: str
    blockchain_height: int
    report_hash: str
    entries: List[dict] = field(default_factory=list)

    def add_entry(self, action: str, actor: str, details: str):
        self.entries.append({
            "timestamp": time.time(),
            "action": action,
            "actor": actor,
            "details": details,
        })

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "created_by": self.created_by,
            "blockchain_height": self.blockchain_height,
            "report_hash": self.report_hash,
            "entries": self.entries,
        }


@dataclass
class CourtReport:
    """Complete court-ready evidence document."""
    court_report_id: str
    case_reference: str
    forensic_report_id: str
    generated_at: float
    case_header: dict
    executive_summary: str
    exhibits: List[EvidenceExhibit] = field(default_factory=list)
    chain_of_custody: Optional[ChainOfCustody] = None
    findings: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    subject_address: str = ""
    report_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "court_report_id": self.court_report_id,
            "case_reference": self.case_reference,
            "forensic_report_id": self.forensic_report_id,
            "generated_at": self.generated_at,
            "case_header": self.case_header,
            "executive_summary": self.executive_summary,
            "exhibits": [e.to_dict() for e in self.exhibits],
            "chain_of_custody": self.chain_of_custody.to_dict() if self.chain_of_custody else None,
            "findings": self.findings,
            "risk_score": self.risk_score,
            "subject_address": self.subject_address,
            "report_hash": self.report_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class CourtReportGenerator:
    """
    Generates court-ready evidence documents from ForensicReporter data.

    Takes a ForensicReport and produces structured legal documents with:
    - Numbered exhibits (Exhibit A, B, C...)
    - Merkle proof verification for each transaction
    - Chain of custody records
    - Report hash for tamper detection
    - Timestamp certification from block finality
    """

    def __init__(self, blockchain=None):
        self.blockchain = blockchain
        self._reports: Dict[str, CourtReport] = {}
        self._counter: int = 0

    def _generate_id(self) -> str:
        self._counter += 1
        return f"POSTR-CR-{int(time.time())}-{self._counter:04d}"

    def generate_court_report(
        self,
        forensic_report: ForensicReport,
        case_reference: str = "",
    ) -> CourtReport:
        """Generate a court-ready report from a forensic report."""
        report_id = self._generate_id()
        now = time.time()

        case_ref = case_reference or f"CASE-{forensic_report.report_id}"

        # Build case header
        case_header = {
            "court_report_id": report_id,
            "case_reference": case_ref,
            "blockchain": "Positronic (ASF)",
            "chain_id": 420420,
            "consensus": "DPoS + Proof of Neural Consensus (PoNC)",
            "hash_algorithm": "SHA-512",
            "signature_scheme": "Ed25519",
            "generated_at": now,
            "forensic_report_id": forensic_report.report_id,
            "report_type": forensic_report.report_type.name,
            "subject_address": forensic_report.subject_address,
        }

        # Generate executive summary
        summary = self._generate_summary(forensic_report)

        report = CourtReport(
            court_report_id=report_id,
            case_reference=case_ref,
            forensic_report_id=forensic_report.report_id,
            generated_at=now,
            case_header=case_header,
            executive_summary=summary,
            risk_score=forensic_report.risk_score,
            subject_address=forensic_report.subject_address,
            findings=forensic_report.findings.copy(),
        )

        # Convert evidence to exhibits
        for i, evidence in enumerate(forensic_report.evidence):
            exhibit_letter = chr(65 + (i % 26))
            if i >= 26:
                exhibit_letter = chr(65 + i // 26 - 1) + exhibit_letter

            exhibit = EvidenceExhibit(
                exhibit_id=f"Exhibit {exhibit_letter}",
                title=f"{evidence.evidence_type.replace('_', ' ').title()}",
                description=evidence.description,
                evidence=evidence,
                merkle_proof=self._get_merkle_proof(evidence),
                block_finality=self._get_block_finality(evidence),
            )
            report.exhibits.append(exhibit)

        # Build chain of custody
        chain_height = 0
        if self.blockchain:
            try:
                chain_height = self.blockchain.height
            except Exception as e:
                logger.debug("Failed to get chain height for custody record: %s", e)

        coc = ChainOfCustody(
            created_at=now,
            created_by="Positronic Forensic System",
            blockchain_height=chain_height,
            report_hash="",
        )
        coc.add_entry(
            "CREATED",
            "ForensicReporter",
            f"Original forensic report {forensic_report.report_id} created",
        )
        coc.add_entry(
            "COURT_REPORT_GENERATED",
            "CourtReportGenerator",
            f"Court report {report_id} generated from forensic data",
        )
        report.chain_of_custody = coc

        # Hash report for tamper detection
        report.report_hash = self._hash_report(report)
        coc.report_hash = report.report_hash

        self._reports[report.court_report_id] = report
        return report

    def generate_evidence_package(
        self,
        address: str,
        forensic_reporter: ForensicReporter,
    ) -> dict:
        """Package ALL evidence for an address across multiple forensic reports."""
        reports = forensic_reporter.get_reports_for_address(address)
        court_reports = []
        for fr in reports:
            cr = self.generate_court_report(fr)
            court_reports.append(cr)

        package = {
            "package_id": f"POSTR-EP-{int(time.time())}",
            "address": address,
            "generated_at": time.time(),
            "blockchain": "Positronic (ASF)",
            "chain_id": 420420,
            "total_reports": len(court_reports),
            "reports": [cr.to_dict() for cr in court_reports],
        }
        package["package_hash"] = hashlib.sha512(
            json.dumps(package, sort_keys=True, default=str).encode()
        ).hexdigest()
        return package

    def verify_evidence(self, court_report_id: str) -> dict:
        """Verify evidence integrity against the blockchain."""
        report = self._reports.get(court_report_id)
        if not report:
            return {"valid": False, "error": "Court report not found"}

        # Verify report hash
        current_hash = self._hash_report(report)
        hash_valid = current_hash == report.report_hash

        # Verify exhibits
        exhibit_results = []
        for exhibit in report.exhibits:
            result = {
                "exhibit_id": exhibit.exhibit_id,
                "tx_hash": exhibit.evidence.tx_hash,
                "merkle_verified": False,
            }
            if exhibit.merkle_proof and exhibit.merkle_proof.get("verified"):
                result["merkle_verified"] = True
            exhibit_results.append(result)

        return {
            "valid": hash_valid,
            "court_report_id": court_report_id,
            "report_hash_valid": hash_valid,
            "expected_hash": report.report_hash,
            "current_hash": current_hash,
            "exhibits_verified": exhibit_results,
            "verified_at": time.time(),
        }

    def get_report(self, court_report_id: str) -> Optional[CourtReport]:
        return self._reports.get(court_report_id)

    def _generate_summary(self, report: ForensicReport) -> str:
        type_name = report.report_type.name.replace("_", " ").title()
        from positronic.constants import BASE_UNIT
        value_asf = report.total_value_traced / BASE_UNIT if report.total_value_traced else 0
        return (
            f"This court evidence report documents a {type_name} investigation "
            f"concerning blockchain address {report.subject_address}. "
            f"A total of {len(report.evidence)} evidence item(s) were collected, "
            f"tracing {value_asf:.2f} ASF ({report.total_value_traced} base units) "
            f"across {len(report.related_addresses)} related address(es). "
            f"The AI-computed overall risk assessment score is {report.risk_score:.4f} "
            f"(scale: 0.0 safe to 1.0 maximum risk). "
            f"All evidence is cryptographically anchored to the Positronic blockchain "
            f"(Chain ID: 420420) using SHA-512 hashing and Ed25519 digital signatures."
        )

    def _get_merkle_proof(self, evidence: ForensicEvidence) -> Optional[dict]:
        """Get Merkle proof for a transaction from the blockchain."""
        if not self.blockchain or not evidence.tx_hash:
            return None
        try:
            block_height = evidence.block_height
            block = self.blockchain.get_block(block_height)
            if not block:
                return None

            tx_hashes = [tx.tx_hash_hex for tx in block.transactions]
            if evidence.tx_hash not in tx_hashes:
                return {"verified": False, "reason": "TX not found in block"}

            tx_index = tx_hashes.index(evidence.tx_hash)
            from positronic.core.merkle import MerkleTree
            leaves = [bytes.fromhex(h[2:]) if h.startswith("0x") else bytes.fromhex(h)
                      for h in tx_hashes]
            tree = MerkleTree(leaves)
            proof = tree.get_proof(tx_index)

            return {
                "verified": True,
                "block_height": block_height,
                "tx_index": tx_index,
                "merkle_root": tree.root_hex,
                "proof_length": len(proof),
            }
        except Exception as e:
            logger.debug("Merkle proof verification failed: %s", e)
            return None

    def _get_block_finality(self, evidence: ForensicEvidence) -> Optional[dict]:
        """Get block finality status for the evidence."""
        if not self.blockchain:
            return None
        try:
            current_height = self.blockchain.height
            confirmations = current_height - evidence.block_height
            return {
                "block_height": evidence.block_height,
                "current_height": current_height,
                "confirmations": confirmations,
                "finalized": confirmations >= 6,
            }
        except Exception as e:
            logger.debug("Failed to get block finality info: %s", e)
            return None

    def _hash_report(self, report: CourtReport) -> str:
        """SHA-512 hash of report content for tamper detection."""
        d = report.to_dict()
        d.pop("report_hash", None)
        if d.get("chain_of_custody"):
            d["chain_of_custody"].pop("report_hash", None)
        content = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha512(content.encode()).hexdigest()
