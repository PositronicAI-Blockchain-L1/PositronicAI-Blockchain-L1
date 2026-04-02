"""
Positronic - RWA Compliance Engine (Phase 30)
Orchestrates KYC checks via DID credentials and compliance scoring.

Works alongside PRC-3643 tokens to enforce transfer restrictions
based on KYC level, jurisdiction, and AI-driven anomaly detection.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.constants import RWA_MIN_KYC_LEVEL


@dataclass
class KYCRecord:
    """KYC information extracted from DID credentials."""
    address: bytes
    kyc_level: int = 0          # 0=none, 1=basic, 2=standard, 3=enhanced
    jurisdiction: str = ""      # ISO 3166-1 alpha-2
    verified_at: float = 0.0
    expires_at: float = 0.0     # 0 = never expires
    credential_id: str = ""     # Link back to DID credential

    @property
    def is_valid(self) -> bool:
        if self.kyc_level < 1:
            return False
        if self.expires_at > 0 and time.time() > self.expires_at:
            return False
        return True


class ComplianceEngine:
    """Orchestrates compliance checks for RWA transfers.

    Integrates with DID system for KYC verification and provides
    a unified interface for PRC-3643 tokens to check compliance.
    """

    def __init__(self):
        self._kyc_records: Dict[bytes, KYCRecord] = {}
        self._blocked_jurisdictions: Dict[str, List[str]] = {}  # token_id -> [jurisdictions]
        self._transfer_log: List[dict] = []
        self._total_checks: int = 0
        self._total_passed: int = 0
        self._total_failed: int = 0

    # ---- KYC Management ----

    def register_kyc(
        self,
        address: bytes,
        kyc_level: int,
        jurisdiction: str,
        credential_id: str = "",
        expires_at: float = 0.0,
    ) -> KYCRecord:
        """Register or update KYC info for an address."""
        record = KYCRecord(
            address=address,
            kyc_level=max(0, min(3, kyc_level)),
            jurisdiction=jurisdiction.upper(),
            verified_at=time.time(),
            expires_at=expires_at,
            credential_id=credential_id,
        )
        self._kyc_records[address] = record
        return record

    def register_kyc_from_credential(
        self,
        address: bytes,
        credential,
    ) -> Optional[KYCRecord]:
        """Extract KYC info from a DID Credential object.

        Expects claim_type='kyc' and claim_data with 'level' and 'jurisdiction'.
        """
        if not credential or not credential.is_valid:
            return None
        if credential.claim_type != "kyc":
            return None

        level = credential.claim_data.get("level", 0)
        jurisdiction = credential.claim_data.get("jurisdiction", "")

        return self.register_kyc(
            address=address,
            kyc_level=level,
            jurisdiction=jurisdiction,
            credential_id=credential.credential_id,
            expires_at=credential.expires_at,
        )

    def get_kyc(self, address: bytes) -> Optional[KYCRecord]:
        """Get KYC record for an address."""
        return self._kyc_records.get(address)

    def get_kyc_level(self, address: bytes) -> int:
        """Get effective KYC level (0 if expired or not registered)."""
        record = self._kyc_records.get(address)
        if not record or not record.is_valid:
            return 0
        return record.kyc_level

    def get_jurisdiction(self, address: bytes) -> str:
        """Get jurisdiction for an address."""
        record = self._kyc_records.get(address)
        if not record:
            return ""
        return record.jurisdiction

    # ---- Compliance Check ----

    def check_transfer_compliance(
        self,
        token_id: str,
        sender: bytes,
        recipient: bytes,
        amount: int,
        token=None,
    ) -> Tuple[bool, str]:
        """Full compliance check for an RWA transfer.

        Returns (passed, reason_if_failed).
        """
        self._total_checks += 1

        # Get KYC info
        sender_kyc = self.get_kyc_level(sender)
        recipient_kyc = self.get_kyc_level(recipient)
        sender_jurisdiction = self.get_jurisdiction(sender)
        recipient_jurisdiction = self.get_jurisdiction(recipient)

        # KYC level check
        min_level = RWA_MIN_KYC_LEVEL
        if token:
            min_level = token.min_kyc_level

        if sender_kyc < min_level:
            self._total_failed += 1
            return False, f"Sender KYC level {sender_kyc} < required {min_level}"

        if recipient_kyc < min_level:
            self._total_failed += 1
            return False, f"Recipient KYC level {recipient_kyc} < required {min_level}"

        # Use token's own compliance if available
        if token:
            from positronic.tokens.prc3643 import ComplianceResult
            result = token.check_compliance(
                sender, recipient, amount,
                sender_kyc, recipient_kyc,
                sender_jurisdiction, recipient_jurisdiction,
            )
            if result != ComplianceResult.PASS:
                self._total_failed += 1
                return False, f"Token compliance failed: {result.name}"

        self._total_passed += 1

        # Log the check
        self._transfer_log.append({
            "token_id": token_id,
            "sender": sender.hex(),
            "recipient": recipient.hex(),
            "amount": amount,
            "passed": True,
            "timestamp": time.time(),
        })

        return True, "PASS"

    # ---- Jurisdiction Management ----

    def block_jurisdiction(self, token_id: str, jurisdiction: str):
        """Block a jurisdiction for a specific token."""
        blocked = self._blocked_jurisdictions.setdefault(token_id, [])
        j = jurisdiction.upper()
        if j not in blocked:
            blocked.append(j)

    def unblock_jurisdiction(self, token_id: str, jurisdiction: str):
        """Unblock a jurisdiction for a specific token."""
        blocked = self._blocked_jurisdictions.get(token_id, [])
        j = jurisdiction.upper()
        if j in blocked:
            blocked.remove(j)

    def is_jurisdiction_blocked(self, token_id: str, jurisdiction: str) -> bool:
        """Check if a jurisdiction is blocked for a token."""
        blocked = self._blocked_jurisdictions.get(token_id, [])
        return jurisdiction.upper() in blocked

    # ---- Statistics ----

    def get_stats(self) -> dict:
        return {
            "total_kyc_records": len(self._kyc_records),
            "valid_kyc_records": sum(
                1 for r in self._kyc_records.values() if r.is_valid
            ),
            "total_checks": self._total_checks,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "pass_rate": (
                self._total_passed / self._total_checks
                if self._total_checks > 0 else 0.0
            ),
        }
