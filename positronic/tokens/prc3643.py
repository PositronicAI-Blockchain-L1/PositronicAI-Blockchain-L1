"""
Positronic - PRC-3643 Security Token Standard (Phase 30)
Regulated token for Real-World Assets with compliance hooks.

Extends PRC-20 with:
- KYC/AML compliance checks before every transfer
- Jurisdiction-based transfer restrictions
- Transfer cooldowns for large transfers
- Issuer controls (freeze, force-transfer for legal compliance)
- Dividend distribution capability
"""

import time
import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from positronic.constants import (
    BASE_UNIT,
    RWA_MAX_HOLDERS,
    RWA_TRANSFER_COOLDOWN,
    RWA_LARGE_TRANSFER_THRESHOLD,
)


class AssetType(IntEnum):
    """Type of real-world asset backing the token."""
    REAL_ESTATE = 0
    EQUITY = 1
    COMMODITY = 2
    BOND = 3
    ART = 4


class TokenStatus(IntEnum):
    """Lifecycle of an RWA token."""
    PENDING = 0
    AI_REVIEWING = 1
    AI_APPROVED = 2
    AI_REJECTED = 3
    COUNCIL_VOTING = 4
    APPROVED = 5
    ACTIVE = 6
    FROZEN = 7       # Issuer or regulator freeze
    RETIRED = 8      # Asset no longer backing token


class ComplianceResult(IntEnum):
    """Result of a compliance check."""
    PASS = 0
    FAIL_SENDER_KYC = 1
    FAIL_RECIPIENT_KYC = 2
    FAIL_JURISDICTION = 3
    FAIL_TRANSFER_LIMIT = 4
    FAIL_COOLDOWN = 5
    FAIL_FROZEN = 6
    FAIL_MAX_HOLDERS = 7


@dataclass
class PRC3643Token:
    """PRC-3643 Security Token for Real-World Assets.

    Every transfer passes through compliance checks before execution.
    """
    token_id: str
    name: str
    symbol: str
    decimals: int
    total_supply: int
    issuer: bytes               # Issuer's blockchain address
    asset_type: AssetType
    status: TokenStatus = TokenStatus.PENDING

    # Asset metadata
    description: str = ""
    jurisdiction: str = ""      # ISO 3166-1 alpha-2 (e.g., "US", "DE")
    legal_doc_hash: str = ""    # SHA-512 hash of legal documentation
    valuation: int = 0          # Asset valuation in ASF

    # Compliance
    allowed_jurisdictions: List[str] = field(default_factory=list)
    max_holders: int = RWA_MAX_HOLDERS
    min_kyc_level: int = 2      # Minimum KYC level required

    # AI review
    ai_risk_score: float = 0.0
    votes_for: int = 0
    votes_against: int = 0
    voters: List[str] = field(default_factory=list)

    # Statistics
    total_transfers: int = 0
    total_dividends_distributed: int = 0
    holder_count: int = 0

    # Timestamps
    created_at: float = 0.0
    approved_at: float = 0.0

    # Internal state
    _balances: Dict[bytes, int] = field(default_factory=dict)
    _allowances: Dict[Tuple[bytes, bytes], int] = field(default_factory=dict)
    _frozen_addresses: set = field(default_factory=set)
    _last_large_transfer: Dict[bytes, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.total_supply > 0 and self.issuer not in self._balances:
            self._balances[self.issuer] = self.total_supply
            self.holder_count = 1
        if not self.allowed_jurisdictions:
            self.allowed_jurisdictions = []  # Empty = all allowed

    # ---- Balance ----

    def balance_of(self, address: bytes) -> int:
        return self._balances.get(address, 0)

    def get_holders(self) -> Dict[bytes, int]:
        """Return all non-zero holders."""
        return {k: v for k, v in self._balances.items() if v > 0}

    # ---- Compliance Check ----

    def check_compliance(
        self,
        sender: bytes,
        recipient: bytes,
        amount: int,
        sender_kyc_level: int = 0,
        recipient_kyc_level: int = 0,
        sender_jurisdiction: str = "",
        recipient_jurisdiction: str = "",
    ) -> ComplianceResult:
        """Check if a transfer is compliant.

        Returns ComplianceResult.PASS if transfer is allowed.
        """
        # Token must be active
        if self.status in (TokenStatus.FROZEN, TokenStatus.RETIRED):
            return ComplianceResult.FAIL_FROZEN

        # Frozen addresses
        if sender in self._frozen_addresses:
            return ComplianceResult.FAIL_FROZEN
        if recipient in self._frozen_addresses:
            return ComplianceResult.FAIL_FROZEN

        # KYC check
        if sender_kyc_level < self.min_kyc_level:
            return ComplianceResult.FAIL_SENDER_KYC
        if recipient_kyc_level < self.min_kyc_level:
            return ComplianceResult.FAIL_RECIPIENT_KYC

        # Jurisdiction check
        if self.allowed_jurisdictions:
            if sender_jurisdiction and sender_jurisdiction not in self.allowed_jurisdictions:
                return ComplianceResult.FAIL_JURISDICTION
            if recipient_jurisdiction and recipient_jurisdiction not in self.allowed_jurisdictions:
                return ComplianceResult.FAIL_JURISDICTION

        # Max holders check (only for new holders)
        if self.balance_of(recipient) == 0:
            if self.holder_count >= self.max_holders:
                return ComplianceResult.FAIL_MAX_HOLDERS

        # Large transfer cooldown
        if self.total_supply > 0:
            ratio = amount / self.total_supply
            if ratio > RWA_LARGE_TRANSFER_THRESHOLD:
                last = self._last_large_transfer.get(sender, 0)
                if time.time() - last < RWA_TRANSFER_COOLDOWN:
                    return ComplianceResult.FAIL_COOLDOWN

        return ComplianceResult.PASS

    # ---- Transfer (with compliance) ----

    def transfer(
        self,
        sender: bytes,
        recipient: bytes,
        amount: int,
        sender_kyc_level: int = 0,
        recipient_kyc_level: int = 0,
        sender_jurisdiction: str = "",
        recipient_jurisdiction: str = "",
    ) -> Tuple[bool, ComplianceResult]:
        """Transfer with compliance check. Returns (success, compliance_result)."""
        if amount <= 0 or sender == recipient:
            return False, ComplianceResult.FAIL_TRANSFER_LIMIT
        if self.balance_of(sender) < amount:
            return False, ComplianceResult.FAIL_TRANSFER_LIMIT

        compliance = self.check_compliance(
            sender, recipient, amount,
            sender_kyc_level, recipient_kyc_level,
            sender_jurisdiction, recipient_jurisdiction,
        )
        if compliance != ComplianceResult.PASS:
            return False, compliance

        # Execute transfer
        was_new_holder = self.balance_of(recipient) == 0
        self._balances[sender] = self.balance_of(sender) - amount
        self._balances[recipient] = self.balance_of(recipient) + amount

        # Track holder count
        if was_new_holder:
            self.holder_count += 1
        if self._balances[sender] == 0:
            self.holder_count -= 1

        # Track large transfer cooldown
        if self.total_supply > 0 and amount / self.total_supply > RWA_LARGE_TRANSFER_THRESHOLD:
            self._last_large_transfer[sender] = time.time()

        self.total_transfers += 1
        return True, ComplianceResult.PASS

    # ---- Allowance ----

    def approve(self, owner: bytes, spender: bytes, amount: int) -> bool:
        if amount < 0:
            return False
        self._allowances[(owner, spender)] = amount
        return True

    def allowance(self, owner: bytes, spender: bytes) -> int:
        return self._allowances.get((owner, spender), 0)

    # ---- Issuer Controls ----

    def freeze_address(self, address: bytes, caller: bytes) -> bool:
        """Freeze an address (issuer only)."""
        if caller != self.issuer:
            return False
        self._frozen_addresses.add(address)
        return True

    def unfreeze_address(self, address: bytes, caller: bytes) -> bool:
        """Unfreeze an address (issuer only)."""
        if caller != self.issuer:
            return False
        self._frozen_addresses.discard(address)
        return True

    def force_transfer(
        self, from_addr: bytes, to_addr: bytes, amount: int, caller: bytes
    ) -> bool:
        """Force transfer for legal compliance (issuer only, bypasses checks)."""
        if caller != self.issuer:
            return False
        if amount <= 0 or self.balance_of(from_addr) < amount:
            return False
        was_new = self.balance_of(to_addr) == 0
        self._balances[from_addr] = self.balance_of(from_addr) - amount
        self._balances[to_addr] = self.balance_of(to_addr) + amount
        if was_new:
            self.holder_count += 1
        if self._balances[from_addr] == 0:
            self.holder_count -= 1
        self.total_transfers += 1
        return True

    def freeze_token(self, caller: bytes) -> bool:
        """Freeze entire token (issuer only)."""
        if caller != self.issuer:
            return False
        self.status = TokenStatus.FROZEN
        return True

    def unfreeze_token(self, caller: bytes) -> bool:
        """Unfreeze token (issuer only)."""
        if caller != self.issuer:
            return False
        if self.status == TokenStatus.FROZEN:
            self.status = TokenStatus.ACTIVE
            return True
        return False

    # ---- Serialization ----

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "name": self.name,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "issuer": self.issuer.hex(),
            "asset_type": AssetType(self.asset_type).name,
            "status": TokenStatus(self.status).name,
            "description": self.description,
            "jurisdiction": self.jurisdiction,
            "valuation": self.valuation,
            "allowed_jurisdictions": self.allowed_jurisdictions,
            "min_kyc_level": self.min_kyc_level,
            "holder_count": self.holder_count,
            "total_transfers": self.total_transfers,
            "total_dividends_distributed": self.total_dividends_distributed,
            "ai_risk_score": self.ai_risk_score,
            "created_at": self.created_at,
            "approved_at": self.approved_at,
        }
