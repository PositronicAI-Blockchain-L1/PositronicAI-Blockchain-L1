"""
Positronic - RWA Asset Registry (Phase 30)
Registration, AI review, council approval lifecycle for RWA tokens.

Follows the same lifecycle pattern as AgentRegistry:
Register → AI Review → Council Vote → Approve/Reject → Active
"""

import time
import secrets
import hashlib
from typing import Dict, List, Optional

from positronic.constants import (
    RWA_REGISTRATION_FEE,
    RWA_MAX_TOKENS,
    RWA_COUNCIL_MIN_VOTES,
    RWA_COUNCIL_APPROVAL_PCT,
    RWA_AI_MAX_RISK,
)
from positronic.tokens.prc3643 import (
    PRC3643Token,
    AssetType,
    TokenStatus,
    ComplianceResult,
)
from positronic.tokens.compliance import ComplianceEngine
from positronic.tokens.dividend import DividendEngine


class RWARegistry:
    """Manages registration and lifecycle of RWA tokens.

    Lifecycle:
    1. Issuer registers RWA token (pays fee)
    2. AI reviews for risk (fraud detection, valuation sanity)
    3. Council votes on approval
    4. If approved → ACTIVE, can be traded with compliance checks
    5. Issuer can freeze/unfreeze, distribute dividends
    """

    def __init__(self):
        self._tokens: Dict[str, PRC3643Token] = {}
        self._compliance = ComplianceEngine()
        self._dividends = DividendEngine()
        self._council_members: set = set()  # Shared with governance
        self._counter: int = 0

    @property
    def compliance(self) -> ComplianceEngine:
        return self._compliance

    @property
    def dividends(self) -> DividendEngine:
        return self._dividends

    # ---- Registration ----

    def register_token(
        self,
        name: str,
        symbol: str,
        total_supply: int,
        issuer: bytes,
        asset_type: int,
        description: str = "",
        jurisdiction: str = "",
        allowed_jurisdictions: Optional[List[str]] = None,
        valuation: int = 0,
        legal_doc_hash: str = "",
    ) -> Optional[PRC3643Token]:
        """Register a new RWA token. Returns token or None if cap reached."""
        if len(self._tokens) >= RWA_MAX_TOKENS:
            return None

        self._counter += 1
        token_id = f"RWA-{int(time.time())}-{self._counter:04d}"

        token = PRC3643Token(
            token_id=token_id,
            name=name,
            symbol=symbol,
            decimals=18,
            total_supply=total_supply,
            issuer=issuer,
            asset_type=AssetType(asset_type),
            status=TokenStatus.PENDING,
            description=description,
            jurisdiction=jurisdiction.upper(),
            allowed_jurisdictions=[j.upper() for j in (allowed_jurisdictions or [])],
            valuation=valuation,
            legal_doc_hash=legal_doc_hash,
        )

        self._tokens[token_id] = token
        return token

    # ---- AI Review ----

    def ai_review(
        self,
        token_id: str,
        risk_score: float,
        ai_gate=None,
    ) -> Optional[PRC3643Token]:
        """AI reviews the RWA token for risk.

        If risk_score <= RWA_AI_MAX_RISK → auto-moves to COUNCIL_VOTING.
        If risk_score > RWA_AI_MAX_RISK → AI_REJECTED.
        """
        token = self._tokens.get(token_id)
        if not token or token.status != TokenStatus.PENDING:
            return None

        token.status = TokenStatus.AI_REVIEWING
        token.ai_risk_score = risk_score

        if risk_score <= RWA_AI_MAX_RISK:
            token.status = TokenStatus.AI_APPROVED
        else:
            token.status = TokenStatus.AI_REJECTED

        return token

    def move_to_council(self, token_id: str) -> Optional[PRC3643Token]:
        """Move AI-approved token to council voting."""
        token = self._tokens.get(token_id)
        if not token or token.status != TokenStatus.AI_APPROVED:
            return None
        token.status = TokenStatus.COUNCIL_VOTING
        return token

    # ---- Council Voting ----

    def council_vote(
        self,
        token_id: str,
        voter: str,
        approve: bool,
    ) -> Optional[PRC3643Token]:
        """Council member votes on token approval."""
        token = self._tokens.get(token_id)
        if not token or token.status != TokenStatus.COUNCIL_VOTING:
            return None

        # Must be a council member
        if voter not in self._council_members:
            return None

        # No double voting
        if voter in token.voters:
            return None

        token.voters.append(voter)
        if approve:
            token.votes_for += 1
        else:
            token.votes_against += 1

        # Check if enough votes for decision
        total_votes = token.votes_for + token.votes_against
        if total_votes >= RWA_COUNCIL_MIN_VOTES:
            approval_pct = token.votes_for / total_votes
            if approval_pct >= RWA_COUNCIL_APPROVAL_PCT:
                token.status = TokenStatus.APPROVED
            elif (len(self._council_members) - total_votes + token.votes_for) < \
                    RWA_COUNCIL_MIN_VOTES * RWA_COUNCIL_APPROVAL_PCT:
                # Can't reach threshold even with remaining votes
                token.status = TokenStatus.AI_REJECTED

        return token

    def activate_token(self, token_id: str) -> Optional[PRC3643Token]:
        """Activate an approved token for trading."""
        token = self._tokens.get(token_id)
        if not token or token.status != TokenStatus.APPROVED:
            return None
        token.status = TokenStatus.ACTIVE
        token.approved_at = time.time()
        return token

    # ---- Transfer (with compliance) ----

    def transfer(
        self,
        token_id: str,
        sender: bytes,
        recipient: bytes,
        amount: int,
    ) -> tuple:
        """Transfer RWA tokens with full compliance check.

        Returns (success, reason_string).
        """
        token = self._tokens.get(token_id)
        if not token:
            return False, "Token not found"
        if token.status != TokenStatus.ACTIVE:
            return False, f"Token not active (status={token.status.name})"

        # Run compliance check through the engine
        passed, reason = self._compliance.check_transfer_compliance(
            token_id, sender, recipient, amount, token,
        )
        if not passed:
            return False, reason

        # Get KYC info for the token's internal check
        sender_kyc = self._compliance.get_kyc_level(sender)
        recipient_kyc = self._compliance.get_kyc_level(recipient)
        sender_jurisdiction = self._compliance.get_jurisdiction(sender)
        recipient_jurisdiction = self._compliance.get_jurisdiction(recipient)

        success, result = token.transfer(
            sender, recipient, amount,
            sender_kyc, recipient_kyc,
            sender_jurisdiction, recipient_jurisdiction,
        )
        if not success:
            return False, f"Transfer failed: {result.name}"

        return True, "OK"

    # ---- Dividend Distribution ----

    def distribute_dividend(
        self,
        token_id: str,
        total_amount: int,
        caller: bytes,
    ) -> Optional[dict]:
        """Distribute dividends to all holders of an RWA token.

        Only the issuer can trigger dividend distribution.
        Returns dividend record dict or None.
        """
        token = self._tokens.get(token_id)
        if not token:
            return None
        if caller != token.issuer:
            return None
        if token.status != TokenStatus.ACTIVE:
            return None

        holders = token.get_holders()
        record = self._dividends.distribute(
            token_id=token_id,
            total_amount=total_amount,
            holders=holders,
            total_supply=token.total_supply,
            issuer=caller,
        )
        if not record:
            return None

        token.total_dividends_distributed += record.total_amount
        return record.to_dict()

    # ---- Queries ----

    def get_token(self, token_id: str) -> Optional[PRC3643Token]:
        return self._tokens.get(token_id)

    def list_tokens(
        self,
        status: Optional[int] = None,
        asset_type: Optional[int] = None,
        limit: int = 50,
    ) -> List[dict]:
        """List RWA tokens with optional filters."""
        tokens = list(self._tokens.values())
        if status is not None:
            tokens = [t for t in tokens if t.status == status]
        if asset_type is not None:
            tokens = [t for t in tokens if t.asset_type == asset_type]
        tokens.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tokens[:limit]]

    def get_holders(self, token_id: str) -> Optional[List[dict]]:
        """Get all holders of an RWA token."""
        token = self._tokens.get(token_id)
        if not token:
            return None
        holders = token.get_holders()
        return [
            {
                "address": addr.hex(),
                "balance": bal,
                "share_pct": round(bal / token.total_supply * 100, 4)
                if token.total_supply > 0 else 0,
            }
            for addr, bal in holders.items()
        ]

    def get_stats(self) -> dict:
        tokens = list(self._tokens.values())
        active = [t for t in tokens if t.status == TokenStatus.ACTIVE]
        return {
            "total_tokens": len(tokens),
            "active_tokens": len(active),
            "pending_tokens": len([
                t for t in tokens
                if t.status in (TokenStatus.PENDING, TokenStatus.AI_REVIEWING,
                                TokenStatus.COUNCIL_VOTING)
            ]),
            "total_valuation": sum(t.valuation for t in active),
            "total_transfers": sum(t.total_transfers for t in active),
            "total_dividends": sum(t.total_dividends_distributed for t in active),
            "compliance": self._compliance.get_stats(),
            "dividends": self._dividends.get_stats(),
        }
