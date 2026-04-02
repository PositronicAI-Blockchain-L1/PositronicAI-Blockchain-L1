"""
Positronic - Token Governance System
AI + governance council approval required before anyone can deploy tokens.
Ensures only legitimate tokens are created on the Positronic network.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


class ProposalStatus(IntEnum):
    """Status of a token creation proposal."""
    SUBMITTED = 0
    AI_REVIEWING = 1
    AI_APPROVED = 2
    AI_REJECTED = 3
    COUNCIL_VOTING = 4
    APPROVED = 5
    REJECTED = 6
    DEPLOYED = 7


class ProposalType(IntEnum):
    """Types of governance proposals."""
    TOKEN_CREATION = 1        # Create new token on Positronic network
    TOKEN_MODIFICATION = 2    # Modify existing token parameters
    AI_MODEL_UPDATE = 3       # Update AI model version
    PARAMETER_CHANGE = 4      # Change network parameters
    EMERGENCY_ACTION = 5      # Emergency governance action


@dataclass
class TokenProposal:
    """A proposal to create a new token on the Positronic network."""
    proposal_id: str
    proposer: bytes                     # Address of proposer
    status: ProposalStatus = ProposalStatus.SUBMITTED
    proposal_type: ProposalType = ProposalType.TOKEN_CREATION
    created_at: float = 0.0

    # Token details
    token_name: str = ""
    token_symbol: str = ""
    token_supply: int = 0
    token_decimals: int = 18
    token_description: str = ""

    # AI review
    ai_risk_score: float = 0.0
    ai_review_notes: str = ""

    # Council voting
    votes_for: int = 0
    votes_against: int = 0
    voters: List[str] = field(default_factory=list)

    # Deployment
    contract_address: bytes = b""
    deployed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer.hex(),
            "status": self.status.name,
            "proposal_type": self.proposal_type.name,
            "created_at": self.created_at,
            "token_name": self.token_name,
            "token_symbol": self.token_symbol,
            "token_supply": self.token_supply,
            "token_decimals": self.token_decimals,
            "token_description": self.token_description,
            "ai_risk_score": self.ai_risk_score,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "contract_address": self.contract_address.hex() if self.contract_address else "",
        }


class TokenGovernance:
    """
    Manages the token creation governance process.
    Flow: Submit -> AI Review -> Council Vote -> Deploy
    """

    COUNCIL_MIN_VOTES = 3       # Minimum council votes needed
    COUNCIL_APPROVAL_PCT = 0.6  # 60% approval needed
    AI_MAX_RISK_SCORE = 0.7     # AI must score below 0.7 to pass

    def __init__(self):
        self._proposals: Dict[str, TokenProposal] = {}
        self._council_members: set = set()
        self._deployed_tokens: Dict[str, TokenProposal] = {}
        self._proposal_counter: int = 0

    def add_council_member(self, address: bytes):
        """Add a governance council member."""
        self._council_members.add(address)

    def remove_council_member(self, address: bytes):
        """Remove a governance council member."""
        self._council_members.discard(address)

    def submit_proposal(self, proposer: bytes, token_name: str,
                        token_symbol: str, token_supply: int,
                        token_decimals: int = 18,
                        description: str = "") -> TokenProposal:
        """Submit a new token creation proposal."""
        self._proposal_counter += 1
        proposal_id = f"TGP-{int(time.time())}-{self._proposal_counter:04d}"

        proposal = TokenProposal(
            proposal_id=proposal_id,
            proposer=proposer,
            status=ProposalStatus.SUBMITTED,
            proposal_type=ProposalType.TOKEN_CREATION,
            created_at=time.time(),
            token_name=token_name,
            token_symbol=token_symbol,
            token_supply=token_supply,
            token_decimals=token_decimals,
            token_description=description,
        )

        self._proposals[proposal_id] = proposal
        return proposal

    def ai_review(self, proposal_id: str, risk_score: float,
                  notes: str = "") -> TokenProposal:
        """AI reviews a proposal and scores it."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal.ai_risk_score = risk_score
        proposal.ai_review_notes = notes
        proposal.status = ProposalStatus.AI_REVIEWING

        if risk_score <= self.AI_MAX_RISK_SCORE:
            proposal.status = ProposalStatus.AI_APPROVED
        else:
            proposal.status = ProposalStatus.AI_REJECTED

        return proposal

    def council_vote(self, proposal_id: str, voter: bytes,
                     approve: bool) -> TokenProposal:
        """A council member votes on a proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")

        if voter not in self._council_members:
            raise ValueError("Voter is not a council member")

        voter_hex = voter.hex()
        if voter_hex in proposal.voters:
            raise ValueError("Already voted")

        if proposal.status not in (ProposalStatus.AI_APPROVED,
                                    ProposalStatus.COUNCIL_VOTING):
            raise ValueError(f"Proposal not ready for voting (status: {proposal.status.name})")

        proposal.status = ProposalStatus.COUNCIL_VOTING
        proposal.voters.append(voter_hex)

        if approve:
            proposal.votes_for += 1
        else:
            proposal.votes_against += 1

        # Check if enough votes to decide
        total_votes = proposal.votes_for + proposal.votes_against
        if total_votes >= self.COUNCIL_MIN_VOTES:
            approval_rate = proposal.votes_for / total_votes
            if approval_rate >= self.COUNCIL_APPROVAL_PCT:
                proposal.status = ProposalStatus.APPROVED
            elif total_votes >= len(self._council_members):
                proposal.status = ProposalStatus.REJECTED

        return proposal

    def mark_deployed(self, proposal_id: str,
                      contract_address: bytes) -> TokenProposal:
        """Mark a proposal as deployed."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")

        if proposal.status != ProposalStatus.APPROVED:
            raise ValueError("Proposal not approved")

        proposal.status = ProposalStatus.DEPLOYED
        proposal.contract_address = contract_address
        proposal.deployed_at = time.time()
        self._deployed_tokens[proposal.token_symbol] = proposal

        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[TokenProposal]:
        return self._proposals.get(proposal_id)

    def get_pending_proposals(self) -> List[TokenProposal]:
        return [p for p in self._proposals.values()
                if p.status in (ProposalStatus.SUBMITTED,
                               ProposalStatus.AI_APPROVED,
                               ProposalStatus.COUNCIL_VOTING)]

    @property
    def total_proposals(self) -> int:
        return len(self._proposals)

    @property
    def total_deployed(self) -> int:
        return len(self._deployed_tokens)

    def get_stats(self) -> dict:
        return {
            "total_proposals": self.total_proposals,
            "total_deployed": self.total_deployed,
            "council_members": len(self._council_members),
            "pending": len(self.get_pending_proposals()),
        }
