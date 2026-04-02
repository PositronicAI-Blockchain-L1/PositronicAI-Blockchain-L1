"""
Positronic - AI Model Governance
On-chain governance for AI model updates, weight changes, and kill switch control.
Includes treasury spending via supermajority governance vote.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import IntEnum

from positronic.constants import BASE_UNIT, AI_MODEL_ACTIVATION_DELAY


class ProposalType(IntEnum):
    MODEL_UPDATE = 0       # Deploy a new AI model version
    WEIGHT_CHANGE = 1      # Change model component weights
    KILL_SWITCH = 2        # Enable/disable AI validation
    THRESHOLD_CHANGE = 3   # Change accept/quarantine thresholds
    TREASURY_SPEND = 4     # Spend from AI treasury


class ProposalStatus(IntEnum):
    PENDING = 0
    VOTING = 1
    APPROVED = 2
    REJECTED = 3
    EXECUTED = 4
    EXPIRED = 5


@dataclass
class GovernanceProposal:
    """A governance proposal for AI model changes."""
    proposal_id: int
    proposal_type: ProposalType
    proposer: bytes            # Address of proposer
    title: str
    description: str
    parameters: dict           # Type-specific parameters
    created_at_block: int
    voting_deadline_block: int
    votes_for: int = 0
    votes_against: int = 0
    voters: set = field(default_factory=set)
    status: ProposalStatus = ProposalStatus.PENDING
    executed_at_block: int = 0

    @property
    def vote_ratio(self) -> float:
        total = self.votes_for + self.votes_against
        return self.votes_for / max(total, 1)

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.name,
            "proposer": self.proposer.hex(),
            "title": self.title,
            "description": self.description,
            "parameters": self.parameters,
            "created_at_block": self.created_at_block,
            "voting_deadline_block": self.voting_deadline_block,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "voter_count": len(self.voters),
            "status": self.status.name,
            "vote_ratio": self.vote_ratio,
        }


class ModelGovernance:
    """
    Manages on-chain governance for AI model decisions.

    Rules:
    - Any staker can propose changes
    - Voting period: 1000 blocks (~50 minutes)
    - Approval requires >66% supermajority
    - Minimum 10% of stakers must vote (quorum)
    """

    VOTING_PERIOD = 1000        # blocks
    APPROVAL_THRESHOLD = 0.66   # 66% supermajority
    QUORUM_THRESHOLD = 0.10     # 10% of stakers must vote

    def __init__(self):
        self.proposals: Dict[int, GovernanceProposal] = {}
        self.next_proposal_id: int = 1
        self.executed_proposals: List[int] = []
        self.total_stakers: int = 1  # Updated by consensus module
        # Pending model activations: {activation_height: model_hash}
        self._pending_updates: Dict[int, str] = {}
        self._active_model_version: int = 1
        self._model_update_log: List[dict] = []

    def create_proposal(
        self,
        proposer: bytes,
        proposal_type: ProposalType,
        title: str,
        description: str,
        parameters: dict,
        current_block: int,
    ) -> GovernanceProposal:
        """Create a new governance proposal."""
        proposal = GovernanceProposal(
            proposal_id=self.next_proposal_id,
            proposal_type=proposal_type,
            proposer=proposer,
            title=title,
            description=description,
            parameters=parameters,
            created_at_block=current_block,
            voting_deadline_block=current_block + self.VOTING_PERIOD,
            status=ProposalStatus.VOTING,
        )

        self.proposals[self.next_proposal_id] = proposal
        self.next_proposal_id += 1
        return proposal

    def vote(
        self,
        proposal_id: int,
        voter: bytes,
        vote_for: bool,
        stake_weight: int = 1,
    ) -> bool:
        """Cast a vote on a proposal. Weighted by stake."""
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.VOTING:
            return False

        # Prevent double voting
        voter_key = voter.hex()
        if voter_key in proposal.voters:
            return False

        proposal.voters.add(voter_key)
        if vote_for:
            proposal.votes_for += stake_weight
        else:
            proposal.votes_against += stake_weight

        return True

    def finalize_proposals(self, current_block: int) -> List[GovernanceProposal]:
        """Check and finalize proposals that have passed their voting deadline."""
        finalized = []

        for proposal in self.proposals.values():
            if proposal.status != ProposalStatus.VOTING:
                continue
            if current_block < proposal.voting_deadline_block:
                continue

            total_votes = proposal.votes_for + proposal.votes_against

            # Check quorum
            quorum_met = len(proposal.voters) >= max(
                self.total_stakers * self.QUORUM_THRESHOLD, 1
            )

            if not quorum_met:
                proposal.status = ProposalStatus.EXPIRED
            elif proposal.vote_ratio >= self.APPROVAL_THRESHOLD:
                proposal.status = ProposalStatus.APPROVED
                finalized.append(proposal)
            else:
                proposal.status = ProposalStatus.REJECTED

        return finalized

    # Maximum single treasury spend: 5% of initial treasury (10M ASF)
    MAX_TREASURY_SPEND = 10_000_000 * BASE_UNIT

    # Treasury spending requires 75% supermajority (stricter than normal 66%)
    TREASURY_APPROVAL_THRESHOLD = 0.75

    # Minimum voters for treasury spending (20% quorum, stricter than 10%)
    TREASURY_QUORUM_THRESHOLD = 0.20

    def execute_proposal(
        self, proposal_id: int, current_block: int
    ) -> Optional[GovernanceProposal]:
        """
        Validate and mark a proposal as executed.

        For TREASURY_SPEND proposals, validates:
        - Required parameters (recipient, amount) are present
        - Amount does not exceed MAX_TREASURY_SPEND
        - Recipient is a valid 20-byte address
        - Supermajority threshold met (75% for treasury)
        - Higher quorum threshold met (20% for treasury)

        Returns the proposal if execution is valid, None otherwise.
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.APPROVED:
            return None

        # Treasury spending has stricter validation
        if proposal.proposal_type == ProposalType.TREASURY_SPEND:
            if not self._validate_treasury_spend(proposal):
                return None

        # MODEL_UPDATE: schedule delayed activation
        if proposal.proposal_type == ProposalType.MODEL_UPDATE:
            model_hash = proposal.parameters.get("model_hash", "")
            if not model_hash:
                return None  # Model hash required
            activation_height = proposal.parameters.get(
                "activation_height",
                current_block + AI_MODEL_ACTIVATION_DELAY,
            )
            if activation_height <= current_block:
                activation_height = current_block + AI_MODEL_ACTIVATION_DELAY
            self._pending_updates[activation_height] = model_hash

        proposal.status = ProposalStatus.EXECUTED
        proposal.executed_at_block = current_block
        self.executed_proposals.append(proposal_id)
        return proposal

    def _validate_treasury_spend(self, proposal: GovernanceProposal) -> bool:
        """Validate a TREASURY_SPEND proposal meets stricter requirements."""
        params = proposal.parameters

        # Must have recipient and amount
        if "recipient" not in params or "amount" not in params:
            return False

        # Validate recipient is a hex string that decodes to 20 bytes
        try:
            recipient_hex = params["recipient"].replace("0x", "")
            recipient_bytes = bytes.fromhex(recipient_hex)
            if len(recipient_bytes) != 20:
                return False
        except (ValueError, AttributeError):
            return False

        # Validate amount is positive and within cap
        try:
            amount = int(params["amount"])
            if amount <= 0 or amount > self.MAX_TREASURY_SPEND:
                return False
        except (ValueError, TypeError):
            return False

        # Treasury requires higher supermajority (75%)
        if proposal.vote_ratio < self.TREASURY_APPROVAL_THRESHOLD:
            return False

        # Treasury requires higher quorum (20% of stakers)
        quorum_met = len(proposal.voters) >= max(
            self.total_stakers * self.TREASURY_QUORUM_THRESHOLD, 1
        )
        if not quorum_met:
            return False

        return True

    def get_treasury_spend_tx_params(
        self, proposal_id: int
    ) -> Optional[dict]:
        """
        Get transaction parameters for an executed TREASURY_SPEND proposal.

        Returns a dict with 'recipient' (bytes) and 'amount' (int) suitable
        for creating a treasury spending transaction, or None if the proposal
        is not a valid executed treasury spend.
        """
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return None
        if proposal.status != ProposalStatus.EXECUTED:
            return None
        if proposal.proposal_type != ProposalType.TREASURY_SPEND:
            return None

        params = proposal.parameters
        recipient_hex = params["recipient"].replace("0x", "")
        return {
            "recipient": bytes.fromhex(recipient_hex),
            "amount": int(params["amount"]),
            "proposal_id": proposal_id,
            "title": proposal.title,
        }

    def check_pending_activations(self, current_height: int, ai_gate=None) -> List[dict]:
        """
        Check if any pending model updates should activate at current_height.
        Called from blockchain during block processing.
        Returns list of activations performed.
        """
        activated = []
        heights_to_remove = []

        for height, model_hash in sorted(self._pending_updates.items()):
            if current_height >= height:
                self._active_model_version += 1
                if ai_gate is not None:
                    ai_gate.model_version = self._active_model_version
                activation = {
                    "model_hash": model_hash,
                    "activation_height": height,
                    "new_version": self._active_model_version,
                    "activated_at_block": current_height,
                }
                self._model_update_log.append(activation)
                activated.append(activation)
                heights_to_remove.append(height)

        for h in heights_to_remove:
            del self._pending_updates[h]

        return activated

    def get_pending_updates(self) -> Dict[int, str]:
        """Get all pending model updates (height -> model_hash)."""
        return dict(self._pending_updates)

    def get_model_update_log(self) -> List[dict]:
        """Get history of all model activations."""
        return list(self._model_update_log)

    def get_active_proposals(self) -> List[GovernanceProposal]:
        """Get all proposals currently in voting period."""
        return [
            p for p in self.proposals.values()
            if p.status == ProposalStatus.VOTING
        ]

    def get_stats(self) -> dict:
        total_treasury_spent = sum(
            int(self.proposals[pid].parameters.get("amount", 0))
            for pid in self.executed_proposals
            if self.proposals[pid].proposal_type == ProposalType.TREASURY_SPEND
        )
        return {
            "total_proposals": len(self.proposals),
            "active_proposals": len(self.get_active_proposals()),
            "executed_proposals": len(self.executed_proposals),
            "total_stakers": self.total_stakers,
            "total_treasury_spent": total_treasury_spent,
        }
