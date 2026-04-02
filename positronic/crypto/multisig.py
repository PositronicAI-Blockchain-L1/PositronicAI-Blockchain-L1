"""
Positronic - Multi-Signature (M-of-N) Support
Enables threshold signatures where M out of N signers must approve.
Superior to Bitcoin's Script-based multisig with native Ed25519 support.

Features:
- Ed25519-based M-of-N multisig
- Deterministic multisig address derivation
- On-chain multisig account tracking
- Time-locked proposals for security
"""

import time
import logging
from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

from positronic.crypto.hashing import sha512, blake2b_160
from positronic.crypto.keys import KeyPair
from positronic.constants import MAX_MULTISIG_SIGNERS, MIN_MULTISIG_THRESHOLD, CHAIN_ID

logger = logging.getLogger("positronic.crypto.multisig")


class MultisigProposalStatus(IntEnum):
    """Status of a multisig transaction proposal."""
    PENDING = 0     # Waiting for signatures
    APPROVED = 1    # Got enough signatures
    EXECUTED = 2    # Transaction executed
    EXPIRED = 3     # Timed out
    CANCELLED = 4   # Cancelled by proposer


@dataclass
class MultisigWallet:
    """
    A multi-signature wallet requiring M-of-N approvals.

    The address is deterministically derived from the sorted
    list of signer public keys and the threshold.
    """
    threshold: int          # M: minimum signatures required
    signers: List[bytes]    # N: list of signer public keys (32 bytes each)
    nonce: int = 0          # Incremented with each executed transaction
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def n_signers(self) -> int:
        return len(self.signers)

    @property
    def address(self) -> bytes:
        """
        Derive the multisig address from threshold + sorted signer keys.
        address = Blake2b-160(SHA-512(threshold || sorted_signers))
        """
        # Sort signers for deterministic address
        sorted_keys = sorted(self.signers)
        data = self.threshold.to_bytes(4, "big")
        for key in sorted_keys:
            data += key
        return blake2b_160(sha512(data))

    @property
    def address_hex(self) -> str:
        return "0x" + self.address.hex()

    def is_signer(self, pubkey: bytes) -> bool:
        """Check if a public key is an authorized signer."""
        return pubkey in self.signers

    def validate(self) -> Tuple[bool, str]:
        """Validate the multisig configuration."""
        if self.threshold < MIN_MULTISIG_THRESHOLD:
            return False, f"Threshold {self.threshold} below minimum {MIN_MULTISIG_THRESHOLD}"

        if self.threshold > len(self.signers):
            return False, f"Threshold {self.threshold} exceeds signer count {len(self.signers)}"

        if len(self.signers) > MAX_MULTISIG_SIGNERS:
            return False, f"Too many signers: {len(self.signers)} > {MAX_MULTISIG_SIGNERS}"

        if len(self.signers) < 2:
            return False, "Multisig requires at least 2 signers"

        # Check for duplicate signers
        if len(set(s.hex() for s in self.signers)) != len(self.signers):
            return False, "Duplicate signers detected"

        # Check key lengths
        for signer in self.signers:
            if len(signer) != 32:
                return False, f"Invalid signer key length: {len(signer)}"

        return True, ""

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "signers": [s.hex() for s in self.signers],
            "n_signers": self.n_signers,
            "address": self.address.hex(),
            "nonce": self.nonce,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultisigWallet":
        return cls(
            threshold=d["threshold"],
            signers=[bytes.fromhex(s) for s in d["signers"]],
            nonce=d.get("nonce", 0),
            created_at=d.get("created_at", 0.0),
        )


@dataclass
class MultisigProposal:
    """
    A proposed transaction that requires M-of-N signatures.
    """
    proposal_id: bytes         # SHA-512 hash of proposal data
    wallet_address: bytes      # Multisig wallet address
    proposer: bytes            # Public key of proposer
    recipient: bytes           # Destination address
    value: int                 # Amount to send
    data: bytes = b""          # Optional calldata
    nonce: int = 0             # Wallet nonce at time of proposal
    created_at: float = 0.0
    expires_at: float = 0.0    # Expiry time (default: 24 hours)
    status: MultisigProposalStatus = MultisigProposalStatus.PENDING
    signatures: Dict[str, bytes] = field(default_factory=dict)  # pubkey_hex -> signature

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.expires_at == 0.0:
            self.expires_at = self.created_at + 86400  # 24 hours
        if not self.proposal_id:
            self.proposal_id = self._compute_id()

    def _compute_id(self) -> bytes:
        """Compute unique proposal ID.

        Security fix: includes CHAIN_ID to prevent cross-chain replay attacks.
        """
        data = (
            self.wallet_address
            + self.recipient
            + self.value.to_bytes(32, "big")
            + self.data
            + self.nonce.to_bytes(8, "big")
            + CHAIN_ID.to_bytes(8, "big")
        )
        return sha512(data)

    @property
    def signing_data(self) -> bytes:
        """Data that each signer must sign.

        Security fix: includes CHAIN_ID alongside proposal_id for
        defense-in-depth against cross-chain replay.
        """
        return self.proposal_id + CHAIN_ID.to_bytes(8, "big")

    @property
    def signature_count(self) -> int:
        return len(self.signatures)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def add_signature(
        self,
        signer_pubkey: bytes,
        signature: bytes,
        wallet: MultisigWallet,
    ) -> Tuple[bool, str]:
        """
        Add a signature to the proposal.
        Returns (success, error_message).
        """
        if self.status != MultisigProposalStatus.PENDING:
            return False, f"Proposal not pending (status={self.status.name})"

        if self.is_expired:
            self.status = MultisigProposalStatus.EXPIRED
            return False, "Proposal expired"

        if not wallet.is_signer(signer_pubkey):
            return False, "Not an authorized signer"

        pubkey_hex = signer_pubkey.hex()
        if pubkey_hex in self.signatures:
            return False, "Already signed"

        # Verify signature
        if not KeyPair.verify(signer_pubkey, signature, self.signing_data):
            return False, "Invalid signature"

        self.signatures[pubkey_hex] = signature

        # Check if threshold is met
        if self.signature_count >= wallet.threshold:
            self.status = MultisigProposalStatus.APPROVED

        return True, ""

    def has_enough_signatures(self, threshold: int) -> bool:
        """Check if threshold is met."""
        return self.signature_count >= threshold

    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id.hex(),
            "wallet_address": self.wallet_address.hex(),
            "proposer": self.proposer.hex(),
            "recipient": self.recipient.hex(),
            "value": self.value,
            "data": self.data.hex(),
            "nonce": self.nonce,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "status": self.status.name,
            "signature_count": self.signature_count,
            "signatures": {k: v.hex() for k, v in self.signatures.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultisigProposal":
        return cls(
            proposal_id=bytes.fromhex(d["proposal_id"]),
            wallet_address=bytes.fromhex(d["wallet_address"]),
            proposer=bytes.fromhex(d["proposer"]),
            recipient=bytes.fromhex(d["recipient"]),
            value=d["value"],
            data=bytes.fromhex(d.get("data", "")),
            nonce=d.get("nonce", 0),
            created_at=d.get("created_at", 0.0),
            expires_at=d.get("expires_at", 0.0),
            status=MultisigProposalStatus[d.get("status", "PENDING")],
            signatures={k: bytes.fromhex(v) for k, v in d.get("signatures", {}).items()},
        )


class MultisigManager:
    """
    Manages multisig wallets and proposals.

    Provides:
    - Wallet creation and tracking
    - Proposal lifecycle management
    - Signature collection and verification
    - Threshold enforcement
    """

    def __init__(self):
        self._wallets: Dict[bytes, MultisigWallet] = {}  # address -> wallet
        self._proposals: Dict[bytes, MultisigProposal] = {}  # proposal_id -> proposal
        self._wallet_proposals: Dict[bytes, List[bytes]] = {}  # wallet_addr -> [proposal_ids]

    def create_wallet(
        self,
        threshold: int,
        signers: List[bytes],
    ) -> Tuple[Optional[MultisigWallet], str]:
        """
        Create a new multisig wallet.
        Returns (wallet, error_message).
        """
        wallet = MultisigWallet(threshold=threshold, signers=signers)
        valid, error = wallet.validate()
        if not valid:
            return None, error

        if wallet.address in self._wallets:
            return None, "Wallet already exists"

        self._wallets[wallet.address] = wallet
        self._wallet_proposals[wallet.address] = []

        logger.info(
            f"Created {threshold}-of-{len(signers)} multisig wallet: "
            f"{wallet.address_hex}"
        )

        return wallet, ""

    def get_wallet(self, address: bytes) -> Optional[MultisigWallet]:
        """Get a multisig wallet by address."""
        return self._wallets.get(address)

    def create_proposal(
        self,
        wallet_address: bytes,
        proposer: bytes,
        recipient: bytes,
        value: int,
        data: bytes = b"",
    ) -> Tuple[Optional[MultisigProposal], str]:
        """Create a new multisig transaction proposal."""
        wallet = self._wallets.get(wallet_address)
        if not wallet:
            return None, "Wallet not found"

        if not wallet.is_signer(proposer):
            return None, "Proposer is not a signer"

        proposal = MultisigProposal(
            proposal_id=b"",
            wallet_address=wallet_address,
            proposer=proposer,
            recipient=recipient,
            value=value,
            data=data,
            nonce=wallet.nonce,
        )

        self._proposals[proposal.proposal_id] = proposal
        self._wallet_proposals.setdefault(wallet_address, []).append(
            proposal.proposal_id
        )

        return proposal, ""

    def sign_proposal(
        self,
        proposal_id: bytes,
        signer_pubkey: bytes,
        signature: bytes,
    ) -> Tuple[bool, str]:
        """Add a signature to a proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return False, "Proposal not found"

        wallet = self._wallets.get(proposal.wallet_address)
        if not wallet:
            return False, "Wallet not found"

        return proposal.add_signature(signer_pubkey, signature, wallet)

    def execute_proposal(
        self,
        proposal_id: bytes,
    ) -> Tuple[bool, str]:
        """Mark a proposal as executed (after on-chain execution)."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return False, "Proposal not found"

        wallet = self._wallets.get(proposal.wallet_address)
        if not wallet:
            return False, "Wallet not found"

        if proposal.status != MultisigProposalStatus.APPROVED:
            return False, f"Proposal not approved (status={proposal.status.name})"

        proposal.status = MultisigProposalStatus.EXECUTED
        wallet.nonce += 1

        return True, ""

    def get_proposal(self, proposal_id: bytes) -> Optional[MultisigProposal]:
        return self._proposals.get(proposal_id)

    def get_wallet_proposals(
        self, wallet_address: bytes
    ) -> List[MultisigProposal]:
        """Get all proposals for a wallet."""
        ids = self._wallet_proposals.get(wallet_address, [])
        return [self._proposals[pid] for pid in ids if pid in self._proposals]

    def cleanup_expired(self):
        """Mark expired proposals."""
        for proposal in self._proposals.values():
            if (
                proposal.status == MultisigProposalStatus.PENDING
                and proposal.is_expired
            ):
                proposal.status = MultisigProposalStatus.EXPIRED

    def get_stats(self) -> dict:
        pending = sum(
            1 for p in self._proposals.values()
            if p.status == MultisigProposalStatus.PENDING
        )
        return {
            "wallets": len(self._wallets),
            "total_proposals": len(self._proposals),
            "pending_proposals": pending,
        }
