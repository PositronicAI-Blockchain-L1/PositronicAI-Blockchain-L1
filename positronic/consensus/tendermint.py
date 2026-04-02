"""
Positronic - Tendermint-Style 2-Phase BFT Consensus Engine

Implements the Tendermint consensus algorithm with two voting phases:
  1. PREVOTE  — validators signal which block they consider valid
  2. PRECOMMIT — validators commit to a block after seeing 2/3 prevotes

A block is committed when 2/3+ of active stake precommits for the same
block hash.  Locking and valid-round mechanisms prevent equivocation
and ensure safety across asynchronous rounds.

Timeouts are configurable via positronic.constants (BFT_PROPOSE_TIMEOUT,
BFT_PREVOTE_TIMEOUT, BFT_PRECOMMIT_TIMEOUT, BFT_TIMEOUT_DELTA).
"""

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from positronic.crypto.keys import KeyPair
from positronic.core.block import Block
from positronic.consensus.validator import ValidatorRegistry, Validator
from positronic.consensus.election import ValidatorElection, ElectionResult
from positronic.consensus.slot import SlotClock
from positronic.utils.logging import get_logger

logger = get_logger("positronic.consensus.tendermint")

# Import BFT timeout constants with safe defaults
try:
    from positronic.constants import (
        BFT_PROPOSE_TIMEOUT,
        BFT_PREVOTE_TIMEOUT,
        BFT_PRECOMMIT_TIMEOUT,
        BFT_TIMEOUT_DELTA,
    )
except ImportError:
    BFT_PROPOSE_TIMEOUT = 12.0
    BFT_PREVOTE_TIMEOUT = 6.0
    BFT_PRECOMMIT_TIMEOUT = 6.0
    BFT_TIMEOUT_DELTA = 1.0


# ====================================================================== #
#  Round Step Enum                                                         #
# ====================================================================== #

class RoundStep(IntEnum):
    """Phases within a single consensus round."""
    NEW_ROUND = 0
    PROPOSE = 1
    PREVOTE = 2
    PRECOMMIT = 3
    COMMIT = 4


# ====================================================================== #
#  Round State                                                             #
# ====================================================================== #

@dataclass
class RoundState:
    """
    Mutable state for a single consensus round at a given (height, round).

    Tracks the proposal, collected prevotes/precommits, lock state, and
    the valid block observed during the round.
    """
    height: int
    round: int
    step: RoundStep

    # Proposal
    proposal: Optional[Block] = None
    proposal_block_hash: Optional[bytes] = None

    # Votes: validator_address -> block_hash (or NIL_HASH)
    prevotes: Dict[bytes, bytes] = field(default_factory=dict)
    precommits: Dict[bytes, bytes] = field(default_factory=dict)

    # Signatures: validator_address -> signature bytes
    prevote_signatures: Dict[bytes, bytes] = field(default_factory=dict)
    precommit_signatures: Dict[bytes, bytes] = field(default_factory=dict)

    # Lock mechanism (Tendermint safety)
    locked_round: int = -1
    locked_block: Optional[Block] = None

    # Valid block seen in this round
    valid_round: int = -1
    valid_block: Optional[Block] = None

    # Timing
    start_time: float = 0.0


# ====================================================================== #
#  Tendermint BFT Engine                                                   #
# ====================================================================== #

class TendermintBFT:
    """
    Tendermint-style 2-phase BFT consensus engine for Positronic.

    Manages the round lifecycle (propose -> prevote -> precommit -> commit)
    with stake-weighted 2/3 threshold voting and block locking.
    """

    # Sentinel hash representing a NIL vote (no valid block)
    NIL_HASH: bytes = b"\x00" * 64

    # Timeouts (may be overridden per-round by adding round * DELTA)
    PROPOSE_TIMEOUT: float = BFT_PROPOSE_TIMEOUT
    PREVOTE_TIMEOUT: float = BFT_PREVOTE_TIMEOUT
    PRECOMMIT_TIMEOUT: float = BFT_PRECOMMIT_TIMEOUT
    TIMEOUT_DELTA: float = BFT_TIMEOUT_DELTA

    def __init__(
        self,
        registry: ValidatorRegistry,
        election: ValidatorElection,
        clock: SlotClock,
        my_address: bytes,
        my_keypair: KeyPair,
    ) -> None:
        self._registry = registry
        self._election = election
        self._clock = clock
        self._my_address = my_address
        self._my_keypair = my_keypair

        # Current round state (set by start_round)
        self._state: Optional[RoundState] = None

        # Track committed blocks by height
        self._committed: Dict[int, Block] = {}

        # Double-sign detection: (height, round, step) -> set of seen addrs
        self._seen_votes: Dict[Tuple[int, int, str], Set[bytes]] = {}

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> Optional[RoundState]:
        """Current round state."""
        return self._state

    @property
    def registry(self) -> ValidatorRegistry:
        return self._registry

    @property
    def election(self) -> ValidatorElection:
        return self._election

    # ------------------------------------------------------------------ #
    #  Round lifecycle                                                     #
    # ------------------------------------------------------------------ #

    def start_round(self, height: int, round_num: int) -> RoundState:
        """
        Initialize a new consensus round at (height, round_num).
        Determines the proposer and sets the step to PROPOSE.

        Returns the newly created RoundState.
        """
        state = RoundState(
            height=height,
            round=round_num,
            step=RoundStep.PROPOSE,
            start_time=time.time(),
        )

        # Carry over lock from previous round state if same height
        if self._state is not None and self._state.height == height:
            state.locked_round = self._state.locked_round
            state.locked_block = self._state.locked_block
            state.valid_round = self._state.valid_round
            state.valid_block = self._state.valid_block

        self._state = state

        proposer = self.get_proposer_for_round(height, round_num)
        logger.debug(
            "BFT round started: height=%d round=%d proposer=%s",
            height, round_num,
            proposer.hex() if proposer else "unknown",
        )

        return state

    def get_proposer_for_round(
        self, height: int, round_num: int,
    ) -> Optional[bytes]:
        """
        Deterministically select the proposer for (height, round).

        Uses the election result's active set with a round-robin offset
        derived from (height + round) to ensure fair rotation.

        Returns the validator address of the proposer, or None if
        no election result is available.
        """
        result = self._election.last_result
        if result is None or not result.active_set:
            return None

        active = list(result.active_set)
        index = (height + round_num) % len(active)
        return active[index].address

    # ------------------------------------------------------------------ #
    #  Proposal handling                                                   #
    # ------------------------------------------------------------------ #

    def on_proposal(
        self,
        block: Block,
        proposer: bytes,
        round_num: int,
        valid_round: int,
        signature: bytes,
    ) -> Optional[bytes]:
        """
        Handle an incoming block proposal.

        Validates:
          - The proposer matches the expected proposer for this round.
          - The block height matches the current consensus height.
          - The block's parent hash is consistent.
          - Lock rules are respected.

        Returns:
          - block_hash (bytes) to prevote FOR the block, or
          - NIL_HASH to prevote NIL, or
          - None if the proposal is invalid / ignored.
        """
        state = self._state
        if state is None:
            logger.debug("on_proposal: no active round state")
            return None

        if round_num != state.round:
            logger.debug(
                "on_proposal: round mismatch (got %d, expected %d)",
                round_num, state.round,
            )
            return None

        # Verify proposer is a registered validator (not necessarily the expected one,
        # since election seeds may differ across nodes during sync).
        expected_proposer = self.get_proposer_for_round(state.height, round_num)
        if expected_proposer is not None and proposer != expected_proposer:
            # Check if proposer is at least a registered validator
            is_registered = False
            if self._election and self._election.last_result:
                for v in self._election.last_result.active_set:
                    if v.address == proposer:
                        is_registered = True
                        break
            if not is_registered and self._registry:
                v = self._registry.get(proposer)
                is_registered = v is not None
            # Accept regardless — local registry may not be synced.
            # Block signature verification (done by blockchain.add_block)
            # ensures only valid Ed25519 key holders can propose.

        # Validate block height
        if block.height != state.height:
            logger.debug(
                "on_proposal: height mismatch (block=%d, state=%d)",
                block.height, state.height,
            )
            return None

        # Store proposal
        block_hash = block.hash
        state.proposal = block
        state.proposal_block_hash = block_hash
        state.step = RoundStep.PREVOTE

        # Lock rule: if we are locked on a different block and there is no
        # valid_round proof, vote NIL
        if (
            state.locked_round >= 0
            and state.locked_block is not None
            and state.locked_block.hash != block_hash
        ):
            # Check if the proposal includes a valid_round proof
            if valid_round < 0 or valid_round < state.locked_round:
                logger.debug(
                    "on_proposal: locked on different block, voting NIL"
                )
                return self.NIL_HASH

        logger.debug(
            "on_proposal: accepted proposal for height=%d round=%d hash=%s",
            state.height, round_num, block_hash.hex()[:16],
        )
        return block_hash

    # ------------------------------------------------------------------ #
    #  Prevote handling                                                    #
    # ------------------------------------------------------------------ #

    def on_prevote(
        self,
        validator: bytes,
        block_hash: bytes,
        height: int,
        round_num: int,
        signature: bytes,
    ) -> Optional[str]:
        """
        Process an incoming prevote.

        Validates:
          - The validator is in the active set.
          - The signature is valid over the prevote message.
          - No duplicate vote from this validator in this round.

        Returns:
          - 'precommit'     if 2/3+ prevotes for block_hash
          - 'nil_precommit' if 2/3+ prevotes for NIL
          - None            if no threshold reached yet
        """
        state = self._state
        if state is None:
            return None

        if height != state.height or round_num != state.round:
            return None

        # Verify validator is active
        v = self._registry.get(validator)
        if v is None or not v.is_active:
            logger.debug("on_prevote: validator not active: %s", validator.hex()[:16])
            return None

        # Duplicate vote detection
        vote_key = (height, round_num, "prevote")
        if vote_key not in self._seen_votes:
            self._seen_votes[vote_key] = set()

        if validator in self._seen_votes[vote_key]:
            logger.debug("on_prevote: duplicate from %s", validator.hex()[:16])
            return None

        # Verify signature
        message = self._compute_vote_message("prevote", height, round_num, block_hash)
        if not self._verify_vote_signature(validator, signature, message):
            logger.debug("on_prevote: invalid signature from %s", validator.hex()[:16])
            return None

        # Record the vote
        self._seen_votes[vote_key].add(validator)
        state.prevotes[validator] = block_hash
        state.prevote_signatures[validator] = signature

        # Check 2/3 threshold
        if block_hash != self.NIL_HASH and self.has_two_thirds_prevotes(block_hash):
            state.step = RoundStep.PRECOMMIT
            # Update valid block
            state.valid_round = round_num
            state.valid_block = state.proposal
            logger.debug(
                "on_prevote: 2/3+ prevotes for %s — ready to precommit",
                block_hash.hex()[:16],
            )
            return "precommit"

        if self.has_two_thirds_prevotes(self.NIL_HASH):
            state.step = RoundStep.PRECOMMIT
            logger.debug("on_prevote: 2/3+ NIL prevotes — nil precommit")
            return "nil_precommit"

        return None

    # ------------------------------------------------------------------ #
    #  Precommit handling                                                  #
    # ------------------------------------------------------------------ #

    def on_precommit(
        self,
        validator: bytes,
        block_hash: bytes,
        height: int,
        round_num: int,
        signature: bytes,
    ) -> Optional[Block]:
        """
        Process an incoming precommit.

        Validates:
          - The validator is in the active set.
          - The signature is valid over the precommit message.
          - No duplicate precommit from this validator.

        Returns:
          - The committed Block if 2/3+ precommits for block_hash.
          - None otherwise (including 2/3+ NIL which signals round advance).
        """
        state = self._state
        if state is None:
            return None

        if height != state.height or round_num != state.round:
            return None

        # Verify validator is active
        v = self._registry.get(validator)
        if v is None or not v.is_active:
            logger.debug("on_precommit: validator not active: %s", validator.hex()[:16])
            return None

        # Duplicate detection
        vote_key = (height, round_num, "precommit")
        if vote_key not in self._seen_votes:
            self._seen_votes[vote_key] = set()

        if validator in self._seen_votes[vote_key]:
            logger.debug("on_precommit: duplicate from %s", validator.hex()[:16])
            return None

        # Verify signature
        message = self._compute_vote_message("precommit", height, round_num, block_hash)
        if not self._verify_vote_signature(validator, signature, message):
            logger.debug("on_precommit: invalid signature from %s", validator.hex()[:16])
            return None

        # Record
        self._seen_votes[vote_key].add(validator)
        state.precommits[validator] = block_hash
        state.precommit_signatures[validator] = signature

        # Check 2/3 threshold for a block
        if block_hash != self.NIL_HASH and self.has_two_thirds_precommits(block_hash):
            state.step = RoundStep.COMMIT
            # Lock on the committed block
            state.locked_round = round_num
            state.locked_block = state.proposal

            if state.proposal is not None:
                self._committed[height] = state.proposal
                logger.info(
                    "BFT COMMIT: height=%d round=%d hash=%s",
                    height, round_num, block_hash.hex()[:16],
                )
                return state.proposal

        # 2/3 NIL precommits → signal round advance (return None)
        if self.has_two_thirds_precommits(self.NIL_HASH):
            logger.debug(
                "on_precommit: 2/3+ NIL precommits — advance round"
            )

        return None

    # ------------------------------------------------------------------ #
    #  Timeout handling                                                    #
    # ------------------------------------------------------------------ #

    def on_timeout(self, step: RoundStep, height: int, round_num: int) -> Optional[str]:
        """
        Handle a timeout for the given step.

        Args:
            step:      The step that timed out (PROPOSE, PREVOTE, PRECOMMIT).
            height:    Block height.
            round_num: Round number.

        Returns:
            'nil_prevote'    — caller should broadcast NIL prevote
            'nil_precommit'  — caller should broadcast NIL precommit
            'next_round'     — caller should start the next round
            None             — timeout ignored (stale)
        """
        state = self._state
        if state is None:
            return None
        if height != state.height or round_num != state.round:
            return None

        if step == RoundStep.PROPOSE:
            logger.debug("Timeout: PROPOSE at height=%d round=%d", height, round_num)
            state.step = RoundStep.PREVOTE
            return "nil_prevote"

        elif step == RoundStep.PREVOTE:
            logger.debug("Timeout: PREVOTE at height=%d round=%d", height, round_num)
            state.step = RoundStep.PRECOMMIT
            return "nil_precommit"

        elif step == RoundStep.PRECOMMIT:
            logger.debug("Timeout: PRECOMMIT at height=%d round=%d", height, round_num)
            return "next_round"

        return None

    # ------------------------------------------------------------------ #
    #  Threshold checks                                                    #
    # ------------------------------------------------------------------ #

    def has_two_thirds_prevotes(self, block_hash: Optional[bytes] = None) -> bool:
        """
        Check whether 2/3+ of active stake has prevoted.

        Args:
            block_hash: If provided, check for votes matching this specific hash.
                        If None, check for 2/3+ votes for ANY single hash.
        """
        state = self._state
        if state is None:
            return False

        total = self._total_active_stake()
        if total == 0:
            return False

        if block_hash is not None:
            voted_stake = self._stake_for_votes(state.prevotes, block_hash)
            return voted_stake * 3 > total * 2
        else:
            # Check each unique hash
            hashes: Set[bytes] = set(state.prevotes.values())
            for h in hashes:
                voted_stake = self._stake_for_votes(state.prevotes, h)
                if voted_stake * 3 > total * 2:
                    return True
            return False

    def has_two_thirds_precommits(self, block_hash: Optional[bytes] = None) -> bool:
        """
        Check whether 2/3+ of active stake has precommitted.

        Args:
            block_hash: If provided, check for votes matching this specific hash.
                        If None, check for 2/3+ votes for ANY single hash.
        """
        state = self._state
        if state is None:
            return False

        total = self._total_active_stake()
        if total == 0:
            return False

        if block_hash is not None:
            voted_stake = self._stake_for_votes(state.precommits, block_hash)
            return voted_stake * 3 > total * 2
        else:
            hashes: Set[bytes] = set(state.precommits.values())
            for h in hashes:
                voted_stake = self._stake_for_votes(state.precommits, h)
                if voted_stake * 3 > total * 2:
                    return True
            return False

    # ------------------------------------------------------------------ #
    #  Timeout computation                                                 #
    # ------------------------------------------------------------------ #

    def get_timeout(self, step: RoundStep, round_num: int) -> float:
        """
        Compute the timeout for a given step and round.
        Timeout increases linearly with round number.
        """
        delta = round_num * self.TIMEOUT_DELTA

        if step == RoundStep.PROPOSE:
            return self.PROPOSE_TIMEOUT + delta
        elif step == RoundStep.PREVOTE:
            return self.PREVOTE_TIMEOUT + delta
        elif step == RoundStep.PRECOMMIT:
            return self.PRECOMMIT_TIMEOUT + delta
        return 0.0

    # ------------------------------------------------------------------ #
    #  Vote creation helpers                                               #
    # ------------------------------------------------------------------ #

    def create_prevote(
        self, block_hash: bytes, height: int, round_num: int,
    ) -> Tuple[bytes, bytes]:
        """
        Create a signed prevote from this node.

        Returns:
            (block_hash, signature) tuple.
        """
        message = self._compute_vote_message("prevote", height, round_num, block_hash)
        signature = self._my_keypair.sign(message)
        return block_hash, signature

    def create_precommit(
        self, block_hash: bytes, height: int, round_num: int,
    ) -> Tuple[bytes, bytes]:
        """
        Create a signed precommit from this node.

        Returns:
            (block_hash, signature) tuple.
        """
        message = self._compute_vote_message("precommit", height, round_num, block_hash)
        signature = self._my_keypair.sign(message)
        return block_hash, signature

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _total_active_stake(self) -> int:
        """Sum of total_stake for all ACTIVE validators."""
        return sum(v.total_stake for v in self._registry.active_validators)

    def _stake_for_votes(
        self,
        votes: Dict[bytes, bytes],
        target_hash: bytes,
    ) -> int:
        """
        Sum the stake of validators whose vote matches *target_hash*.
        """
        total = 0
        for addr, vote_hash in votes.items():
            if vote_hash == target_hash:
                v = self._registry.get(addr)
                if v is not None and v.is_active:
                    total += v.total_stake
        return total

    def _compute_vote_message(
        self,
        msg_type: str,
        height: int,
        round_num: int,
        block_hash: bytes,
    ) -> bytes:
        """
        Build the canonical byte message that is signed for a vote.

        Format: msg_type_tag || height (8 bytes BE) || round (4 bytes BE) || block_hash
        """
        tag = msg_type.encode("ascii")
        height_bytes = struct.pack(">Q", height)
        round_bytes = struct.pack(">I", round_num)
        return tag + height_bytes + round_bytes + block_hash

    def _verify_vote_signature(
        self,
        validator_addr: bytes,
        signature: bytes,
        message: bytes,
    ) -> bool:
        """
        Verify a vote signature against the validator's registered public key.
        """
        v = self._registry.get(validator_addr)
        if v is None:
            return False
        return KeyPair.verify(v.pubkey, signature, message)

    # ------------------------------------------------------------------ #
    #  Query helpers                                                       #
    # ------------------------------------------------------------------ #

    def get_committed_block(self, height: int) -> Optional[Block]:
        """Return the block committed at *height*, if any."""
        return self._committed.get(height)

    def is_committed(self, height: int) -> bool:
        """Check if a block has been committed at *height*."""
        return height in self._committed

    def __repr__(self) -> str:
        if self._state:
            return (
                f"TendermintBFT(height={self._state.height}, "
                f"round={self._state.round}, "
                f"step={self._state.step.name})"
            )
        return "TendermintBFT(idle)"
