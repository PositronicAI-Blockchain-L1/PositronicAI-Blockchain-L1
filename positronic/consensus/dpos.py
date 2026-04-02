"""
Positronic - Delegated Proof of Stake (DPoS) Consensus Engine
Top-level orchestrator that ties together slot timing, validator election,
proposer selection, staking, finality, slashing, and attestation tracking.

Consensus v2: Three-Layer System
    1.  SlotClock determines the current slot and epoch.
    2.  At each epoch boundary, ALL eligible validators are activated.
    3.  21 block producers are selected via weighted-random per epoch.
    4.  The assigned producer builds and signs the block.
    5.  ALL active validators attest; attestation rewards pro-rata by stake.
    6.  BFT finality requires 2/3 supermajority.
    7.  Missed slots and double-signs trigger slashing.
"""

from __future__ import annotations

import time
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.crypto.keys import KeyPair
from positronic.crypto.hashing import sha512, hash_to_int
from positronic.crypto.address import address_from_pubkey
from positronic.consensus.validator import (
    Validator,
    ValidatorRegistry,
    ValidatorStatus,
)
from positronic.consensus.slot import SlotClock, SlotInfo
from positronic.consensus.election import ValidatorElection, ElectionResult
from positronic.consensus.staking import StakingManager
from positronic.consensus.finality import FinalityTracker
from positronic.consensus.slashing import SlashingManager
from positronic.consensus.censorship_detector import CensorshipDetector
from positronic.consensus.attestation import AttestationTracker
from positronic.constants import (
    MAX_VALIDATORS,
    MIN_VALIDATORS,
    SLOTS_PER_EPOCH,
    BLOCK_TIME,
    EPOCH_DURATION,
    FINALITY_THRESHOLD,
    MIN_STAKE,
    FEE_VALIDATOR_SHARE,
    FEE_NVN_SHARE,
    HASH_SIZE,
)


@dataclass
class ConsensusState:
    """
    Snapshot of the consensus engine state at a point in time.

    Attributes:
        current_epoch:     Active epoch number.
        current_slot:      Active global slot number.
        last_finalized:    Slot of the most recently finalized block.
        last_block_hash:   Hash of the most recently accepted block.
        election:          Active ElectionResult for the current epoch.
        epoch_blocks:      Number of blocks produced in the current epoch.
        total_blocks:      Total blocks produced since genesis.
    """
    current_epoch: int = 0
    current_slot: int = 0
    last_finalized: int = 0
    last_block_hash: bytes = b"\x00" * HASH_SIZE
    election: Optional[ElectionResult] = None
    epoch_blocks: int = 0
    total_blocks: int = 0

    def to_dict(self) -> dict:
        return {
            "current_epoch": self.current_epoch,
            "current_slot": self.current_slot,
            "last_finalized": self.last_finalized,
            "last_block_hash": self.last_block_hash.hex(),
            "election": self.election.to_dict() if self.election else None,
            "epoch_blocks": self.epoch_blocks,
            "total_blocks": self.total_blocks,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConsensusState:
        election = None
        if d.get("election"):
            election = ElectionResult.from_dict(d["election"])
        return cls(
            current_epoch=d.get("current_epoch", 0),
            current_slot=d.get("current_slot", 0),
            last_finalized=d.get("last_finalized", 0),
            last_block_hash=bytes.fromhex(
                d.get("last_block_hash", "00" * HASH_SIZE)
            ),
            election=election,
            epoch_blocks=d.get("epoch_blocks", 0),
            total_blocks=d.get("total_blocks", 0),
        )


class DPoSConsensus:
    """
    Main DPoS consensus engine for the Positronic blockchain.

    Consensus v2: Three-Layer Validator System.
    Integrates:
        - SlotClock:            Wall-clock slot / epoch timing.
        - ValidatorRegistry:    Validator data store.
        - ValidatorElection:    Per-epoch election — ALL validators active.
        - StakingManager:       Self-bond, delegation, unbonding.
        - FinalityTracker:      BFT 2/3 attestation finality.
        - SlashingManager:      Double-sign and liveness penalties.
        - AttestationTracker:   Consensus v2 attestation reward tracking.
    """

    def __init__(self, genesis_time: float) -> None:
        # Core components
        self.clock = SlotClock(genesis_time)
        self.registry = ValidatorRegistry()
        self.election = ValidatorElection(self.registry)
        self.staking = StakingManager(self.registry)
        self.finality = FinalityTracker(self.registry)
        self.slashing = SlashingManager(self.registry)

        # Phase 15: Censorship detection
        self.censorship_detector = CensorshipDetector()

        # Consensus v2: Attestation tracking
        self.attestation_tracker = AttestationTracker()

        # Consensus state
        self.state = ConsensusState()

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        validators: Optional[List[dict]] = None,
    ) -> ElectionResult:
        """
        Bootstrap the consensus engine.

        *validators* is an optional list of dicts, each with:
            pubkey   (bytes): 32-byte Ed25519 public key
            stake    (int):   initial self-bonded stake
            is_nvn   (bool):  whether the node runs an NVN
            commission_rate (float): optional, default 0.05

        If not provided, the registry must already contain validators.
        Returns the ElectionResult for epoch 0.
        """
        if validators:
            for v in validators:
                self.staking.create_validator(
                    pubkey=v["pubkey"],
                    stake=v["stake"],
                    commission_rate=v.get("commission_rate", 0.05),
                    is_nvn=v.get("is_nvn", False),
                )

        # Run epoch-0 election with the genesis seed
        seed = ValidatorElection.genesis_seed()
        result = self.election.run_election(epoch=0, epoch_seed=seed)

        self.state.current_epoch = 0
        self.state.current_slot = 0
        self.state.election = result

        return result

    # ------------------------------------------------------------------ #
    #  Epoch transitions                                                  #
    # ------------------------------------------------------------------ #

    def on_epoch_boundary(
        self,
        new_epoch: int,
        last_block_hash: bytes,
    ) -> ElectionResult:
        """
        Called when the slot clock crosses an epoch boundary.

        1. Derive the new epoch seed from the last block hash.
        2. Process any pending unjails.
        3. Re-run the validator election.
        4. Reset epoch counters.

        Returns the new ElectionResult.
        """
        # Derive seed
        epoch_seed = ValidatorElection.derive_epoch_seed(
            last_block_hash, new_epoch,
        )

        # Attempt to unjail any jailed validators whose sentence expired
        for v in self.registry.all_validators:
            if v.is_jailed and time.time() >= v.jailed_until:
                self.registry.unjail(v.address)

        # Run election
        result = self.election.run_election(
            epoch=new_epoch,
            epoch_seed=epoch_seed,
        )

        # Phase 15: Censorship detection at epoch boundary
        flagged_censors = self.censorship_detector.on_epoch_end(new_epoch)

        # Update state
        self.state.current_epoch = new_epoch
        self.state.election = result
        self.state.epoch_blocks = 0
        self.state.last_block_hash = last_block_hash

        # Clear finality tracker for the new epoch
        self.finality.new_epoch()

        # Store flagged censors for immune system reporting
        self._last_flagged_censors = flagged_censors

        return result

    # ------------------------------------------------------------------ #
    #  Proposer selection                                                 #
    # ------------------------------------------------------------------ #

    def get_proposer(
        self,
        slot: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> Optional[bytes]:
        """
        Return the validator address that should propose the block for
        the given *slot* (or the current slot if None).
        """
        if slot is None:
            info = self.clock.current_slot(ts)
            slot = info.slot

        epoch = self.clock.slot_to_epoch(slot)
        slot_in_epoch = self.clock.slot_in_epoch(slot)

        return self.election.get_proposer_for_slot(epoch, slot_in_epoch)

    def is_my_slot(
        self,
        my_address: bytes,
        slot: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> bool:
        """Check whether *my_address* is the designated proposer for *slot*."""
        proposer = self.get_proposer(slot, ts)
        return proposer == my_address

    def get_proposer_validator(
        self,
        slot: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> Optional[Validator]:
        """Return the full Validator object for the slot's proposer."""
        addr = self.get_proposer(slot, ts)
        if addr is None:
            return None
        return self.registry.get(addr)

    # ------------------------------------------------------------------ #
    #  Block processing                                                   #
    # ------------------------------------------------------------------ #

    def on_block_proposed(
        self,
        slot: int,
        proposer_address: bytes,
        block_hash: bytes,
        timestamp: float,
    ) -> bool:
        """
        Notify the consensus engine that a block was proposed.

        Validates that the proposer matches the slot assignment.
        Records the proposal in the validator's stats.
        Updates consensus state.

        Returns True if the proposal is valid.
        """
        epoch = self.clock.slot_to_epoch(slot)
        slot_in_epoch = self.clock.slot_in_epoch(slot)

        # Verify proposer assignment (skip when no election exists,
        # e.g. single-validator mode where only the founder produces)
        expected = self.election.get_proposer_for_slot(epoch, slot_in_epoch)
        if expected is not None:
            if proposer_address != expected:
                return False

            # Check the block arrives within the slot window
            if not self.clock.is_within_slot_window(slot, timestamp):
                return False

        # Record proposal
        self.registry.record_proposed_block(proposer_address, timestamp)

        # Update state
        self.state.current_slot = slot
        self.state.last_block_hash = block_hash
        self.state.epoch_blocks += 1
        self.state.total_blocks += 1

        # Register block with finality tracker
        self.finality.register_block(slot, block_hash, proposer_address)

        # Sync last-finalized slot from finality tracker
        finalized_slot = self.finality.last_finalized_slot
        if finalized_slot > self.state.last_finalized:
            self.state.last_finalized = finalized_slot

        return True

    def on_block_missed(self, slot: int) -> Optional[bytes]:
        """
        Called when a slot passes with no valid block.
        Records the miss against the assigned proposer.
        Returns the address of the proposer who missed, or None.
        """
        epoch = self.clock.slot_to_epoch(slot)
        slot_in_epoch = self.clock.slot_in_epoch(slot)
        proposer = self.election.get_proposer_for_slot(epoch, slot_in_epoch)
        if proposer is None:
            return None

        self.registry.record_missed_block(proposer)

        # Check if slashing threshold is met
        validator = self.registry.get(proposer)
        if validator and self.slashing.should_slash_for_downtime(validator):
            self.slashing.slash_downtime(proposer)

        return proposer

    # ------------------------------------------------------------------ #
    #  Attestation and finality                                           #
    # ------------------------------------------------------------------ #

    def submit_attestation(
        self,
        slot: int,
        block_hash: bytes,
        validator_address: bytes,
        signature: bytes,
    ) -> bool:
        """
        Submit a validator attestation (vote) for a block.
        Returns True if accepted.
        """
        # Verify the attester is in the active set.
        # Fall back to registry check when no election exists for the
        # block's slot-based epoch (happens when wall-clock epoch diverges
        # from block slot numbers, e.g. after node restart).
        epoch = self.clock.slot_to_epoch(slot)
        if not self.election.is_validator_active(validator_address, epoch):
            validator = self.registry.get(validator_address)
            if validator is None or not validator.is_active:
                return False

        accepted = self.finality.add_attestation(
            slot=slot,
            block_hash=block_hash,
            validator_address=validator_address,
            signature=signature,
        )

        # Record in attestation tracker for reward distribution and stats
        if accepted:
            validator = self.registry.get(validator_address)
            stake = validator.total_stake if validator else 0
            self.attestation_tracker.record_attestation(
                block_hash, validator_address, stake,
            )

            # Sync last-finalized slot (attestation may have triggered finality)
            finalized_slot = self.finality.last_finalized_slot
            if finalized_slot > self.state.last_finalized:
                self.state.last_finalized = finalized_slot

        return accepted

    def is_finalized(self, slot: int, block_hash: bytes) -> bool:
        """Check whether a block has reached BFT finality."""
        return self.finality.is_finalized(slot, block_hash)

    # ------------------------------------------------------------------ #
    #  Double-sign detection                                              #
    # ------------------------------------------------------------------ #

    def report_double_sign(
        self,
        validator_address: bytes,
        block_hash_a: bytes,
        block_hash_b: bytes,
        slot: int,
        signature_a: bytes,
        signature_b: bytes,
    ) -> bool:
        """
        Submit evidence of a double-sign (two different blocks at the
        same slot by the same validator).  If valid, the validator is
        slashed and jailed.

        Returns True if the evidence was accepted and slashing applied.
        """
        return self.slashing.slash_double_sign(
            validator_address=validator_address,
            slot=slot,
            block_hash_a=block_hash_a,
            block_hash_b=block_hash_b,
            signature_a=signature_a,
            signature_b=signature_b,
        )

    # ------------------------------------------------------------------ #
    #  Fee distribution                                                   #
    # ------------------------------------------------------------------ #

    def distribute_block_fees(
        self,
        proposer_address: bytes,
        total_fees: int,
    ) -> Dict[bytes, int]:
        """
        Split block fees according to the tokenomics:
            30 % -> DPoS validator reward pool
            25 % -> NVN reward pool (split among NVN operators)
            20 % -> AI Treasury (handled externally)
            15 % -> Node operators / Community (handled externally)
            10 % -> Burn (handled externally)

        This method handles the 30 % validator share by depositing into
        the proposer's staking reward pool, and returns per-address NVN
        payouts for the 25 % NVN share.

        Returns a dict of {address: amount} for NVN payouts.
        """
        validator_share = int(total_fees * FEE_VALIDATOR_SHARE)
        nvn_share = int(total_fees * FEE_NVN_SHARE)

        # Deposit validator share into the proposer's reward pool
        if validator_share > 0:
            self.staking.deposit_reward(proposer_address, validator_share)

        # Split NVN share among active NVN operators
        nvn_payouts: Dict[bytes, int] = {}
        nvn_nodes = self.registry.nvn_validators
        if nvn_share > 0 and nvn_nodes:
            per_nvn = nvn_share // len(nvn_nodes)
            for nvn in nvn_nodes:
                if per_nvn > 0:
                    nvn_payouts[nvn.address] = per_nvn

        return nvn_payouts

    # ------------------------------------------------------------------ #
    #  Validation helpers                                                 #
    # ------------------------------------------------------------------ #

    def validate_block_proposer(
        self,
        slot: int,
        proposer_pubkey: bytes,
    ) -> bool:
        """
        Verify that the given public key is the legitimate proposer for
        the slot according to the current election.
        """
        expected_addr = self.get_proposer(slot)
        if expected_addr is None:
            return False
        actual_addr = address_from_pubkey(proposer_pubkey)
        return actual_addr == expected_addr

    def needs_epoch_transition(
        self, ts: Optional[float] = None,
    ) -> Tuple[bool, int]:
        """
        Check if the current time requires an epoch transition.
        Returns (needs_transition, new_epoch).
        """
        if ts is None:
            ts = time.time()
        current_epoch = self.clock.epoch_at_time(ts)
        needs = current_epoch > self.state.current_epoch
        return needs, current_epoch

    # ------------------------------------------------------------------ #
    #  Status and info                                                    #
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """Return a human-readable status summary."""
        info = self.clock.current_slot()
        proposer = self.get_proposer()
        proposer_hex = (
            "0x" + proposer.hex() if proposer else "unknown"
        )

        return {
            "epoch": info.epoch,
            "slot": info.slot,
            "slot_in_epoch": info.slot_in_epoch,
            "proposer": proposer_hex,
            "active_validators": self.registry.active_count,
            "total_validators": self.registry.total_count,
            "last_finalized_slot": self.state.last_finalized,
            "total_blocks": self.state.total_blocks,
            "epoch_blocks": self.state.epoch_blocks,
            "censorship": self.censorship_detector.get_stats(),
        }

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "genesis_time": self.clock.genesis_time,
            "state": self.state.to_dict(),
            "registry": self.registry.to_dict(),
            "election": self.election.to_dict(),
            "staking": self.staking.to_dict(),
            "finality": self.finality.to_dict(),
            "slashing": self.slashing.to_dict(),
            "attestation": self.attestation_tracker.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DPoSConsensus:
        engine = cls(genesis_time=d["genesis_time"])
        engine.state = ConsensusState.from_dict(d["state"])
        engine.registry = ValidatorRegistry.from_dict(d["registry"])
        engine.election = ValidatorElection.from_dict(
            d["election"], engine.registry,
        )
        engine.staking = StakingManager.from_dict(
            d["staking"], engine.registry,
        )
        engine.finality = FinalityTracker.from_dict(
            d["finality"], engine.registry,
        )
        engine.slashing = SlashingManager.from_dict(
            d["slashing"], engine.registry,
        )
        if "attestation" in d:
            engine.attestation_tracker = AttestationTracker.from_dict(
                d["attestation"]
            )
        return engine

    def __repr__(self) -> str:
        return (
            f"DPoSConsensus(epoch={self.state.current_epoch}, "
            f"slot={self.state.current_slot}, "
            f"validators={self.registry.active_count}/{self.registry.total_count})"
        )
