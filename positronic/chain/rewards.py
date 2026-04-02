"""
Positronic - Block Rewards and Fee Distribution
Bitcoin-style mining with halving schedule.
Hard cap: 500M ASF from mining, 200M from play-to-mine.

Three-Layer Consensus v2 Reward Distribution:
  Block Reward (24 ASF, halving every 2 years):
    ~17% Block Producer  — 4 ASF to the elected block producer
     50% Attesters       — 12 ASF to all active validators (pro-rata by stake)
    ~17% NVN/Validators  — 4 ASF to NVN operators; fallback: active validators
    ~17% DAO Treasury    — 4 ASF to community pool (DAO-governed)

  Fee Distribution (per-transaction):
    25% Block Producer
    25% Attesters (pro-rata by stake)
    20% Node Operators / Community pool
    10% Treasury (DAO-governed)
    20% Burned (deflationary pressure, remainder after rounding)

Note: create_burn_transaction() and create_treasury_transaction() are
utility/audit methods for programmatic fee estimation. The actual
fee distribution happens as direct state adjustments in executor.py's
_distribute_fees() method during block execution.
"""

from typing import List

from positronic.core.transaction import Transaction, TxType
from positronic.core.state import StateManager
from positronic.crypto.address import TREASURY_ADDRESS, BURN_ADDRESS
from positronic.constants import (
    INITIAL_BLOCK_REWARD,
    HALVING_INTERVAL,
    MINING_SUPPLY_CAP,
    TAIL_EMISSION,
    FEE_PRODUCER_SHARE,
    FEE_ATTESTER_SHARE,
    FEE_NODE_SHARE,
    FEE_TREASURY_SHARE,
    FEE_BURN_SHARE,
    PRODUCER_REWARD_SHARE,
    ATTESTATION_REWARD_SHARE,
    TREASURY_REWARD_SHARE,
    BASE_UNIT,
    # Backward-compat aliases (used by audit methods)
    FEE_VALIDATOR_SHARE,
    FEE_NVN_SHARE,
)


class RewardCalculator:
    """Calculates and distributes block rewards with Bitcoin-style halving."""

    def __init__(self):
        self._total_mined = 0  # Track cumulative mining emissions

    def get_block_reward(self, block_height: int) -> int:
        """
        Calculate block reward with halving schedule.
        Starts at 24 ASF, halves every ~5.2M blocks (~2 years).
        Hard cap: after mining supply exhausted, returns TAIL_EMISSION
        (0.5 ASF) to maintain perpetual validator incentive.
        """
        halvings = block_height // HALVING_INTERVAL
        if halvings >= 64:
            return TAIL_EMISSION

        reward = INITIAL_BLOCK_REWARD >> halvings

        # Hard cap enforcement: check if we'd exceed mining supply
        emitted = self.get_total_emitted(block_height)
        if emitted >= MINING_SUPPLY_CAP:
            return TAIL_EMISSION
        if emitted + reward > MINING_SUPPLY_CAP:
            return max(MINING_SUPPLY_CAP - emitted, TAIL_EMISSION)

        return max(reward, TAIL_EMISSION)

    def create_reward_transaction(
        self,
        block_height: int,
        validator_address: bytes,
        active_validator_count: int = 1,
    ) -> Transaction:
        """Create the block reward transaction.

        The TX value is the PRODUCER share only (~17% of block reward = 4 ASF).
        Attesters (50%), NVN/validators (17%), and treasury (17%) are distributed
        separately via _distribute_attestation_rewards() in blockchain.py.
        Solo mode (1 validator): producer gets 100%.
        """
        reward = self.get_block_reward(block_height)

        if reward <= 0:
            return None

        split = self.get_reward_split(reward, active_validator_count)
        producer_reward = split["producer"]

        return Transaction(
            tx_type=TxType.REWARD,
            nonce=0,
            sender=b"\x00" * 32,  # System
            recipient=validator_address,
            value=producer_reward,
            gas_price=0,
            gas_limit=0,
        )

    def create_burn_transaction(
        self,
        total_fees: int,
    ) -> Transaction:
        """Create fee burn transaction (20% of fees permanently destroyed)."""
        burn_amount = int(total_fees * FEE_BURN_SHARE)

        if burn_amount <= 0:
            return None

        return Transaction(
            tx_type=TxType.AI_TREASURY,  # Reuse type for system TX
            nonce=0,
            sender=b"\x00" * 32,  # System
            recipient=BURN_ADDRESS,
            value=burn_amount,
            gas_price=0,
            gas_limit=0,
        )

    def create_treasury_transaction(
        self,
        block_height: int,
        total_fees: int,
    ) -> Transaction:
        """Create the AI treasury allocation transaction (10% of fees)."""
        treasury_amount = int(total_fees * FEE_TREASURY_SHARE)

        if treasury_amount <= 0:
            return None

        return Transaction(
            tx_type=TxType.AI_TREASURY,
            nonce=0,
            sender=b"\x00" * 32,  # System
            recipient=TREASURY_ADDRESS,
            value=treasury_amount,
            gas_price=0,
            gas_limit=0,
        )

    def calculate_total_fees(self, transactions: List[Transaction]) -> int:
        """Calculate total fees from all transactions in a block."""
        total = 0
        for tx in transactions:
            if tx.tx_type not in (TxType.REWARD, TxType.AI_TREASURY):
                total += tx.gas_limit * tx.gas_price
        return total

    def get_reward_split(self, total_reward: int, active_validator_count: int = 2) -> dict:
        """
        Split block reward four ways: producer / attesters / nvn / treasury.
          producer  ~16.67% → 4 ASF  (block producer)
          attesters  50.00% → 12 ASF (all active validators, pro-rata by stake)
          nvn       ~16.67% → 4 ASF  (NVN operators; fallback: active validators)
          treasury  ~16.67% → 4 ASF  (DAO treasury / community pool)
        Solo mode: when 0 validators, producer gets 100%.
        """
        if active_validator_count <= 0:
            return {"producer": total_reward, "attesters": 0, "nodes": 0, "treasury": 0}
        producer = int(total_reward * PRODUCER_REWARD_SHARE)
        attesters = int(total_reward * ATTESTATION_REWARD_SHARE)
        treasury = int(total_reward * TREASURY_REWARD_SHARE)
        nodes = total_reward - producer - attesters - treasury  # remainder to nvn/validators
        return {"producer": producer, "attesters": attesters, "nodes": nodes, "treasury": treasury}

    def get_fee_distribution(self, total_fees: int, active_validator_count: int = 2) -> dict:
        """Calculate fee distribution breakdown (Consensus v2).
        Solo mode: producer gets everything except burn share."""
        burn = int(total_fees * FEE_BURN_SHARE)
        if active_validator_count <= 0:
            return {
                "producer": total_fees - burn,
                "attesters": 0,
                "nodes": 0,
                "treasury": 0,
                "burn": burn,
            }
        producer = int(total_fees * FEE_PRODUCER_SHARE)
        attesters = int(total_fees * FEE_ATTESTER_SHARE)
        nodes = int(total_fees * FEE_NODE_SHARE)
        treasury = int(total_fees * FEE_TREASURY_SHARE)
        burn = total_fees - producer - attesters - nodes - treasury
        return {
            "producer": producer,
            "attesters": attesters,
            "nodes": nodes,
            "treasury": treasury,
            "burn": burn,
        }

    def get_emission_schedule(self, blocks: int = 100) -> List[dict]:
        """Get the emission schedule for the next N blocks from current height."""
        schedule = []
        for i in range(blocks):
            reward = self.get_block_reward(i)
            schedule.append({
                "block": i,
                "reward": reward,
                "reward_sma": reward / BASE_UNIT,
            })
        return schedule

    def get_total_emitted(self, up_to_height: int) -> int:
        """Calculate total coins emitted up to a given height."""
        total = 0
        current_reward = INITIAL_BLOCK_REWARD
        height = 0

        while height < up_to_height:
            blocks_in_era = min(HALVING_INTERVAL, up_to_height - height)
            era_emission = current_reward * blocks_in_era

            # Cap at mining supply
            if total + era_emission > MINING_SUPPLY_CAP:
                remaining = MINING_SUPPLY_CAP - total
                total = MINING_SUPPLY_CAP
                break

            total += era_emission
            height += blocks_in_era
            current_reward >>= 1
            if current_reward == 0:
                break

        return min(total, MINING_SUPPLY_CAP)

    def get_mining_progress(self, block_height: int) -> dict:
        """Get mining progress as percentage of supply cap."""
        emitted = self.get_total_emitted(block_height)
        return {
            "mined_sma": emitted / BASE_UNIT,
            "cap_sma": MINING_SUPPLY_CAP / BASE_UNIT,
            "progress_pct": (emitted / MINING_SUPPLY_CAP) * 100,
            "remaining_sma": (MINING_SUPPLY_CAP - emitted) / BASE_UNIT,
            "current_reward_sma": self.get_block_reward(block_height) / BASE_UNIT,
        }
