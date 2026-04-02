"""
Positronic - Censorship Detector (Phase 15)
Detects validators who systematically exclude valid transactions
from their proposed blocks — possible cartel/MEV censoring.

Monitors:
1. TX inclusion rate per validator over sliding epoch window
2. High-fee TX exclusion pattern (cherry-picking)
3. Reports flagged validators to the neural immune system
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from positronic.constants import (
    CENSORSHIP_WINDOW_EPOCHS,
    CENSORSHIP_MIN_INCLUSION_RATE,
    CENSORSHIP_HIGH_FEE_EXCLUSION,
)


@dataclass
class ValidatorInclusionRecord:
    """Tracks a single validator's TX inclusion behavior."""
    address: bytes
    epoch_rates: List[float] = field(default_factory=list)
    high_fee_exclusions: int = 0
    total_blocks_proposed: int = 0
    total_txs_included: int = 0
    total_txs_available: int = 0
    flagged: bool = False
    flag_count: int = 0


class CensorshipDetector:
    """
    Detect validators who systematically exclude valid transactions.

    Algorithm:
    - Track inclusion rate per validator per epoch
    - If avg inclusion rate < 70% over 3 epochs, flag validator
    - If >5 high-fee TXs excluded, flag validator
    - Report flagged validators to immune system for further action
    """

    def __init__(self):
        self._records: Dict[bytes, ValidatorInclusionRecord] = {}
        self._current_epoch: int = 0
        self._epoch_data: Dict[bytes, Dict] = {}  # per-epoch accumulator

        # Stats
        self.total_blocks_analyzed: int = 0
        self.total_flagged: int = 0

    def on_block_produced(
        self,
        proposer: bytes,
        included_count: int,
        available_count: int,
        high_fee_excluded: int = 0,
    ):
        """
        Track what a proposer included vs what was available.

        Args:
            proposer: Validator address
            included_count: Number of TXs included in block
            available_count: Number of valid TXs in mempool
            high_fee_excluded: Number of high-fee TXs NOT included
        """
        if proposer not in self._records:
            self._records[proposer] = ValidatorInclusionRecord(address=proposer)

        record = self._records[proposer]
        record.total_blocks_proposed += 1
        record.total_txs_included += included_count
        record.total_txs_available += available_count
        record.high_fee_exclusions += high_fee_excluded

        # Track in epoch accumulator
        if proposer not in self._epoch_data:
            self._epoch_data[proposer] = {
                "included": 0, "available": 0, "blocks": 0
            }
        self._epoch_data[proposer]["included"] += included_count
        self._epoch_data[proposer]["available"] += available_count
        self._epoch_data[proposer]["blocks"] += 1

        self.total_blocks_analyzed += 1

    def on_epoch_end(self, epoch: int) -> List[bytes]:
        """
        Check validators for censorship patterns at epoch boundary.
        Returns list of flagged validator addresses.
        """
        self._current_epoch = epoch
        flagged = []

        # Compute inclusion rate for this epoch
        for addr, data in self._epoch_data.items():
            available = data["available"]
            # If no transactions were available, skip — empty mempool is not censorship
            if available == 0:
                continue
            rate = data["included"] / available

            record = self._records.get(addr)
            if record:
                record.epoch_rates.append(rate)
                # Keep only recent epochs
                if len(record.epoch_rates) > CENSORSHIP_WINDOW_EPOCHS * 2:
                    record.epoch_rates = record.epoch_rates[
                        -CENSORSHIP_WINDOW_EPOCHS * 2:
                    ]

        # Check for censorship patterns
        for addr, record in self._records.items():
            if len(record.epoch_rates) >= CENSORSHIP_WINDOW_EPOCHS:
                recent = record.epoch_rates[-CENSORSHIP_WINDOW_EPOCHS:]
                avg_rate = sum(recent) / len(recent)

                if avg_rate < CENSORSHIP_MIN_INCLUSION_RATE:
                    if not record.flagged:
                        record.flagged = True
                        record.flag_count += 1
                        self.total_flagged += 1
                    flagged.append(addr)
                else:
                    # Rehabilitate: unflag if behavior improves
                    record.flagged = False

            # High-fee exclusion check
            if record.high_fee_exclusions > CENSORSHIP_HIGH_FEE_EXCLUSION:
                if addr not in flagged:
                    if not record.flagged:
                        record.flagged = True
                        record.flag_count += 1
                        self.total_flagged += 1
                    flagged.append(addr)

        # Reset epoch accumulator
        self._epoch_data = {}

        return flagged

    def get_validator_record(self, address: bytes) -> Optional[dict]:
        """Get censorship record for a validator."""
        record = self._records.get(address)
        if not record:
            return None
        total_available = max(record.total_txs_available, 1)
        return {
            "address": address.hex(),
            "total_blocks": record.total_blocks_proposed,
            "inclusion_rate": record.total_txs_included / total_available,
            "epoch_rates": record.epoch_rates[-CENSORSHIP_WINDOW_EPOCHS:],
            "high_fee_exclusions": record.high_fee_exclusions,
            "flagged": record.flagged,
            "flag_count": record.flag_count,
        }

    def reset_validator(self, address: bytes):
        """Reset a validator's record (after governance review)."""
        if address in self._records:
            record = self._records[address]
            record.flagged = False
            record.high_fee_exclusions = 0
            record.epoch_rates = []

    def get_stats(self) -> dict:
        flagged_count = sum(
            1 for r in self._records.values() if r.flagged
        )
        return {
            "total_validators_tracked": len(self._records),
            "currently_flagged": flagged_count,
            "total_flags_ever": self.total_flagged,
            "total_blocks_analyzed": self.total_blocks_analyzed,
            "current_epoch": self._current_epoch,
        }
