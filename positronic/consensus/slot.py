"""
Positronic - Slot and Epoch Timing
3-second block slots, 32 slots per epoch (96 seconds per epoch).

A *slot* is the smallest time unit in which exactly one block may be proposed.
An *epoch* is a group of 32 consecutive slots after which the active validator
set may be reshuffled.

Slot index and epoch index both start at 0 (genesis).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from positronic.constants import BLOCK_TIME, SLOTS_PER_EPOCH, EPOCH_DURATION


@dataclass(frozen=True)
class SlotInfo:
    """
    Immutable snapshot describing a particular slot.

    Attributes:
        slot:            Global slot number (0-based, monotonically increasing).
        epoch:           Epoch this slot belongs to.
        slot_in_epoch:   Position within the epoch (0 .. SLOTS_PER_EPOCH-1).
        start_time:      Unix timestamp when this slot opens.
        end_time:        Unix timestamp when this slot closes.
    """
    slot: int
    epoch: int
    slot_in_epoch: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def is_first_in_epoch(self) -> bool:
        """True if this is the first slot of its epoch."""
        return self.slot_in_epoch == 0

    @property
    def is_last_in_epoch(self) -> bool:
        """True if this is the last slot of its epoch."""
        return self.slot_in_epoch == SLOTS_PER_EPOCH - 1

    def to_dict(self) -> dict:
        return {
            "slot": self.slot,
            "epoch": self.epoch,
            "slot_in_epoch": self.slot_in_epoch,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SlotInfo:
        return cls(
            slot=d["slot"],
            epoch=d["epoch"],
            slot_in_epoch=d["slot_in_epoch"],
            start_time=d["start_time"],
            end_time=d["end_time"],
        )

    def __repr__(self) -> str:
        return (
            f"SlotInfo(slot={self.slot}, epoch={self.epoch}, "
            f"pos={self.slot_in_epoch}/{SLOTS_PER_EPOCH})"
        )


class SlotClock:
    """
    Wall-clock based slot timer anchored to the genesis timestamp.

    Usage::

        clock = SlotClock(genesis_time=1700000000.0)
        info  = clock.current_slot()
        print(info.slot, info.epoch)
    """

    def __init__(self, genesis_time: float) -> None:
        """
        Args:
            genesis_time: Unix timestamp of the genesis block (slot 0 start).
        """
        if genesis_time <= 0:
            raise ValueError("genesis_time must be a positive Unix timestamp")
        self._genesis: float = genesis_time

    @property
    def genesis_time(self) -> float:
        return self._genesis

    # ------------------------------------------------------------------ #
    #  Slot / epoch arithmetic                                            #
    # ------------------------------------------------------------------ #

    def slot_at_time(self, ts: float) -> int:
        """Return the slot number for the given Unix timestamp."""
        if ts < self._genesis:
            return 0
        return int((ts - self._genesis) / BLOCK_TIME)

    def epoch_at_time(self, ts: float) -> int:
        """Return the epoch number for the given Unix timestamp."""
        return self.slot_at_time(ts) // SLOTS_PER_EPOCH

    def slot_to_epoch(self, slot: int) -> int:
        """Return the epoch a given slot belongs to."""
        return slot // SLOTS_PER_EPOCH

    def slot_in_epoch(self, slot: int) -> int:
        """Return the position of *slot* within its epoch (0-based)."""
        return slot % SLOTS_PER_EPOCH

    def epoch_start_slot(self, epoch: int) -> int:
        """First slot of the given epoch."""
        return epoch * SLOTS_PER_EPOCH

    def epoch_end_slot(self, epoch: int) -> int:
        """Last slot (inclusive) of the given epoch."""
        return (epoch + 1) * SLOTS_PER_EPOCH - 1

    # ------------------------------------------------------------------ #
    #  Timing helpers                                                     #
    # ------------------------------------------------------------------ #

    def slot_start_time(self, slot: int) -> float:
        """Unix timestamp when *slot* opens."""
        return self._genesis + slot * BLOCK_TIME

    def slot_end_time(self, slot: int) -> float:
        """Unix timestamp when *slot* closes (exclusive)."""
        return self._genesis + (slot + 1) * BLOCK_TIME

    def epoch_start_time(self, epoch: int) -> float:
        """Unix timestamp when *epoch* begins."""
        return self.slot_start_time(self.epoch_start_slot(epoch))

    def epoch_end_time(self, epoch: int) -> float:
        """Unix timestamp when *epoch* ends (exclusive)."""
        return self.slot_end_time(self.epoch_end_slot(epoch))

    def time_until_next_slot(self, ts: Optional[float] = None) -> float:
        """Seconds remaining until the next slot starts."""
        if ts is None:
            ts = time.time()
        current = self.slot_at_time(ts)
        return self.slot_start_time(current + 1) - ts

    def time_into_slot(self, ts: Optional[float] = None) -> float:
        """Seconds elapsed since the current slot started."""
        if ts is None:
            ts = time.time()
        current = self.slot_at_time(ts)
        return ts - self.slot_start_time(current)

    # ------------------------------------------------------------------ #
    #  SlotInfo builders                                                  #
    # ------------------------------------------------------------------ #

    def info_for_slot(self, slot: int) -> SlotInfo:
        """Build a SlotInfo for a specific slot number."""
        epoch = self.slot_to_epoch(slot)
        pos = self.slot_in_epoch(slot)
        return SlotInfo(
            slot=slot,
            epoch=epoch,
            slot_in_epoch=pos,
            start_time=self.slot_start_time(slot),
            end_time=self.slot_end_time(slot),
        )

    def current_slot(self, ts: Optional[float] = None) -> SlotInfo:
        """Build a SlotInfo for the slot active at *ts* (default: now)."""
        if ts is None:
            ts = time.time()
        slot = self.slot_at_time(ts)
        return self.info_for_slot(slot)

    def current_epoch(self, ts: Optional[float] = None) -> int:
        """Return the current epoch number."""
        if ts is None:
            ts = time.time()
        return self.epoch_at_time(ts)

    def slots_in_epoch(self, epoch: int) -> list[SlotInfo]:
        """Return SlotInfo objects for every slot in the given epoch."""
        start = self.epoch_start_slot(epoch)
        return [self.info_for_slot(start + i) for i in range(SLOTS_PER_EPOCH)]

    # ------------------------------------------------------------------ #
    #  Validation helpers                                                 #
    # ------------------------------------------------------------------ #

    def is_slot_current(self, slot: int, ts: Optional[float] = None) -> bool:
        """Check whether *slot* is the currently active slot."""
        if ts is None:
            ts = time.time()
        return self.slot_at_time(ts) == slot

    def is_within_slot_window(
        self, slot: int, ts: Optional[float] = None, tolerance: float = 1.0,
    ) -> bool:
        """
        Check whether *ts* falls within the slot window with a tolerance.
        Allows a block to arrive up to *tolerance* seconds late.
        """
        if ts is None:
            ts = time.time()
        start = self.slot_start_time(slot)
        end = self.slot_end_time(slot) + tolerance
        return start <= ts < end

    # ------------------------------------------------------------------ #
    #  Serialization                                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {"genesis_time": self._genesis}

    @classmethod
    def from_dict(cls, d: dict) -> SlotClock:
        return cls(genesis_time=d["genesis_time"])

    def __repr__(self) -> str:
        return f"SlotClock(genesis={self._genesis})"
