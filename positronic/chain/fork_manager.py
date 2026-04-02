"""
Positronic - Fork Manager
Handles chain reorganization when competing blocks arrive.
Implements longest-chain rule with finality guard.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from positronic.core.block import Block

logger = logging.getLogger(__name__)


@dataclass
class ForkCandidate:
    """A competing block at the same height as the canonical chain."""
    block: Block
    received_at: float = 0.0


class ForkManager:
    """
    Manages chain forks and reorganization.

    Rules:
    1. First valid block at a height wins (canonical)
    2. If a fork grows longer, switch to the longer chain
    3. Never reorg past a finalized checkpoint
    4. Orphaned transactions return to the mempool
    """

    def __init__(self, max_fork_depth: int = 64):
        self.max_fork_depth = max_fork_depth
        self._fork_blocks: Dict[int, List[ForkCandidate]] = {}
        self._finalized_height: int = 0

    @property
    def finalized_height(self) -> int:
        return self._finalized_height

    @finalized_height.setter
    def finalized_height(self, height: int):
        if height > self._finalized_height:
            self._finalized_height = height
            self._prune_old_forks(height)

    def has_competing_block(self, height: int) -> bool:
        """Check if there's already a fork candidate at this height."""
        return height in self._fork_blocks and len(self._fork_blocks[height]) > 0

    def store_fork_candidate(self, block: Block) -> bool:
        """
        Store a competing block for possible future reorg.
        Returns True if stored, False if rejected.
        """
        import time
        height = block.height

        # Never store forks at or below finalized height
        if height <= self._finalized_height:
            logger.debug("Rejecting fork at height %d (finalized at %d)", height, self._finalized_height)
            return False

        if height not in self._fork_blocks:
            self._fork_blocks[height] = []

        # Don't store duplicate blocks
        for candidate in self._fork_blocks[height]:
            if candidate.block.hash == block.hash:
                return False

        self._fork_blocks[height].append(ForkCandidate(block=block, received_at=time.time()))
        logger.info("Stored fork candidate at height %d (hash=%s...)", height, block.hash.hex()[:16] if block.hash else "none")
        return True

    def should_reorg(
        self,
        canonical_height: int,
        fork_tip_height: int,
    ) -> bool:
        """
        Determine if we should reorganize to the fork chain.
        Simple longest-chain rule: fork must be strictly longer.
        """
        if fork_tip_height <= canonical_height:
            return False
        # Don't reorg deeper than max_fork_depth
        depth = fork_tip_height - self._finalized_height
        if depth > self.max_fork_depth:
            logger.warning("Fork too deep (%d > %d), rejecting", depth, self.max_fork_depth)
            return False
        return True

    def get_reorg_range(
        self,
        canonical_height: int,
        fork_base_height: int,
    ) -> Tuple[int, int]:
        """
        Get the range of blocks to revert and reapply.
        Returns (revert_from, revert_to) -- blocks to undo.
        """
        revert_from = max(fork_base_height, self._finalized_height + 1)
        revert_to = canonical_height
        return revert_from, revert_to

    def collect_fork_chain(self, start_height: int, end_height: int) -> List[Block]:
        """Collect ordered fork blocks from start_height to end_height inclusive."""
        chain = []
        for h in range(start_height, end_height + 1):
            candidates = self._fork_blocks.get(h, [])
            if candidates:
                chain.append(candidates[0].block)
        return chain

    def clear_range(self, start_height: int, end_height: int):
        """Remove fork candidates in the given height range (inclusive)."""
        for h in range(start_height, end_height + 1):
            self._fork_blocks.pop(h, None)

    def get_fork_blocks(self, height: int) -> List[Block]:
        """Get all fork candidate blocks at a given height."""
        candidates = self._fork_blocks.get(height, [])
        return [c.block for c in candidates]

    def clear_fork(self, height: int):
        """Remove fork candidates at a specific height."""
        self._fork_blocks.pop(height, None)

    def _prune_old_forks(self, finalized_height: int):
        """Remove fork candidates at or below finalized height."""
        to_remove = [h for h in self._fork_blocks if h <= finalized_height]
        for h in to_remove:
            del self._fork_blocks[h]

    def get_stats(self) -> dict:
        """Return fork manager statistics."""
        return {
            "finalized_height": self._finalized_height,
            "pending_forks": sum(len(v) for v in self._fork_blocks.values()),
            "fork_heights": sorted(self._fork_blocks.keys()),
            "max_fork_depth": self.max_fork_depth,
        }
