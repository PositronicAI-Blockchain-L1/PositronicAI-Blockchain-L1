"""
Positronic - Adaptive Gas Oracle (EIP-1559 Style)

Provides intelligent gas pricing recommendations based on network utilization.
The oracle adjusts a base fee up or down depending on how full recent blocks are.

Phase 17 GOD CHAIN addition.

**CRITICAL**: This is advisory only. Transactions are NEVER rejected based on
gas price. Low-gas transactions simply get lower priority in lane ordering.
Every valid transaction eventually goes through.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger
from positronic.constants import (
    GAS_BASE_FEE_FLOOR,
    GAS_BASE_FEE_CEILING,
    GAS_BASE_FEE_CHANGE_DENOM,
    GAS_FEE_HISTORY_SIZE,
    TARGET_GAS_UTILIZATION,
    BLOCK_GAS_LIMIT,
)


logger = get_logger(__name__)


@dataclass
class BlockFeeData:
    """Fee data recorded for a single block."""
    height: int
    gas_used: int
    gas_limit: int
    base_fee: int
    timestamp: float
    tx_count: int = 0


class GasOracle:
    """EIP-1559-style adaptive gas price oracle.

    Tracks network utilization across recent blocks and adjusts a ``base_fee``
    that rises when blocks are more than 50% full and falls when they are
    under-utilized. The base fee is bounded by ``GAS_BASE_FEE_FLOOR`` (1) and
    ``GAS_BASE_FEE_CEILING`` (1000).

    The oracle also provides fee suggestions (low/medium/high priority) so
    wallets can offer sensible defaults.

    **Fail-open**: All methods return sensible defaults on error. A broken
    oracle never blocks transactions.

    Example::

        oracle = GasOracle()
        oracle.update_base_fee(15_000_000, 30_000_000)
        suggestion = oracle.get_fee_suggestion()
        # {'base_fee': 1, 'priority_fee_low': 1, ...}
    """

    def __init__(self):
        self._base_fee: int = GAS_BASE_FEE_FLOOR
        self._fee_history: List[BlockFeeData] = []
        self._max_history: int = GAS_FEE_HISTORY_SIZE

    @property
    def base_fee(self) -> int:
        """Current base fee."""
        return self._base_fee

    def update_base_fee(
        self,
        block_gas_used: int,
        block_gas_limit: int,
        block_height: int = 0,
        tx_count: int = 0,
    ) -> int:
        """Update the base fee after a new block is confirmed.

        If the block used more than 50% of its gas limit, the base fee
        increases by up to 12.5%. If less, it decreases proportionally.
        The fee is always clamped to ``[FLOOR, CEILING]``.

        Args:
            block_gas_used: Total gas consumed in the block.
            block_gas_limit: Gas limit of the block.
            block_height: Block height (for history tracking).
            tx_count: Number of transactions in the block.

        Returns:
            The new base fee value.
        """
        try:
            if block_gas_limit <= 0:
                return self._base_fee

            utilization = block_gas_used / block_gas_limit
            target = TARGET_GAS_UTILIZATION  # 0.5

            if utilization > target:
                # Over target — increase base fee
                excess = utilization - target
                change = max(1, self._base_fee * excess // GAS_BASE_FEE_CHANGE_DENOM)
                self._base_fee = min(
                    self._base_fee + int(change),
                    GAS_BASE_FEE_CEILING,
                )
            else:
                # Under target — decrease base fee
                deficit = target - utilization
                change = max(1, self._base_fee * deficit // GAS_BASE_FEE_CHANGE_DENOM)
                self._base_fee = max(
                    self._base_fee - int(change),
                    GAS_BASE_FEE_FLOOR,
                )

            # Record history
            entry = BlockFeeData(
                height=block_height,
                gas_used=block_gas_used,
                gas_limit=block_gas_limit,
                base_fee=self._base_fee,
                timestamp=time.time(),
                tx_count=tx_count,
            )
            self._fee_history.append(entry)
            if len(self._fee_history) > self._max_history:
                self._fee_history = self._fee_history[-self._max_history:]

            return self._base_fee

        except Exception as e:
            logger.debug("Gas oracle update_base_fee error (fail-open): %s", e)
            return self._base_fee

    def get_fee_suggestion(self) -> Dict[str, int]:
        """Get fee suggestions for wallets and users.

        Returns:
            Dictionary with keys:
                - ``base_fee``: Current base fee.
                - ``priority_fee_low``: Suggested tip for low priority.
                - ``priority_fee_medium``: Suggested tip for medium priority.
                - ``priority_fee_high``: Suggested tip for high priority.
        """
        try:
            bf = self._base_fee
            return {
                "base_fee": bf,
                "priority_fee_low": max(1, bf // 10),
                "priority_fee_medium": max(1, bf // 4),
                "priority_fee_high": max(1, bf // 2),
            }
        except Exception as e:
            logger.debug("Gas oracle get_fee_suggestion error (fail-open): %s", e)
            return {
                "base_fee": GAS_BASE_FEE_FLOOR,
                "priority_fee_low": 1,
                "priority_fee_medium": 1,
                "priority_fee_high": 1,
            }

    def estimate_priority(self, gas_price: int) -> str:
        """Estimate the priority level for a given gas price.

        Args:
            gas_price: The gas price offered by the transaction.

        Returns:
            One of ``"low"``, ``"medium"``, ``"high"``, or ``"urgent"``.
        """
        try:
            suggestion = self.get_fee_suggestion()
            total_medium = suggestion["base_fee"] + suggestion["priority_fee_medium"]
            total_high = suggestion["base_fee"] + suggestion["priority_fee_high"]

            if gas_price >= total_high * 2:
                return "urgent"
            elif gas_price >= total_high:
                return "high"
            elif gas_price >= total_medium:
                return "medium"
            else:
                return "low"
        except Exception as e:
            logger.debug("Gas oracle estimate_priority error (fail-open): %s", e)
            return "medium"

    def get_fee_history(self, block_count: int = 10) -> List[Dict]:
        """Get fee history for the last N blocks.

        Args:
            block_count: Number of recent blocks to return. Defaults to 10.

        Returns:
            List of dictionaries with block fee data.
        """
        try:
            recent = self._fee_history[-block_count:]
            return [
                {
                    "height": entry.height,
                    "gas_used": entry.gas_used,
                    "gas_limit": entry.gas_limit,
                    "base_fee": entry.base_fee,
                    "utilization": (
                        round(entry.gas_used / entry.gas_limit, 4)
                        if entry.gas_limit > 0 else 0.0
                    ),
                    "tx_count": entry.tx_count,
                }
                for entry in recent
            ]
        except Exception as e:
            logger.debug("Gas oracle get_fee_history error (fail-open): %s", e)
            return []

    def get_stats(self) -> Dict:
        """Return oracle state for monitoring.

        Returns:
            Dictionary with current base fee, history size, and averages.
        """
        avg_utilization = 0.0
        if self._fee_history:
            utils = [
                e.gas_used / e.gas_limit
                for e in self._fee_history
                if e.gas_limit > 0
            ]
            if utils:
                avg_utilization = round(sum(utils) / len(utils), 4)

        return {
            "base_fee": self._base_fee,
            "history_size": len(self._fee_history),
            "avg_utilization": avg_utilization,
            "fee_floor": GAS_BASE_FEE_FLOOR,
            "fee_ceiling": GAS_BASE_FEE_CEILING,
        }
