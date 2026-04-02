"""
Positronic - Transaction Priority Pipeline

Classifies transactions into priority lanes for smarter block ordering.
Fast simple transfers get processed first, complex contract operations last.

Phase 17 GOD CHAIN addition.

Lane priority (processed in this order):
    SYSTEM  → Block rewards, treasury distributions (always first)
    FAST    → Simple value transfers (no data, low gas)
    STANDARD → Contract calls, moderate complexity
    HEAVY   → Contract deployments, high-gas operations

**Fail-open**: If classification fails for any reason, the transaction is
placed in the STANDARD lane. No transaction is ever rejected by the pipeline.
"""

from enum import IntEnum
from typing import List

from positronic.utils.logging import get_logger
from positronic.constants import TX_LANE_FAST_MAX_GAS, TX_LANE_HEAVY_MIN_GAS

logger = get_logger(__name__)


class TxLane(IntEnum):
    """Transaction priority lane. Lower value = higher priority."""
    SYSTEM = 0    # Rewards, treasury — always first
    FAST = 1      # Simple transfers — second priority
    STANDARD = 2  # Contract calls — normal priority
    HEAVY = 3     # Deploys, high-gas — last


class TxPriorityClassifier:
    """Classifies transactions into priority lanes.

    Uses transaction type, data presence, and gas limit to assign each
    transaction to the appropriate lane. The classifier is purely
    advisory — it never rejects or modifies transactions.

    Example::

        classifier = TxPriorityClassifier()
        lane = classifier.classify(tx)
        ordered = classifier.get_lane_ordering(pending_txs)
    """

    def classify(self, tx) -> TxLane:
        """Classify a transaction into a priority lane.

        Args:
            tx: Transaction object with ``tx_type``, ``data``, ``gas_limit``
                attributes.

        Returns:
            TxLane enum value. Falls back to STANDARD on any error.
        """
        try:
            tx_type_name = ""
            if hasattr(tx, "tx_type"):
                tx_type = tx.tx_type
                # Handle both enum and string types
                if hasattr(tx_type, "name"):
                    tx_type_name = tx_type.name.upper()
                elif hasattr(tx_type, "value"):
                    tx_type_name = str(tx_type.value).upper()
                else:
                    tx_type_name = str(tx_type).upper()

            # System transactions (rewards, treasury)
            if tx_type_name in ("REWARD", "AI_TREASURY", "GAME_REWARD"):
                return TxLane.SYSTEM

            # Get data and gas_limit safely
            data = getattr(tx, "data", b"") or b""
            gas_limit = getattr(tx, "gas_limit", 0) or 0

            # Contract creation always heavy
            if tx_type_name == "CONTRACT_CREATE":
                return TxLane.HEAVY

            # High gas operations go to heavy lane
            if gas_limit >= TX_LANE_HEAVY_MIN_GAS:
                return TxLane.HEAVY

            # Simple transfers (no data, low gas) go to fast lane
            if tx_type_name in ("TRANSFER", "STAKE", "UNSTAKE"):
                if len(data) == 0 and gas_limit <= TX_LANE_FAST_MAX_GAS:
                    return TxLane.FAST

            # Everything else is standard
            return TxLane.STANDARD

        except Exception as e:
            # Fail-open: unknown → standard lane
            logger.debug("TX classification error (fail-open): %s", e)
            return TxLane.STANDARD

    def get_lane_ordering(self, transactions: list) -> list:
        """Order transactions by lane priority, then by gas price within each lane.

        Within each lane, transactions are sorted by gas_price descending
        (highest bidder first), then by timestamp ascending (FIFO for ties).

        Args:
            transactions: List of Transaction objects.

        Returns:
            Sorted list of transactions. Original list is not modified.
        """
        if not transactions:
            return []

        try:
            # Classify each transaction
            classified = []
            for tx in transactions:
                lane = self.classify(tx)
                gas_price = getattr(tx, "gas_price", 0) or 0
                timestamp = getattr(tx, "timestamp", 0) or 0
                classified.append((lane, -gas_price, timestamp, tx))

            # Sort by: lane asc, gas_price desc (negated), timestamp asc
            classified.sort(key=lambda x: (x[0], x[1], x[2]))

            return [item[3] for item in classified]

        except Exception as e:
            # Fail-open: return original order
            logger.debug("Lane ordering error (fail-open): %s", e)
            return list(transactions)

    def get_lane_stats(self, transactions: list) -> dict:
        """Get distribution statistics across lanes.

        Args:
            transactions: List of Transaction objects.

        Returns:
            Dictionary with lane names as keys and counts as values.
        """
        stats = {lane.name: 0 for lane in TxLane}
        for tx in transactions:
            lane = self.classify(tx)
            stats[lane.name] += 1
        return stats
