"""
Positronic - Dividend Distribution Engine (Phase 30)
Automatic pro-rata dividend payments for RWA token holders.

When an issuer distributes dividends, each holder receives
a share proportional to their token holdings.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.constants import RWA_DIVIDEND_MIN_AMOUNT, RWA_DIVIDEND_FEE


@dataclass
class DividendRecord:
    """A single dividend distribution event."""
    dividend_id: str
    token_id: str
    total_amount: int           # Total ASF distributed
    amount_per_token: int       # ASF per token unit (scaled)
    holder_count: int           # Number of holders at snapshot
    distributed_at: float = 0.0
    issuer: bytes = b""

    # Per-holder payouts
    payouts: Dict[bytes, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dividend_id": self.dividend_id,
            "token_id": self.token_id,
            "total_amount": self.total_amount,
            "amount_per_token": self.amount_per_token,
            "holder_count": self.holder_count,
            "distributed_at": self.distributed_at,
            "issuer": self.issuer.hex() if self.issuer else "",
        }


class DividendEngine:
    """Distributes dividends to RWA token holders pro-rata.

    Flow:
    1. Issuer calls distribute() with total ASF amount
    2. Engine snapshots current holders and their balances
    3. Calculates per-holder share proportional to holdings
    4. Returns payout map for the blockchain to execute transfers
    """

    def __init__(self):
        self._dividends: Dict[str, DividendRecord] = {}
        self._token_dividends: Dict[str, List[str]] = {}  # token_id -> [dividend_ids]
        self._counter: int = 0
        self._total_distributed: int = 0

    def distribute(
        self,
        token_id: str,
        total_amount: int,
        holders: Dict[bytes, int],
        total_supply: int,
        issuer: bytes = b"",
    ) -> Optional[DividendRecord]:
        """Calculate pro-rata dividend distribution.

        Args:
            token_id: The RWA token ID.
            total_amount: Total ASF to distribute.
            holders: {address: balance} mapping of current holders.
            total_supply: Total supply of the token.
            issuer: Address of the issuer triggering distribution.

        Returns:
            DividendRecord with payouts calculated, or None on error.
        """
        if total_amount < RWA_DIVIDEND_MIN_AMOUNT:
            return None

        if not holders or total_supply <= 0:
            return None

        # Calculate per-holder payouts (pro-rata)
        payouts: Dict[bytes, int] = {}
        distributed = 0

        for address, balance in holders.items():
            if balance <= 0:
                continue
            # Proportional share: (balance / total_supply) * total_amount
            share = (balance * total_amount) // total_supply
            if share > 0:
                payouts[address] = share
                distributed += share

        if not payouts:
            return None

        # Amount per token unit (for record keeping, scaled by BASE_UNIT)
        amount_per_token = total_amount * (10**18) // total_supply

        self._counter += 1
        dividend_id = f"DIV-{token_id}-{self._counter:04d}"
        now = time.time()

        record = DividendRecord(
            dividend_id=dividend_id,
            token_id=token_id,
            total_amount=distributed,  # Actual distributed (may differ due to rounding)
            amount_per_token=amount_per_token,
            holder_count=len(payouts),
            distributed_at=now,
            issuer=issuer,
            payouts=payouts,
        )

        self._dividends[dividend_id] = record
        self._token_dividends.setdefault(token_id, []).append(dividend_id)
        self._total_distributed += distributed

        return record

    # ---- Queries ----

    def get_dividend(self, dividend_id: str) -> Optional[DividendRecord]:
        return self._dividends.get(dividend_id)

    def get_token_dividends(
        self, token_id: str, limit: int = 20
    ) -> List[DividendRecord]:
        """Get dividend history for a token."""
        ids = self._token_dividends.get(token_id, [])
        records = [self._dividends[did] for did in ids if did in self._dividends]
        return sorted(records, key=lambda r: r.distributed_at, reverse=True)[:limit]

    def get_holder_dividends(
        self, address: bytes, limit: int = 20
    ) -> List[dict]:
        """Get all dividends received by an address."""
        results = []
        for record in self._dividends.values():
            payout = record.payouts.get(address)
            if payout and payout > 0:
                results.append({
                    "dividend_id": record.dividend_id,
                    "token_id": record.token_id,
                    "amount": payout,
                    "distributed_at": record.distributed_at,
                })
        results.sort(key=lambda r: r["distributed_at"], reverse=True)
        return results[:limit]

    def get_stats(self) -> dict:
        return {
            "total_distributions": len(self._dividends),
            "total_distributed": self._total_distributed,
            "tokens_with_dividends": len(self._token_dividends),
        }
