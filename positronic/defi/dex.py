"""
Positronic - Decentralized Exchange (DEX) with Automated Market Maker (AMM)

Uniswap-style constant product AMM (x * y = k).
Supports pool creation, liquidity provision, and token swaps with
configurable fee rates.
"""

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from positronic.constants import CHAIN_ID
from positronic.utils.logging import get_logger

logger = get_logger(__name__)

# Default fee: 0.3% (30 basis points)
DEFAULT_FEE_RATE = 0.003

# Minimum liquidity to prevent division-by-zero edge cases
MIN_LIQUIDITY = 1000


@dataclass
class LiquidityPool:
    """Represents a single AMM liquidity pool."""
    pool_id: str
    token_a_id: str
    token_b_id: str
    reserve_a: int
    reserve_b: int
    total_lp_shares: int
    fee_rate: float
    lp_holders: Dict[str, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    total_volume: int = 0
    total_fees_collected: int = 0
    swap_count: int = 0

    def to_dict(self) -> dict:
        """Serialize pool to dictionary."""
        return {
            "pool_id": self.pool_id,
            "token_a_id": self.token_a_id,
            "token_b_id": self.token_b_id,
            "reserve_a": self.reserve_a,
            "reserve_b": self.reserve_b,
            "total_lp_shares": self.total_lp_shares,
            "fee_rate": self.fee_rate,
            "lp_holder_count": len(self.lp_holders),
            "created_at": self.created_at,
            "total_volume": self.total_volume,
            "total_fees_collected": self.total_fees_collected,
            "swap_count": self.swap_count,
        }

    def to_full_dict(self) -> dict:
        """Serialize full pool state including lp_holders for persistence."""
        return {
            "pool_id": self.pool_id,
            "token_a_id": self.token_a_id,
            "token_b_id": self.token_b_id,
            "reserve_a": self.reserve_a,
            "reserve_b": self.reserve_b,
            "total_lp_shares": self.total_lp_shares,
            "fee_rate": self.fee_rate,
            "lp_holders": dict(self.lp_holders),
            "created_at": self.created_at,
            "total_volume": self.total_volume,
            "total_fees_collected": self.total_fees_collected,
            "swap_count": self.swap_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LiquidityPool":
        """Reconstruct LiquidityPool from a full_dict snapshot."""
        pool = cls(
            pool_id=d["pool_id"],
            token_a_id=d["token_a_id"],
            token_b_id=d["token_b_id"],
            reserve_a=d["reserve_a"],
            reserve_b=d["reserve_b"],
            total_lp_shares=d["total_lp_shares"],
            fee_rate=d["fee_rate"],
            lp_holders=dict(d.get("lp_holders", {})),
            created_at=d.get("created_at", 0.0),
            total_volume=d.get("total_volume", 0),
            total_fees_collected=d.get("total_fees_collected", 0),
            swap_count=d.get("swap_count", 0),
        )
        return pool


class DEXEngine:
    """
    Automated Market Maker (AMM) — Uniswap-style constant product DEX.

    Invariant: reserve_a * reserve_b = k  (maintained after every swap).
    Fee: deducted from input before computing output.
    LP shares: proportional to contributed liquidity.
    """

    def __init__(self, db_path: str = None):
        self._pools: Dict[str, LiquidityPool] = {}
        self._pair_index: Dict[str, str] = {}   # "tokenA:tokenB" -> pool_id
        self._nonce: int = 0
        self._total_pools_created: int = 0
        self._total_volume: int = 0
        self._total_swaps: int = 0

        # Persistence
        self._db_conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        if db_path is not None:
            self._init_dex_db(db_path)
            self._load_dex_from_db()
        self._balance_checker = None
        self._transfer_fn = None  # (from_addr, to_addr, token_id, amount) -> bool

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_pool_id(self, token_a: str, token_b: str) -> str:
        """Generate deterministic pool ID from token pair."""
        self._nonce += 1
        data = (
            token_a.encode()
            + token_b.encode()
            + self._nonce.to_bytes(8, "big")
            + CHAIN_ID.to_bytes(4, "big")
        )
        return "pool_" + hashlib.sha256(data).hexdigest()[:16]

    def _pair_key(self, token_a: str, token_b: str) -> str:
        """Canonical pair key (sorted so A:B == B:A)."""
        a, b = sorted([token_a, token_b])
        return f"{a}:{b}"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _init_dex_db(self, db_path: str) -> None:
        """Open/create the DEX SQLite database and create tables."""
        import os
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_conn.execute("PRAGMA synchronous=NORMAL")
        self._db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS dex_pools (
                pool_id TEXT PRIMARY KEY,
                data    TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dex_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._db_conn.commit()

    def _safe_dex_commit(self) -> None:
        """Commit with error suppression (never crash on persistence failure)."""
        try:
            self._db_conn.commit()
        except Exception:  # pragma: no cover
            pass  # Non-fatal: in-memory state is authoritative

    def _load_dex_from_db(self) -> None:
        """Reload all pools from the database on startup."""
        if self._db_conn is None:
            return
        with self._db_lock:
            cursor = self._db_conn.execute("SELECT pool_id, data FROM dex_pools")
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    pool = LiquidityPool.from_dict(data)
                    self._pools[pool.pool_id] = pool
                    # Rebuild pair index
                    pair_key = self._pair_key(pool.token_a_id, pool.token_b_id)
                    self._pair_index[pair_key] = pool.pool_id
                    self._total_pools_created += 1
                    self._total_volume += pool.total_volume
                    self._total_swaps += pool.swap_count
                    if data.get("_nonce_hint", 0) > self._nonce:
                        self._nonce = data["_nonce_hint"]
                except Exception:
                    pass  # Skip corrupted rows

    def _save_pool(self, pool_id: str) -> None:
        """Persist a single pool's state to the database."""
        if self._db_conn is None:
            return
        pool = self._pools.get(pool_id)
        if pool is None:
            return
        data = pool.to_full_dict()
        data["_nonce_hint"] = self._nonce
        with self._db_lock:
            self._db_conn.execute(
                "INSERT OR REPLACE INTO dex_pools (pool_id, data) VALUES (?, ?)",
                (pool_id, json.dumps(data)),
            )
            self._safe_dex_commit()

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------

    def create_pool(
        self,
        token_a: str,
        token_b: str,
        initial_a: int,
        initial_b: int,
        creator: str,
        fee_rate: float = DEFAULT_FEE_RATE,
    ) -> Optional[str]:
        """
        Create a new liquidity pool for a token pair.

        Args:
            token_a: ID of the first token.
            token_b: ID of the second token.
            initial_a: Initial reserve amount for token A.
            initial_b: Initial reserve amount for token B.
            creator: Address (hex) of the pool creator.
            fee_rate: Swap fee rate (default 0.3%).

        Returns:
            pool_id on success, None on failure.
        """
        if token_a == token_b:
            logger.warning("Cannot create pool with identical tokens")
            return None
        if initial_a <= 0 or initial_b <= 0:
            logger.warning("Initial reserves must be positive")
            return None

        pair_key = self._pair_key(token_a, token_b)
        if pair_key in self._pair_index:
            logger.warning("Pool already exists for pair %s", pair_key)
            return None

        pool_id = self._generate_pool_id(token_a, token_b)

        # Initial LP shares = sqrt(initial_a * initial_b) (geometric mean)
        initial_shares = int((initial_a * initial_b) ** 0.5)
        if initial_shares < MIN_LIQUIDITY:
            logger.warning("Initial liquidity too small")
            return None

        pool = LiquidityPool(
            pool_id=pool_id,
            token_a_id=token_a,
            token_b_id=token_b,
            reserve_a=initial_a,
            reserve_b=initial_b,
            total_lp_shares=initial_shares,
            fee_rate=fee_rate,
            lp_holders={creator: initial_shares},
        )

        self._pools[pool_id] = pool
        self._pair_index[pair_key] = pool_id
        self._total_pools_created += 1
        self._save_pool(pool_id)
        logger.info("Pool %s created: %s/%s by %s", pool_id, token_a, token_b, creator)
        return pool_id

    # ------------------------------------------------------------------
    # Liquidity management
    # ------------------------------------------------------------------

    def add_liquidity(
        self,
        pool_id: str,
        amount_a: int,
        amount_b: int,
        provider: str,
    ) -> Optional[int]:
        """
        Add liquidity to an existing pool.

        Amounts must match the current pool ratio (or the contract adjusts).
        Returns LP shares minted, or None on failure.
        """
        pool = self._pools.get(pool_id)
        if pool is None:
            logger.warning("Pool %s not found", pool_id)
            return None
        if amount_a <= 0 or amount_b <= 0:
            logger.warning("Amounts must be positive")
            return None

        # Calculate shares proportional to the smaller contribution ratio
        share_a = (amount_a * pool.total_lp_shares) // pool.reserve_a
        share_b = (amount_b * pool.total_lp_shares) // pool.reserve_b
        new_shares = min(share_a, share_b)

        if new_shares <= 0:
            logger.warning("Liquidity addition too small")
            return None

        # Collect tokens from provider
        if self._transfer_fn:
            pool_addr = f"pool_{pool_id}"
            if not self._transfer_fn(provider, pool_addr, pool.token_a_id, amount_a):
                return None
            if not self._transfer_fn(provider, pool_addr, pool.token_b_id, amount_b):
                return None

        pool.reserve_a += amount_a
        pool.reserve_b += amount_b
        pool.total_lp_shares += new_shares
        pool.lp_holders[provider] = pool.lp_holders.get(provider, 0) + new_shares

        self._save_pool(pool_id)
        logger.info("Added liquidity to %s: +%d/%d → %d shares for %s",
                     pool_id, amount_a, amount_b, new_shares, provider)
        return new_shares

    def remove_liquidity(
        self,
        pool_id: str,
        lp_shares: int,
        provider: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Remove liquidity from a pool by burning LP shares.

        Returns (amount_a, amount_b) withdrawn, or None on failure.
        """
        pool = self._pools.get(pool_id)
        if pool is None:
            logger.warning("Pool %s not found", pool_id)
            return None
        if lp_shares <= 0:
            logger.warning("LP shares must be positive")
            return None

        holder_shares = pool.lp_holders.get(provider, 0)
        if holder_shares < lp_shares:
            logger.warning("Insufficient LP shares: has %d, wants %d", holder_shares, lp_shares)
            return None

        # Pro-rata withdrawal
        amount_a = (lp_shares * pool.reserve_a) // pool.total_lp_shares
        amount_b = (lp_shares * pool.reserve_b) // pool.total_lp_shares

        if amount_a <= 0 or amount_b <= 0:
            logger.warning("Withdrawal too small")
            return None

        pool.reserve_a -= amount_a
        pool.reserve_b -= amount_b
        pool.total_lp_shares -= lp_shares
        pool.lp_holders[provider] -= lp_shares

        if pool.lp_holders[provider] == 0:
            del pool.lp_holders[provider]

        # Return tokens to provider
        if self._transfer_fn:
            pool_addr = f"pool_{pool_id}"
            self._transfer_fn(pool_addr, provider, pool.token_a_id, amount_a)
            self._transfer_fn(pool_addr, provider, pool.token_b_id, amount_b)

        self._save_pool(pool_id)
        logger.info("Removed liquidity from %s: %d shares → %d/%d for %s",
                     pool_id, lp_shares, amount_a, amount_b, provider)
        return (amount_a, amount_b)

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def get_quote(
        self,
        pool_id: str,
        token_in: str,
        amount_in: int,
    ) -> Optional[int]:
        """
        Get expected output amount for a swap (read-only, no state change).

        Uses constant product formula: dy = (y * dx_after_fee) / (x + dx_after_fee)
        """
        pool = self._pools.get(pool_id)
        if pool is None:
            return None
        if amount_in <= 0:
            return None

        # Determine direction
        if token_in == pool.token_a_id:
            reserve_in, reserve_out = pool.reserve_a, pool.reserve_b
        elif token_in == pool.token_b_id:
            reserve_in, reserve_out = pool.reserve_b, pool.reserve_a
        else:
            return None

        # Apply fee
        amount_in_after_fee = int(amount_in * (1 - pool.fee_rate))
        if amount_in_after_fee <= 0:
            return None

        # Constant product: dy = (y * dx) / (x + dx)
        amount_out = (reserve_out * amount_in_after_fee) // (reserve_in + amount_in_after_fee)
        return amount_out

    def swap(
        self,
        pool_id: str,
        token_in: str,
        amount_in: int,
        min_amount_out: int,
        trader: str,
    ) -> Optional[int]:
        """
        Execute a token swap on the AMM.

        Args:
            pool_id: Target pool.
            token_in: Token being sold.
            amount_in: Amount of token_in being sold.
            min_amount_out: Minimum acceptable output (slippage protection).
            trader: Address (hex) of the trader.

        Returns:
            amount_out on success, None on failure.
        """
        pool = self._pools.get(pool_id)
        if pool is None:
            logger.warning("Pool %s not found", pool_id)
            return None
        if amount_in <= 0:
            logger.warning("Swap amount must be positive")
            return None

        # Verify trader has sufficient balance for the input token
        trader_balance = self._get_trader_balance(trader, token_in)
        if trader_balance is not None and amount_in > trader_balance:
            logger.warning("Insufficient balance: trader %s has %d, wants to swap %d of %s",
                         trader[:16], trader_balance, amount_in, token_in)
            return None

        # Determine direction
        if token_in == pool.token_a_id:
            reserve_in, reserve_out = pool.reserve_a, pool.reserve_b
            is_a_to_b = True
        elif token_in == pool.token_b_id:
            reserve_in, reserve_out = pool.reserve_b, pool.reserve_a
            is_a_to_b = False
        else:
            logger.warning("Token %s not in pool %s", token_in, pool_id)
            return None

        # Apply fee
        fee_amount = int(amount_in * pool.fee_rate)
        amount_in_after_fee = amount_in - fee_amount
        if amount_in_after_fee <= 0:
            logger.warning("Swap amount too small after fee")
            return None

        # Constant product: dy = (y * dx) / (x + dx)
        amount_out = (reserve_out * amount_in_after_fee) // (reserve_in + amount_in_after_fee)

        if amount_out <= 0:
            logger.warning("Output amount is zero")
            return None

        if amount_out < min_amount_out:
            logger.warning("Slippage exceeded: got %d, wanted >= %d", amount_out, min_amount_out)
            return None

        if amount_out >= reserve_out:
            logger.warning("Insufficient liquidity: want %d, pool has %d", amount_out, reserve_out)
            return None

        # Move tokens via PRC-20 registry if wired up
        if self._transfer_fn:
            pool_addr = f"pool_{pool_id}"
            token_out_id = pool.token_b_id if is_a_to_b else pool.token_a_id
            # Debit input token from trader to pool
            if not self._transfer_fn(trader, pool_addr, token_in, amount_in):
                return None
            # Credit output token from pool to trader
            if not self._transfer_fn(pool_addr, trader, token_out_id, amount_out):
                return None

        # Update reserves
        if is_a_to_b:
            pool.reserve_a += amount_in
            pool.reserve_b -= amount_out
        else:
            pool.reserve_b += amount_in
            pool.reserve_a -= amount_out

        # Update stats
        pool.total_volume += amount_in
        pool.total_fees_collected += fee_amount
        pool.swap_count += 1
        self._total_volume += amount_in
        self._total_swaps += 1

        self._save_pool(pool_id)
        logger.info("Swap on %s: %s %d → %d by %s",
                     pool_id, token_in, amount_in, amount_out, trader)
        return amount_out

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_pool(self, pool_id: str) -> Optional[dict]:
        """Get pool information by ID."""
        pool = self._pools.get(pool_id)
        return pool.to_dict() if pool else None

    def list_pools(self) -> List[dict]:
        """List all liquidity pools."""
        return [p.to_dict() for p in self._pools.values()]

    def set_balance_checker(self, checker):
        """Set a callback to check trader token balances.
        checker(trader_hex, token_id) -> int or None"""
        self._balance_checker = checker

    def set_transfer_executor(self, fn):
        """Set callback for moving PRC-20 tokens: fn(from_addr, to_addr, token_id, amount) -> bool"""
        self._transfer_fn = fn

    def _get_trader_balance(self, trader: str, token_id: str) -> Optional[int]:
        """Get trader's balance for a token. Returns None if no checker set."""
        if hasattr(self, '_balance_checker') and self._balance_checker:
            try:
                return self._balance_checker(trader, token_id)
            except Exception:
                return None
        return None  # No checker = skip balance verification

    def get_stats(self) -> dict:
        """Get DEX-wide statistics."""
        total_liquidity = sum(
            p.reserve_a + p.reserve_b for p in self._pools.values()
        )
        return {
            "total_pools": self._total_pools_created,
            "active_pools": len(self._pools),
            "total_volume": self._total_volume,
            "total_swaps": self._total_swaps,
            "total_liquidity": total_liquidity,
        }
