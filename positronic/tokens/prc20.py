"""
Positronic - PRC-20 Fungible Token Standard
Equivalent to Ethereum's ERC-20, optimized for the Positronic network.
Features: transfer, approve, transferFrom, mint, burn.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time


@dataclass
class PRC20Token:
    """PRC-20 fungible token on the Positronic blockchain."""
    name: str
    symbol: str
    decimals: int
    total_supply: int
    owner: bytes  # Creator/owner address
    created_at: float = 0.0
    token_id: str = ""  # Unique registry ID

    # State
    _balances: Dict[bytes, int] = field(default_factory=dict)
    _allowances: Dict[Tuple[bytes, bytes], int] = field(default_factory=dict)
    _total_burned: int = 0
    _transfer_count: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        # Mint initial supply to owner
        if self.total_supply > 0 and self.owner not in self._balances:
            self._balances[self.owner] = self.total_supply

    def balance_of(self, address: bytes) -> int:
        """Get token balance of an address."""
        return self._balances.get(address, 0)

    def transfer(self, sender: bytes, recipient: bytes, amount: int) -> bool:
        """Transfer tokens from sender to recipient."""
        if amount <= 0:
            return False
        if self.balance_of(sender) < amount:
            return False
        if sender == recipient:
            return False

        self._balances[sender] = self.balance_of(sender) - amount
        self._balances[recipient] = self.balance_of(recipient) + amount
        self._transfer_count += 1
        return True

    def approve(self, owner: bytes, spender: bytes, amount: int) -> bool:
        """Approve spender to spend owner's tokens."""
        if amount < 0:
            return False
        self._allowances[(owner, spender)] = amount
        return True

    def allowance(self, owner: bytes, spender: bytes) -> int:
        """Get approved spending amount."""
        return self._allowances.get((owner, spender), 0)

    def transfer_from(
        self, spender: bytes, from_addr: bytes, to_addr: bytes, amount: int
    ) -> bool:
        """Transfer tokens using allowance (spender moves from_addr's tokens)."""
        if amount <= 0:
            return False
        allowed = self.allowance(from_addr, spender)
        if allowed < amount:
            return False
        if self.balance_of(from_addr) < amount:
            return False

        self._balances[from_addr] = self.balance_of(from_addr) - amount
        self._balances[to_addr] = self.balance_of(to_addr) + amount
        self._allowances[(from_addr, spender)] = allowed - amount
        self._transfer_count += 1
        return True

    def mint(self, to: bytes, amount: int) -> bool:
        """Mint new tokens (only owner)."""
        if amount <= 0:
            return False
        self._balances[to] = self.balance_of(to) + amount
        self.total_supply += amount
        return True

    def burn(self, from_addr: bytes, amount: int) -> bool:
        """Burn tokens from address."""
        if amount <= 0:
            return False
        if self.balance_of(from_addr) < amount:
            return False
        self._balances[from_addr] = self.balance_of(from_addr) - amount
        self.total_supply -= amount
        self._total_burned += amount
        return True

    @property
    def holder_count(self) -> int:
        """Number of addresses with non-zero balance."""
        return sum(1 for b in self._balances.values() if b > 0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "owner": self.owner.hex(),
            "token_id": self.token_id,
            "created_at": self.created_at,
            "holder_count": self.holder_count,
            "transfer_count": self._transfer_count,
            "total_burned": self._total_burned,
        }

    def to_full_dict(self) -> dict:
        """Serialize full token state (including balances) for persistence."""
        return {
            "name": self.name,
            "symbol": self.symbol,
            "decimals": self.decimals,
            "total_supply": self.total_supply,
            "owner": self.owner.hex(),
            "token_id": self.token_id,
            "created_at": self.created_at,
            "transfer_count": self._transfer_count,
            "total_burned": self._total_burned,
            # Balances: {hex_address: amount}
            "balances": {
                addr.hex(): bal
                for addr, bal in self._balances.items()
            },
            # Allowances: {"owner_hex:spender_hex": amount}
            "allowances": {
                f"{owner.hex()}:{spender.hex()}": amt
                for (owner, spender), amt in self._allowances.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PRC20Token":
        """Reconstruct PRC20Token from a full_dict snapshot."""
        owner_bytes = bytes.fromhex(d["owner"])
        token = cls(
            name=d["name"],
            symbol=d["symbol"],
            decimals=d["decimals"],
            # Pass total_supply=0 to skip __post_init__ auto-mint; we restore
            # balances manually below.
            total_supply=0,
            owner=owner_bytes,
            created_at=d.get("created_at", 0.0),
            token_id=d.get("token_id", ""),
        )
        # Restore actual total_supply (overrides the 0 set above)
        token.total_supply = d["total_supply"]
        token._transfer_count = d.get("transfer_count", 0)
        token._total_burned = d.get("total_burned", 0)
        # Restore balances
        for hex_addr, bal in d.get("balances", {}).items():
            token._balances[bytes.fromhex(hex_addr)] = bal
        # Restore allowances
        for key, amt in d.get("allowances", {}).items():
            owner_hex, spender_hex = key.split(":", 1)
            token._allowances[(bytes.fromhex(owner_hex), bytes.fromhex(spender_hex))] = amt
        return token
