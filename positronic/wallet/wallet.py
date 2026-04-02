"""
Positronic - Wallet
Key management and transaction building.
"""

import os
from typing import List, Optional

from positronic.crypto.keys import KeyPair
from positronic.crypto.address import address_from_pubkey
from positronic.wallet.keystore import KeyStore
from positronic.core.transaction import Transaction, TxType
from positronic.constants import CHAIN_ID, TX_BASE_GAS


class Wallet:
    """Positronic wallet for managing keys and building transactions."""

    def __init__(self, keystore_dir: str = "./data/keystore"):
        self.keystore_dir = keystore_dir
        os.makedirs(keystore_dir, exist_ok=True)
        self.accounts: List[KeyPair] = []
        self.active_account: Optional[KeyPair] = None

    def create_account(self, password: str) -> KeyPair:
        """Create a new account and save encrypted."""
        kp = KeyPair()
        filepath = os.path.join(
            self.keystore_dir, f"{kp.address_hex}.json"
        )
        KeyStore.save_key(kp, password, filepath)
        self.accounts.append(kp)
        if self.active_account is None:
            self.active_account = kp
        return kp

    def load_account(self, address: str, password: str) -> KeyPair:
        """Load an account from keystore."""
        filepath = os.path.join(self.keystore_dir, f"{address}.json")
        kp = KeyStore.load_key(filepath, password)
        self.accounts.append(kp)
        if self.active_account is None:
            self.active_account = kp
        return kp

    def list_accounts(self) -> List[dict]:
        """List all keystore accounts."""
        return KeyStore.list_keys(self.keystore_dir)

    def build_transfer(
        self,
        recipient: bytes,
        value: int,
        nonce: int,
        gas_price: int = 1,
        gas_limit: int = TX_BASE_GAS,
        keypair: KeyPair = None,
    ) -> Transaction:
        """Build and sign a transfer transaction."""
        kp = keypair or self.active_account
        if not kp:
            raise ValueError("No active account")

        tx = Transaction(
            tx_type=TxType.TRANSFER,
            nonce=nonce,
            sender=kp.public_key_bytes,
            recipient=recipient,
            value=value,
            gas_price=gas_price,
            gas_limit=gas_limit,
            chain_id=CHAIN_ID,
        )
        tx.sign(kp)
        return tx

    def build_contract_deploy(
        self,
        bytecode: bytes,
        nonce: int,
        value: int = 0,
        gas_price: int = 1,
        gas_limit: int = 1_000_000,
        keypair: KeyPair = None,
    ) -> Transaction:
        """Build and sign a contract deployment transaction."""
        kp = keypair or self.active_account
        if not kp:
            raise ValueError("No active account")

        tx = Transaction(
            tx_type=TxType.CONTRACT_CREATE,
            nonce=nonce,
            sender=kp.public_key_bytes,
            recipient=b"\x00" * 20,
            value=value,
            gas_price=gas_price,
            gas_limit=gas_limit,
            data=bytecode,
            chain_id=CHAIN_ID,
        )
        tx.sign(kp)
        return tx

    def build_contract_call(
        self,
        contract_address: bytes,
        calldata: bytes,
        nonce: int,
        value: int = 0,
        gas_price: int = 1,
        gas_limit: int = 500_000,
        keypair: KeyPair = None,
    ) -> Transaction:
        """Build and sign a contract call transaction."""
        kp = keypair or self.active_account
        if not kp:
            raise ValueError("No active account")

        tx = Transaction(
            tx_type=TxType.CONTRACT_CALL,
            nonce=nonce,
            sender=kp.public_key_bytes,
            recipient=contract_address,
            value=value,
            gas_price=gas_price,
            gas_limit=gas_limit,
            data=calldata,
            chain_id=CHAIN_ID,
        )
        tx.sign(kp)
        return tx

    def build_stake(
        self,
        amount: int,
        nonce: int,
        keypair: KeyPair = None,
    ) -> Transaction:
        """Build a staking transaction."""
        kp = keypair or self.active_account
        if not kp:
            raise ValueError("No active account")

        tx = Transaction(
            tx_type=TxType.STAKE,
            nonce=nonce,
            sender=kp.public_key_bytes,
            recipient=kp.address,
            value=amount,
            gas_price=1,
            gas_limit=TX_BASE_GAS,
            chain_id=CHAIN_ID,
        )
        tx.sign(kp)
        return tx
