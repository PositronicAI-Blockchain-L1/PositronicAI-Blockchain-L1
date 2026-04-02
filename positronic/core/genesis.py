"""
Positronic - Genesis Block Creation
The very first block of the Positronic blockchain.
Bitcoin-style: NO founder pre-mine. Only locked allocations for
AI Treasury (DAO), Community, Team (vesting), and Security.
750M ASF (75%) must be earned through mining and gameplay.

Deterministic Genesis:
    Every node must produce the exact same genesis block. This is achieved by:
    1. Fixed timestamp (GENESIS_TIMESTAMP = 1750000000.0)
    2. Fixed founder seed (GENESIS_FOUNDER_SEED = 32-byte constant)
    3. Canonical state root computed from deterministic allocations
    4. Hardcoded genesis hash (GENESIS_HASH) for validation
"""

import time
from typing import Optional, Tuple

from positronic.core.block import Block, BlockHeader
from positronic.core.account import Account
from positronic.crypto.keys import KeyPair
from positronic.crypto.address import (
    TREASURY_ADDRESS,
    BURN_ADDRESS,
    address_from_pubkey,
)
from positronic.crypto.hashing import sha512
from positronic.constants import (
    CHAIN_ID,
    BLOCK_GAS_LIMIT,
    AI_TREASURY_ALLOCATION,
    COMMUNITY_ALLOCATION,
    TEAM_ALLOCATION,
    SECURITY_ALLOCATION,
    GENESIS_SUPPLY,
    TOTAL_SUPPLY,
    HASH_SIZE,
    GENESIS_TIMESTAMP,
    TESTNET_GENESIS_TIMESTAMP,
    GENESIS_FOUNDER_SEED,
    GENESIS_HASH,
    TESTNET_GENESIS_HASH,
)


# Special addresses for initial allocations
COMMUNITY_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000002")
STAKING_POOL_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000003")
TEAM_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000004")
SECURITY_ADDRESS = bytes.fromhex("0000000000000000000000000000000000000005")


def get_genesis_founder_keypair() -> KeyPair:
    """
    Return the deterministic genesis founder keypair.

    Uses GENESIS_FOUNDER_SEED (32 bytes) with KeyPair.from_seed() to produce
    the same Ed25519 keypair on every node. The founder gets 0 pre-mined
    coins — this keypair only signs the genesis block.
    """
    return KeyPair.from_seed(GENESIS_FOUNDER_SEED)


def create_genesis_block(
    founder_keypair: KeyPair,
    timestamp: Optional[float] = None,
) -> Block:
    """
    Create the genesis block of the Positronic blockchain.

    Bitcoin-style distribution — NO founder pre-mine:
    - 10% (100M) -> AI Treasury (DAO-governed, locked)
    - 10% (100M) -> Community (faucet/airdrops)
    -  3% ( 30M) -> Team (36-month cliff, 96-month vesting)
    -  2% ( 20M) -> Security (6-month cliff, 96-month vesting)
    - 55% (550M) -> Mining rewards (earned per block)
    - 20% (200M) -> Play-to-Mine (earned through gameplay)

    Args:
        founder_keypair: KeyPair to sign the genesis block.
        timestamp: Block timestamp. Defaults to GENESIS_TIMESTAMP for
                   deterministic genesis across all nodes.
    """
    if timestamp is None:
        timestamp = GENESIS_TIMESTAMP

    header = BlockHeader(
        version=1,
        height=0,
        timestamp=timestamp,
        previous_hash=b"\x00" * HASH_SIZE,
        state_root=b"\x00" * HASH_SIZE,
        transactions_root=b"\x00" * HASH_SIZE,
        receipts_root=b"\x00" * HASH_SIZE,
        ai_score_root=b"\x00" * HASH_SIZE,
        ai_model_version=1,
        validator_pubkey=founder_keypair.public_key_bytes,
        slot=0,
        epoch=0,
        gas_limit=BLOCK_GAS_LIMIT,
        gas_used=0,
        extra_data=b"Positronic Genesis - The First AI-Validated Blockchain",
        chain_id=CHAIN_ID,
    )

    block = Block(header=header, transactions=[])
    header.sign(founder_keypair)

    return block


def get_genesis_allocations(founder_address: bytes, *, network_type: str = "main") -> dict:
    """
    Returns the initial account states for the genesis block.
    Only 250M ASF (25%) allocated at genesis. 750M earned through mining.
    Founder gets 0 pre-mined coins — must earn like everyone else.

    When ``network_type`` is ``"local"`` **or** env
    ``POSITRONIC_STRESS_TEST=1``, 10 deterministic test keypairs (seeded
    ``stress_test_XX``) are pre-funded with 100K ASF each.

    When ``network_type`` is ``"testnet"``, 10 deterministic testnet
    keypairs (seeded ``testnet_wallet_XX``) are pre-funded with 100K ASF
    each so that testnet users can submit real signed transactions.
    """
    from positronic.constants import BASE_UNIT

    allocations = {
        # Founder is a validator but gets NO pre-mined tokens
        founder_address: Account(
            address=founder_address,
            balance=0,
            is_validator=True,
        ),
        # AI Treasury: 100M locked, only DAO can unlock (5% per year max)
        TREASURY_ADDRESS: Account(
            address=TREASURY_ADDRESS,
            balance=AI_TREASURY_ALLOCATION,
        ),
        # Community: 100M for faucet and airdrops
        COMMUNITY_ADDRESS: Account(
            address=COMMUNITY_ADDRESS,
            balance=COMMUNITY_ALLOCATION,
        ),
        # Team: 30M with 36-month cliff, 96-month linear vesting
        TEAM_ADDRESS: Account(
            address=TEAM_ADDRESS,
            balance=TEAM_ALLOCATION,
        ),
        # Security: 20M with 6-month cliff, 96-month linear vesting
        SECURITY_ADDRESS: Account(
            address=SECURITY_ADDRESS,
            balance=SECURITY_ALLOCATION,
        ),
    }

    # ── Pre-funded test addresses (local / stress-test) ──────────────
    import os
    if network_type == "local" or os.environ.get("POSITRONIC_STRESS_TEST") == "1":
        test_balance = 100_000 * BASE_UNIT  # 100K ASF each
        for i in range(10):
            seed = f"stress_test_{i:02d}".encode().ljust(32, b"\x00")
            kp = KeyPair.from_seed(seed)
            addr = kp.address
            if addr not in allocations:
                allocations[addr] = Account(address=addr, balance=test_balance)

    # ── Pre-funded testnet wallets (separate seeds from local) ───────
    if network_type == "testnet":
        test_balance = 100_000 * BASE_UNIT  # 100K ASF each
        for i in range(10):
            seed = f"testnet_wallet_{i:02d}".encode().ljust(32, b"\x00")
            kp = KeyPair.from_seed(seed)
            addr = kp.address
            if addr not in allocations:
                allocations[addr] = Account(address=addr, balance=test_balance)

    return allocations


def get_canonical_genesis(*, network_type: str = "main") -> Tuple[Block, dict, bytes]:
    """
    Create the canonical (deterministic) genesis block with full state.

    Args:
        network_type: ``"main"`` (default) or ``"testnet"``.  Testnet uses
            a different timestamp (TESTNET_GENESIS_TIMESTAMP) and includes
            10 pre-funded testnet wallets, producing a distinct genesis hash.

    Returns:
        Tuple of (genesis_block, allocations_dict, founder_address).

    The genesis block produced by this function is identical on every node
    for a given network_type:
    - Same founder keypair (from GENESIS_FOUNDER_SEED)
    - Same timestamp (GENESIS_TIMESTAMP or TESTNET_GENESIS_TIMESTAMP)
    - Same allocations (250M ASF base + testnet wallets if applicable)
    - Same state root (computed from allocations)
    - Same signature (deterministic Ed25519)
    - Same block hash

    This is the single source of truth for the Positronic genesis.
    """
    from positronic.core.state import StateManager

    kp = get_genesis_founder_keypair()
    founder_address = kp.address

    # Select timestamp based on network type
    if network_type == "testnet":
        ts = TESTNET_GENESIS_TIMESTAMP
    else:
        ts = GENESIS_TIMESTAMP

    # Create genesis block (without state root yet)
    block = create_genesis_block(kp, ts)

    # Build initial state
    allocations = get_genesis_allocations(founder_address, network_type=network_type)

    # Compute state root from allocations
    state = StateManager()
    for addr, account in allocations.items():
        state.set_account(addr, account)
    state_root = state.compute_state_root()

    # Set state root and re-sign
    block.header.state_root = state_root
    block.header.sign(kp)

    return block, allocations, founder_address


def verify_genesis_allocations(allocations: dict, *, network_type: str = "main") -> bool:
    """Verify that genesis allocations sum to GENESIS_SUPPLY (250M), not total supply.

    When network_type is ``"local"``, ``"testnet"``, or
    ``POSITRONIC_STRESS_TEST=1``, extra test allocations (up to 1M ASF)
    are permitted on top of GENESIS_SUPPLY.
    """
    import os
    total = sum(acc.balance for acc in allocations.values())
    if network_type in ("local", "testnet") or os.environ.get("POSITRONIC_STRESS_TEST") == "1":
        from positronic.constants import BASE_UNIT
        max_test_extra = 10 * 100_000 * BASE_UNIT  # 10 addrs * 100K ASF
        return GENESIS_SUPPLY <= total <= GENESIS_SUPPLY + max_test_extra
    return total == GENESIS_SUPPLY


def verify_genesis_hash(block: Block, *, network_type: str = "main") -> bool:
    """
    Verify that a genesis block matches the hardcoded genesis hash.

    For mainnet, checks against GENESIS_HASH.
    For testnet, checks against TESTNET_GENESIS_HASH.

    Returns True if the relevant hash constant is ``None``
    (pre-hardcode phase) or if the block hash matches.
    """
    if network_type == "testnet":
        if TESTNET_GENESIS_HASH is None:
            return True  # Not hardcoded yet — skip validation
        return block.hash == TESTNET_GENESIS_HASH
    if GENESIS_HASH is None:
        return True  # Not hardcoded yet — skip validation
    return block.hash == GENESIS_HASH
