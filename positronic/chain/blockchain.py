"""
Positronic - Main Blockchain Class
The central orchestrator tying together state, consensus, AI, VM,
compliance, ranking, governance, game, and immune systems.

Integrates DPoS consensus for proposer verification, attestation
tracking, BFT finality, and slashing enforcement.
"""

import logging
import os
import time
from typing import Optional, List, Dict

from positronic.utils.logging import get_logger

from positronic.types import (
    TreasurySpendResult, TeamVestingStatus, WalletBalanceInfo,
    AdminTransferResult, TreasuryTransactionRecord,
)
from positronic.core.block import Block, BlockHeader, TransactionReceipt
from positronic.core.transaction import Transaction, TxType, TxStatus
from positronic.core.state import StateManager
from positronic.core.genesis import (
    create_genesis_block,
    get_genesis_allocations,
    get_genesis_founder_keypair,
    get_canonical_genesis,
    verify_genesis_allocations,
    verify_genesis_hash,
)
from positronic.crypto.keys import KeyPair
from positronic.crypto.address import address_from_pubkey
from positronic.chain.executor import TransactionExecutor
from positronic.chain.validation import BlockValidator, TransactionValidator
from positronic.chain.rewards import RewardCalculator
from positronic.ai.meta_model import AIValidationGate
from positronic.ai.quarantine import QuarantinePool
from positronic.ai.trainer import AITrainer
from positronic.ai.rank_system import AIRankManager
from positronic.ai.neural_immune import NeuralImmuneSystem
from positronic.compliance.wallet_registry import WalletRegistry, WalletStatus
from positronic.compliance.forensic_report import ForensicReporter
from positronic.consensus.node_ranking import NodeRankingManager, EVALUATION_INTERVAL_BLOCKS
from positronic.consensus.dpos import DPoSConsensus
from positronic.chain.token_governance import TokenGovernance
from positronic.ai.model_governance import ModelGovernance
from positronic.game.play_to_earn import PlayToEarnEngine
from positronic.game.auto_promotion import PlayerPromotionManager
from positronic.core.trust import TrustManager
from positronic.tokens.registry import TokenRegistry
from positronic.core.paymaster import PaymasterRegistry
from positronic.wallet.smart_account import SmartWalletRegistry
from positronic.game.onchain_game import OnChainGameEngine
from positronic.ai.agents import AgentRegistry
from positronic.crypto.commitments import CommitmentManager as ZKManager  # renamed; alias kept for compat
from positronic.bridge.cross_chain import CrossChainBridge, LockMintBridge
from positronic.bridge.connector import ExternalChainConnector
from positronic.depin.registry import DePINRegistry
from positronic.crypto.post_quantum import PostQuantumManager
from positronic.identity.did import DIDRegistry
from positronic.game.bridge_api import GameBridgeAPI
from positronic.game.game_token_bridge import GameTokenBridge
from positronic.storage.game_db import GameDatabase
from positronic.storage.agent_db import AgentDatabase
from positronic.storage.rwa_db import RWADatabase
from positronic.storage import DiskFullError, StorageFatalError
from positronic.agent.registry import AgentRegistry as MarketplaceAgentRegistry
from positronic.agent.marketplace import AgentMarketplace
from positronic.tokens.rwa_registry import RWARegistry
from positronic.ai.zkml_circuit import build_scoring_circuit
from positronic.ai.zkml import ZKMLProver
from positronic.ai.zkml_verifier import ZKMLVerifier
from positronic.defi.dex import DEXEngine
from positronic.gift.faucet import GiftFaucet
from positronic.storage.database import Database
from positronic.storage.chain_db import ChainDB
from positronic.storage.checkpoints import CheckpointManager
from positronic.storage.state_db import StateDB
from positronic.chain.fork_manager import ForkManager
from positronic.constants import (
    BASE_UNIT,
    BLOCK_GAS_LIMIT,
    BLOCK_TIME,
    CHAIN_ID,
    GENESIS_TIMESTAMP,
    MIN_BLOCK_GAS_LIMIT,
    MIN_STAKE,
    MAX_BLOCK_GAS_LIMIT,
    GAS_LIMIT_ADJUSTMENT_FACTOR,
    TARGET_GAS_UTILIZATION,
)

logger = get_logger(__name__)


class Blockchain:
    """
    Positronic Blockchain - The world's first AI-validated blockchain.

    Coordinates:
    - State management & transaction execution
    - Block validation & AI validation gate
    - Wallet registration & compliance
    - AI rank system & node ranking
    - Neural immune system (threat defense)
    - Token governance (AI + council approval)
    - Play-to-Earn game engine
    - Forensic reporting (court evidence)
    - Quarantine pool & reward distribution
    - Persistence
    """

    def __init__(self, db_path: str = "./data/positronic.db", config=None,
                 encryption_password: str = None):
        """
        Initialize the Positronic Blockchain.

        Args:
            db_path: Path to the SQLite database file.
            config: Optional NodeConfig instance. When provided, AI threshold
                    overrides from config.ai are passed through to the
                    AIValidationGate for display/logging purposes.
            encryption_password: Optional password for AES-256-GCM database
                    encryption. When provided, all databases are encrypted at rest.
        """
        self._config = config
        self._encryption_password = encryption_password
        import time as _time
        self._start_time = _time.time()
        self._pending_system_txs = []  # System TXs queued by RPC (stake, unstake, etc.)

        # Founder address (always available for admin transfers)
        try:
            from positronic.core.genesis import get_canonical_genesis
            _, _, self.founder_address = get_canonical_genesis(
                network_type=getattr(config, 'network_type', 'testnet') if config else 'testnet'
            )
        except Exception:
            self.founder_address = None

        # Storage
        self.db = Database(db_path, encryption_password=encryption_password)
        self.chain_db = ChainDB(self.db)
        self.state_db = StateDB(self.db)

        # Core components
        self.state = StateManager()
        self.executor = TransactionExecutor(self.state, token_registry=None)
        _bv_net = "main"
        if config and hasattr(config, "network"):
            _bv_net = getattr(config.network, "network_type", "main")
        self.block_validator = BlockValidator(network_type=_bv_net)
        self.tx_validator = TransactionValidator()  # immune_system set below
        self.reward_calculator = RewardCalculator()

        # AI components — pass config.ai for display threshold overrides
        ai_config = config.ai if config is not None else None
        self.ai_gate = AIValidationGate(ai_config=ai_config)
        self.quarantine_pool = QuarantinePool()
        self.ai_trainer = AITrainer(
            self.ai_gate.anomaly_detector,
            self.ai_gate.feature_extractor,
        )

        # === NEW INTEGRATED MODULES ===

        # Compliance & Traceability
        self.wallet_registry = WalletRegistry()
        self.forensic_reporter = ForensicReporter()

        # AI Rank System (military ranks for AI validators)
        self.ai_rank_manager = AIRankManager()

        # Node Ranking (gamified ranks)
        self.node_ranking = NodeRankingManager()

        # Neural Immune System (threat detection & response)
        self.immune_system = NeuralImmuneSystem()
        self.tx_validator._immune_system = self.immune_system

        # Token Governance (AI + council approval for tokens)
        self.token_governance = TokenGovernance()

        # AI Model Governance (model updates, weights, treasury spending)
        self.model_governance = ModelGovernance()

        # Play-to-Earn Game Engine
        self.game_engine = PlayToEarnEngine()

        # Play-to-Mine Auto-Promotion Manager
        self.promotion_manager = PlayerPromotionManager(
            game_engine=self.game_engine,
            state=self.state,
            blockchain=self,
        )

        # TRUST System (Soulbound Token reputation)
        self.trust_manager = TrustManager()

        # Token & NFT Registry (PRC-20 + PRC-721)
        self.token_registry = TokenRegistry(
            db_path=os.path.join(
                os.path.dirname(self.db.db_path) if os.path.dirname(self.db.db_path) else ".",
                "tokens.db",
            )
        )
        # Wire token registry into executor so TX types 9-12 can modify state
        self.executor._token_registry = self.token_registry

        # Gasless Transactions (Paymaster)
        self.paymaster_registry = PaymasterRegistry()

        # Smart Wallet (Account Abstraction)
        self.smart_wallet_registry = SmartWalletRegistry()

        # Fully On-Chain Game Engine
        self.onchain_game = OnChainGameEngine()

        # AI Agents On-Chain
        self.agent_registry = AgentRegistry()

        # ZK-Privacy
        self.zk_manager = ZKManager()

        # Cross-Chain Bridge
        self.bridge = CrossChainBridge()

        # Phase 20: Lock/Mint Bridge (wired to state manager)
        self.lock_mint_bridge = LockMintBridge(
            state_manager=self.state,
            blockchain=self,
            connector=ExternalChainConnector(),
            db_path=os.path.join(
                os.path.dirname(self.db.db_path) if hasattr(self.db, 'db_path') else '.',
                "bridge.db",
            ),
        )
        # Wire blockchain reference into executor for bridge TX types 14-17
        self.executor._blockchain = self

        # DePIN (Decentralized Physical Infrastructure)
        self.depin_registry = DePINRegistry()

        # Post-Quantum Security
        self.pq_manager = PostQuantumManager()

        # Decentralized Identity (DID)
        self.did_registry = DIDRegistry()

        # Game Data Persistence (SQLite)
        self.game_db = GameDatabase(db_path=os.path.join(
            os.path.dirname(self.db.db_path) if hasattr(self.db, 'db_path') else '.',
            "game_data.db",
        ), encryption_password=encryption_password)

        # Game Bridge (external game → blockchain connection)
        self.game_bridge = GameBridgeAPI(game_db=self.game_db)

        # Game-to-Token Bridge (games create custom tokens & mint NFTs)
        self.game_token_bridge = GameTokenBridge(
            game_registry=self.game_bridge.registry,
            token_registry=self.token_registry,
            game_db=self.game_db,
            blockchain=self,
        )

        # AI Agent Marketplace (Phase 29)
        self.agent_db = AgentDatabase(db_path=os.path.join(
            os.path.dirname(self.db.db_path) if hasattr(self.db, 'db_path') else '.',
            "agent_data.db",
        ), encryption_password=encryption_password)
        self.marketplace_registry = MarketplaceAgentRegistry()
        self.marketplace = AgentMarketplace(self.marketplace_registry)
        # Share council: governance council also governs marketplace agents
        self.marketplace_registry._council_members = self.token_governance._council_members

        # RWA Tokenization Engine (Phase 30)
        self.rwa_db = RWADatabase(db_path=os.path.join(
            os.path.dirname(self.db.db_path) if hasattr(self.db, 'db_path') else '.',
            "rwa_data.db",
        ), encryption_password=encryption_password)
        self.rwa_registry = RWARegistry()
        # Share council: governance council also governs RWA approvals
        self.rwa_registry._council_members = self.token_governance._council_members

        # ZKML — Zero-Knowledge Machine Learning (Phase 31)
        self.zkml_circuit = build_scoring_circuit()
        self.zkml_prover = ZKMLProver(self.zkml_circuit, prover_id="node")
        self.zkml_verifier = ZKMLVerifier()
        # Register initial model commitment at height 0
        self.zkml_verifier.register_model_commitment(
            0, self.zkml_circuit.model_commitment()
        )

        # DEX — Automated Market Maker (AMM)
        self.dex = DEXEngine(
            db_path=os.path.join(
                os.path.dirname(self.db.db_path) if os.path.dirname(self.db.db_path) else ".",
                "dex.db",
            )
        )
        # Wire balance checker so DEX verifies trader has funds
        def _dex_balance_check(trader_hex: str, token_id: str) -> int:
            addr = bytes.fromhex(trader_hex.replace("0x", ""))
            if token_id == "ASF":
                return self.state.get_balance(addr)
            tok = self.token_registry.get_token(token_id)
            if tok:
                return tok.balance_of(addr)
            return 0
        self.dex.set_balance_checker(_dex_balance_check)

        # Gift Faucet (testnet ASF distribution)
        # Use founder keypair for faucet (community pool distributions)
        try:
            _faucet_kp = get_genesis_founder_keypair()
            self._faucet = GiftFaucet(_faucet_kp)
        except Exception:
            self._faucet = None  # Faucet unavailable

        # DPoS Consensus Engine (initialized lazily during initialize())
        self.consensus: Optional[DPoSConsensus] = None
        # True when DPoS was bootstrapped with multiple initial validators.
        # When False (single founder), proposer checks are skipped because
        # only one entity can produce blocks anyway.
        self._multi_validator_mode = False

        # Checkpoint Manager (periodic state snapshots for fast sync)
        self.checkpoint_manager = CheckpointManager(
            encryption_password=encryption_password,
        )

        # Chain state
        self.chain_head: Optional[Block] = None
        self.height: int = -1
        self.fork_manager = ForkManager()
        self.genesis_time: float = 0
        self.receipts: dict = {}  # tx_hash -> receipt
        self._treasury_tx_log = []
        self._init_treasury_log_table()
        self._load_treasury_log_from_db()
        self._storage_failed = False  # Set True on disk-full; rejects further blocks

    def initialize(
        self,
        founder_keypair: Optional[KeyPair] = None,
        validators: Optional[List[dict]] = None,
    ) -> Block:
        """
        Initialize the blockchain with a deterministic genesis block.

        Creates genesis block, sets up initial state, and bootstraps the
        DPoS consensus engine. Uses the canonical genesis (deterministic
        founder keypair + fixed timestamp) so every node produces the same
        genesis block.

        Args:
            founder_keypair: KeyPair of the founder/initial validator.
                If None, uses the canonical genesis founder keypair
                derived from GENESIS_FOUNDER_SEED.
            validators: Optional list of initial validators for DPoS.
                Each dict: {pubkey, stake, is_nvn, commission_rate}.
                If None, the founder is registered as the sole genesis
                validator with stake from their genesis allocation.
        """
        # Check if chain already exists
        existing_height = self.chain_db.get_chain_height()
        if existing_height >= 0:
            return self._load_chain(founder_keypair=founder_keypair, validators=validators)

        # Use canonical founder keypair if none provided
        using_canonical = founder_keypair is None
        if using_canonical:
            founder_keypair = get_genesis_founder_keypair()

        # Determine network type for genesis variations
        _net_type = "main"
        if self._config and hasattr(self._config, "network"):
            _net_type = getattr(self._config.network, "network_type", "main")

        # Use get_canonical_genesis() for deterministic genesis creation.
        # This ensures ALL nodes produce the exact same genesis hash by
        # computing state_root from allocations in a single atomic step.
        from positronic.core.genesis import get_canonical_genesis
        genesis, allocations, founder_address = get_canonical_genesis(
            network_type=_net_type
        )
        self.founder_address = founder_address

        from positronic.constants import TESTNET_GENESIS_TIMESTAMP
        _genesis_ts = TESTNET_GENESIS_TIMESTAMP if _net_type == "testnet" else GENESIS_TIMESTAMP
        self.genesis_time = _genesis_ts
        state_root = genesis.header.state_root

        # Apply allocations to state
        for addr, account in allocations.items():
            self.state.set_account(addr, account)

        # Register all genesis addresses in the wallet registry so that
        # wallet_registry.total_wallets reflects the actual account count.
        for addr in allocations:
            reg = self.wallet_registry.ensure_registered(addr, source_type="genesis")
            reg.status = WalletStatus.REGISTERED
            reg.ai_trust_score = 1.0

        # Verify genesis hash matches hardcoded value (only for canonical founder)
        if using_canonical:
            assert verify_genesis_hash(genesis, network_type=_net_type), (
                f"Genesis hash mismatch! Computed {genesis.hash.hex()[:32]}... "
                f"does not match hardcoded genesis hash for {_net_type}"
            )

        # Persist — atomic: block + state committed together (FIX 3/4)
        self.chain_db.put_block(genesis, commit=False)
        self.state.save_to_db(self.state_db, commit=False)
        self.state_db.commit()  # single atomic commit

        # Genesis checkpoint
        self.checkpoint_manager.create_checkpoint(
            height=0,
            block_hash=genesis.hash,
            state_root=state_root,
            total_supply=self.state.get_total_supply(),
            total_accounts=len(self.state.accounts),
        )

        self.chain_head = genesis
        self.height = 0

        # Bootstrap DPoS consensus engine.
        # If no explicit validator list provided, register the founder as
        # the sole genesis validator so that block production, elections,
        # and attestations work in single-node / founder mode.
        if not validators and founder_keypair is not None:
            founder_addr = founder_keypair.address
            founder_balance = self.state.get_balance(founder_addr)
            stake = max(founder_balance, MIN_STAKE)
            validators = [{
                "pubkey": founder_keypair.public_key_bytes,
                "stake": stake,
                "is_nvn": False,
                "commission_rate": 0.05,
            }]
        if validators:
            self.consensus = DPoSConsensus(genesis_time=GENESIS_TIMESTAMP)
            self.consensus.initialize(validators=validators)
            # Wire validator registry to executor for STAKE transactions
            self.executor._validator_registry = self.consensus.registry
            self._multi_validator_mode = len(validators) > 1

        return genesis

    def _load_chain(self, founder_keypair=None, validators=None) -> Block:
        """Load existing chain from database and re-initialize consensus."""
        self.height = self.chain_db.get_chain_height()
        self.chain_head = self.chain_db.get_block_by_height(self.height)
        self.state.load_from_db(self.state_db)

        # Re-initialize DPoS consensus engine after restart.
        # Without this, consensus/attestation/validator stats are empty.
        if not validators and founder_keypair is not None:
            founder_addr = founder_keypair.address
            founder_balance = self.state.get_balance(founder_addr)
            stake = max(founder_balance, MIN_STAKE)
            validators = [{
                "pubkey": founder_keypair.public_key_bytes,
                "stake": stake,
                "is_nvn": False,
                "commission_rate": 0.05,
            }]
        if not validators:
            # Fallback: use canonical founder keypair
            from positronic.core.genesis import get_genesis_founder_keypair
            fkp = get_genesis_founder_keypair()
            founder_balance = self.state.get_balance(fkp.address)
            stake = max(founder_balance, MIN_STAKE)
            validators = [{
                "pubkey": fkp.public_key_bytes,
                "stake": stake,
                "is_nvn": False,
                "commission_rate": 0.05,
            }]
        self.consensus = DPoSConsensus(genesis_time=GENESIS_TIMESTAMP)
        self.consensus.initialize(validators=validators)
        self.executor._validator_registry = self.consensus.registry
        self._multi_validator_mode = len(validators) > 1

        # Restore persisted validators from the database
        try:
            loaded = self.consensus.registry.load_from_db(self.db)
            if loaded > 0:
                self._multi_validator_mode = True
        except Exception as e:
            logger.debug("Failed to load validators from DB: %s", e)

        return self.chain_head

    def _sync_registry_from_state(self):
        """Derive the validator registry from on-chain account state.

        Ghost-proof: ONLY registers accounts that have a REAL 32-byte
        Ed25519 pubkey (set by _execute_stake or _register_node_as_validator).
        Accounts that staked via RPC without running a node have empty
        pubkeys and are skipped — they cannot produce blocks.
        """
        if not self.consensus:
            return
        registry = self.consensus.registry
        from positronic.constants import MIN_STAKE
        from positronic.consensus.validator import Validator, ValidatorStatus

        for addr, acc in self.state.accounts.items():
            has_real_pubkey = (
                acc.validator_pubkey
                and len(acc.validator_pubkey) == 32
                and acc.validator_pubkey != b'\x00' * 32
                and acc.validator_pubkey[20:] != b'\x00' * 12
            )

            if (acc.staked_amount >= MIN_STAKE
                    and acc.is_validator
                    and has_real_pubkey):
                pubkey = acc.validator_pubkey
                if addr in registry._validators:
                    v = registry._validators[addr]
                    v.stake = acc.staked_amount
                    v.pubkey = pubkey
                    if v.status != ValidatorStatus.ACTIVE:
                        registry.activate(addr)
                else:
                    v = Validator(
                        address=addr, pubkey=pubkey,
                        stake=acc.staked_amount,
                        status=ValidatorStatus.ACTIVE,
                    )
                    registry._validators[addr] = v
                    registry._pubkey_index[pubkey] = addr
            elif addr in registry._validators:
                registry.deactivate(addr)

        # Re-run election with updated validator set
        try:
            current_slot = self.height + 1
            from positronic.constants import SLOTS_PER_EPOCH
            current_epoch = current_slot // SLOTS_PER_EPOCH
            from positronic.consensus.election import ValidatorElection
            last_hash = self.chain_head.hash if self.chain_head else b"\x00" * 64
            epoch_seed = ValidatorElection.derive_epoch_seed(last_hash, current_epoch)
            self.consensus.election._history.pop(current_epoch, None)
            self.consensus.election.run_election(
                epoch=current_epoch, epoch_seed=epoch_seed)
        except Exception:
            pass

    def _distribute_attestation_rewards(self, block):
        """Distribute block reward shares after the producer's reward TX.

        Four-way split per block (24 ASF):
          producer  4 ASF → already paid via reward TX
          attesters 12 ASF → all active validators pro-rata by stake (pending_rewards)
          nvn        4 ASF → NVN operators; fallback: active validators pro-rata
          treasury   4 ASF → DAO treasury (COMMUNITY_POOL_ADDRESS)
        """
        try:
            reward = self.reward_calculator.get_block_reward(block.height)
            # Use eligible_validators (PENDING + ACTIVE) so RPC stakers
            # without a node pubkey also receive attester/NVN rewards.
            eligible = self.consensus.registry.eligible_validators if self.consensus else []
            eligible_count = len(eligible)
            split = self.reward_calculator.get_reward_split(
                reward,
                eligible_count if eligible_count > 0 else 1,
            )
            logging.getLogger(__name__).info(
                "Attestation dist: reward=%d split=%s eligible=%d",
                reward, {k: v // 10**18 for k, v in split.items()},
                eligible_count)

            total_stake = sum(v.total_stake for v in eligible)

            def _distribute_pro_rata(amount: int, validators, stake_total: int):
                """Distribute amount to validators pro-rata by stake into pending_rewards."""
                if not validators or stake_total == 0 or amount <= 0:
                    return
                distributed = 0
                for i, v in enumerate(validators):
                    if i == len(validators) - 1:
                        share = amount - distributed
                    else:
                        share = int(amount * v.total_stake / stake_total)
                    if share > 0:
                        self.state.add_pending_rewards(v.address, share)
                    distributed += share

            # Attestation rewards (50%) → all eligible validators pro-rata
            if split["attesters"] > 0:
                _distribute_pro_rata(split["attesters"], eligible, total_stake)

            # NVN / node share (4 ASF) → eligible validators (fallback when no NVN)
            if split["nodes"] > 0:
                _distribute_pro_rata(split["nodes"], eligible, total_stake)

            # Treasury share (4 ASF) → DAO treasury (community pool address)
            if split.get("treasury", 0) > 0:
                from positronic.crypto.address import COMMUNITY_POOL_ADDRESS
                self.state.add_balance(COMMUNITY_POOL_ADDRESS, split["treasury"])

        except Exception as e:
            logging.getLogger(__name__).warning("Attestation reward dist error: %s", e)

    def _ensure_epoch_election(self, slot: int) -> None:
        """Ensure the DPoS election has been run for the epoch that *slot*
        belongs to.  In normal operation the epoch boundary is detected by
        wall-clock time after each ``add_block``.  However, in tests (and
        when blocks are created faster than real-time) the slot number may
        cross an epoch boundary before the wall-clock does.  This helper
        fills the gap by directly running the election for the target epoch
        if it is missing from the election history.
        """
        if not self.consensus:
            return
        from positronic.constants import SLOTS_PER_EPOCH
        from positronic.consensus.election import ValidatorElection
        target_epoch = slot // SLOTS_PER_EPOCH
        # Check if the election already exists in history
        if self.consensus.election._history.get(target_epoch) is not None:
            return  # election already computed
        # Use the hash of the LAST block of the previous epoch as election seed.
        # This ensures all nodes compute the same election regardless of when
        # they run _ensure_epoch_election (chain_head may differ by 1-2 blocks).
        epoch_start_height = target_epoch * SLOTS_PER_EPOCH
        seed_block_height = max(0, epoch_start_height - 1)
        seed_block = self.chain_db.get_block_by_height(seed_block_height)
        if seed_block:
            seed_hash = seed_block.hash
        elif self.chain_head:
            seed_hash = self.chain_head.hash
        else:
            seed_hash = b"\x00" * 64
        epoch_seed = ValidatorElection.derive_epoch_seed(seed_hash, target_epoch)
        self.consensus.election.run_election(
            epoch=target_epoch, epoch_seed=epoch_seed,
        )

    def add_block(self, block: Block, skip_state_root: bool = False) -> bool:
        """
        Validate and add a block to the chain.

        Steps:
        1. Validate block structure and header
        2. Verify DPoS proposer assignment (if consensus engine active)
        3. Execute all transactions
        4. Verify state root matches after execution (unless skip_state_root)
        5. Persist to database
        6. Notify consensus engine of block acceptance

        Args:
            block: The block to add.
            skip_state_root: If True, skip state root verification. Used
                during sync where blocks are trusted from peers and state
                root may not be reproducible due to trie rebuild ordering.
        """
        # Reject blocks if storage has failed (disk full, etc.)
        if self._storage_failed:
            logger.critical(
                "Node in storage-failed state. Rejecting block %d. "
                "Free disk space and restart.", block.height
            )
            return False

        # Genesis block handling:
        # - If we already have genesis (height >= 0): skip (return True)
        # - If we don't have genesis (height == -1) and block is genesis:
        #   accept it from sync and initialize chain state with allocations
        if block.height == 0 and self.height >= 0:
            logger.debug("Genesis block already exists, skipping")
            return True

        # Fork detection: if block height <= current height, it's a competing block
        if block.height <= self.height and block.height > 0:
            stored = self.fork_manager.store_fork_candidate(block)
            if stored:
                logging.getLogger(__name__).info(
                    "Competing block at height %d stored as fork candidate",
                    block.height,
                )
                # Check if this creates a longer fork worth switching to
                # Find the highest contiguous fork chain starting from this block
                fork_tip = block.height
                while self.fork_manager.has_competing_block(fork_tip + 1):
                    fork_tip += 1

                if self.fork_manager.should_reorg(self.height, fork_tip):
                    fork_base = block.height
                    fork_chain = self.fork_manager.collect_fork_chain(
                        fork_base, fork_tip
                    )
                    if fork_chain and self.execute_reorg(fork_chain):
                        return True

            return False  # Don't add to canonical chain yet

        # Validate block structure
        validation = self.block_validator.validate(
            block, self.chain_head, self.state
        )
        if not validation.valid:
            logging.getLogger(__name__).debug("Block validation failed: %s", validation.error)
            return False

        # DPoS proposer verification — accept fallback producers
        # Primary proposer is preferred, but if offline, any registered
        # validator can produce (fallback). This prevents chain halts.
        # Nodes that just joined may not know all validators yet, so
        # unknown proposers are accepted with a debug log if block
        # signature is valid (verified below).
        if (self.consensus and block.height > 0
                and block.header.validator_pubkey):
            self._ensure_epoch_election(block.header.slot)
            if not self.consensus.validate_block_proposer(
                block.header.slot, block.header.validator_pubkey
            ):
                # Check if producer is at least a registered validator
                proposer_addr = address_from_pubkey(block.header.validator_pubkey)
                if proposer_addr in self.consensus.registry._validators:
                    logging.getLogger(__name__).debug(
                        "Fallback proposer %s... accepted for slot %d",
                        proposer_addr.hex()[:16], block.header.slot,
                    )
                else:
                    # Unknown proposer — accept if block signature is valid.
                    # Local registry may not be synced yet (especially on desktop
                    # apps that sync from remote). Signature verification below
                    # ensures only holders of valid Ed25519 keys can produce.
                    pass

        # Create state snapshot for rollback
        snapshot_id = self.state.snapshot()

        try:
            validator_addr = address_from_pubkey(block.header.validator_pubkey) if block.header.validator_pubkey else b""

            # Collect NVN addresses (must match create_block for deterministic fee distribution)
            nvn_addrs = []
            if self.consensus:
                try:
                    nvn_addrs = [v.address for v in self.consensus.registry.nvn_validators]
                except Exception:
                    nvn_addrs = []

            # Execute all transactions
            receipts = []
            for idx, tx in enumerate(block.transactions):
                receipt = self.executor.execute(
                    tx,
                    block_height=block.height,
                    block_hash=block.hash,
                    validator_address=validator_addr,
                    nvn_addresses=nvn_addrs,
                    block_timestamp=block.header.timestamp,
                )
                receipt.tx_index = idx
                receipts.append(receipt)

                if not receipt.status and tx.tx_type not in (TxType.REWARD, TxType.AI_TREASURY):
                    # Transaction failed but block can still be valid
                    pass

            # Verify state root AFTER executing all transactions.
            # Skip state root verification only when explicitly requested
            # (e.g. during initial bootstrap from a trusted snapshot).
            # Trie rebuild is now deterministic (sorted keys) so all nodes
            # compute identical state roots.
            _skip_root = skip_state_root
            if block.header.state_root and block.height > 0 and not _skip_root:
                computed_root = self.state.compute_state_root()
                if computed_root != block.header.state_root:
                    self.state.revert(snapshot_id)
                    logging.getLogger(__name__).error(
                        "State root mismatch at height %d: block=%s... computed=%s...",
                        block.height,
                        block.header.state_root.hex()[:32],
                        computed_root.hex()[:32],
                    )
                    return False

            # Verify total gas used doesn't exceed block gas limit
            total_gas_used = sum(r.gas_used for r in receipts)
            if total_gas_used > block.header.gas_limit:
                self.state.revert(snapshot_id)
                logging.getLogger(__name__).error(
                    "Block gas overflow at height %d: used=%d limit=%d",
                    block.height, total_gas_used, block.header.gas_limit,
                )
                return False

            # Store receipts
            for receipt in receipts:
                self.receipts[receipt.tx_hash] = receipt

            # Sync validator registry from on-chain state after TX execution
            # so newly staked validators are picked up by all nodes.
            if self.consensus and block.height > 0:
                self._sync_registry_from_state()

            # Distribute attestation rewards BEFORE persist
            if self.consensus and block.height > 0:
                self._distribute_attestation_rewards(block)

            # Persist — atomic commit: block + state in one SQLite transaction.
            # FIX 3/4: Both put_block and save_to_db write without committing;
            # a single safe_commit() at the end ensures block, state, and contract
            # code are persisted atomically. Crash between writes cannot cause
            # block/state divergence or contract code loss.
            self.chain_db.put_block(block, commit=False)
            self.state.save_to_db(self.state_db, commit=False)
            self.state_db.commit()  # single atomic commit for block + state + contracts
            # Persist validator registry alongside state
            try:
                if hasattr(self, 'consensus') and self.consensus and hasattr(self.consensus, 'registry'):
                    self.consensus.registry.save_to_db(self.db)
            except Exception as e:
                logging.getLogger(__name__).debug("Validator registry save: %s", e)
            self.state.commit_snapshot(snapshot_id)

            # Create checkpoint at interval for fast node sync
            if self.checkpoint_manager.should_create_checkpoint(block.height):
                self.checkpoint_manager.create_checkpoint(
                    height=block.height,
                    block_hash=block.hash,
                    state_root=self.state.compute_state_root(),
                    total_supply=self.state.get_total_supply(),
                    total_accounts=len(self.state.accounts),
                )

            # Update chain head
            self.chain_head = block
            self.height = block.height

            # Notify DPoS consensus engine of new block.
            # In single-validator mode the runtime keypair may differ from
            # the genesis founder registered in the DPoS registry, so we
            # guard against KeyError from an unregistered proposer address.
            if self.consensus and block.height > 0:
                try:
                    self.consensus.on_block_proposed(
                        slot=block.header.slot,
                        proposer_address=validator_addr,
                        block_hash=block.hash,
                        timestamp=block.header.timestamp,
                    )
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        "on_block_proposed failed for %s: %s",
                        validator_addr.hex()[:16], e,
                    )

                # Always register block with finality tracker regardless of
                # whether on_block_proposed succeeded. This ensures BFT
                # attestation processing works even when the proposer is not
                # in the DPoS registry (e.g. after node restart).
                try:
                    if hasattr(self.consensus, 'finality'):
                        self.consensus.finality.register_block(
                            block.header.slot, block.hash, validator_addr)
                except Exception as e:
                    logging.getLogger(__name__).debug(
                        "Finality register_block failed: %s", e)

                # Phase 15: Feed censorship detector with block inclusion data
                tx_count = len([tx for tx in block.transactions
                                if tx.tx_type == TxType.TRANSFER])
                mempool_size = self.mempool_size if hasattr(self, 'mempool_size') else max(tx_count, 1)
                # Only report to censorship detector if mempool has transactions
                # Empty mempool = nothing to include, NOT censorship
                if mempool_size > 0:
                    self.consensus.censorship_detector.on_block_produced(
                        proposer=validator_addr,
                        included_count=tx_count,
                        available_count=mempool_size,
                    )

                # Check for epoch transitions
                needs_transition, new_epoch = self.consensus.needs_epoch_transition(
                    block.header.timestamp
                )
                if needs_transition:
                    self.consensus.on_epoch_boundary(new_epoch, block.hash)
                    # Phase 15+18: Report flagged censoring validators to immune system
                    # Severity is graduated based on actual inclusion rate
                    flagged = getattr(self.consensus, '_last_flagged_censors', [])
                    for censor_addr in flagged:
                        # Get actual inclusion rate for graduated severity
                        inclusion_rate = 0.0
                        try:
                            cd = self.consensus.censorship_detector
                            stats = cd.get_validator_record(censor_addr)
                            if stats and stats.get("total_available", 0) > 0:
                                inclusion_rate = stats["total_included"] / stats["total_available"]
                        except Exception as e:
                            logger.debug("Failed to get inclusion rate for censorship check: %s", e)
                        # Graduated severity: 0% inclusion = 1.0, 70% = 0.30
                        severity = max(0.1, min(1.0, 1.0 - inclusion_rate))
                        self.immune_system.report_anomaly(
                            address=censor_addr,
                            score=severity,
                            description=(
                                f"Censorship detected: inclusion rate "
                                f"{inclusion_rate:.1%} at epoch {new_epoch}"
                            ),
                            block_height=block.height,
                        )

            # Train AI on confirmed transactions
            confirmed = [tx for tx in block.transactions if tx.tx_type == TxType.TRANSFER]
            if confirmed:
                self.ai_trainer.add_training_data(confirmed)
                if self.ai_trainer.should_train():
                    self.ai_trainer.train_step()

            # === Post-block hooks for integrated modules ===

            # Update wallet activity for all senders/recipients
            for tx in block.transactions:
                if tx.tx_type == TxType.TRANSFER:
                    sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""
                    self.wallet_registry.update_activity(sender_addr, tx.value)
                    self.wallet_registry.update_activity(tx.recipient, tx.value)

            # Record node heartbeat & block validation for the validator
            if validator_addr:
                self.node_ranking.record_heartbeat(validator_addr, BLOCK_TIME)
                self.node_ranking.record_block_validated(validator_addr)
                self.node_ranking.record_block_proposed(validator_addr)

                # TRUST: reward block miner
                self.trust_manager.on_block_mined(validator_addr)

            # AI rank: record scoring accuracy for NVN validators
            nvn_validators = (
                self.consensus.registry.nvn_validators
                if self.consensus else []
            )
            for tx in block.transactions:
                if tx.tx_type == TxType.TRANSFER and tx.ai_score > 0:
                    was_accurate = tx.ai_score < 0.85  # Correct accept
                    for nvn in nvn_validators:
                        self.ai_rank_manager.record_score(nvn.address, was_accurate)

            # Immune system: monitor for anomalies
            high_risk_txs = [tx for tx in block.transactions
                            if tx.ai_score >= 0.7 and tx.tx_type == TxType.TRANSFER]
            for tx in high_risk_txs:
                sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""
                self.immune_system.report_anomaly(
                    address=sender_addr,
                    score=tx.ai_score,
                    description=f"High-risk TX in block {block.height}: score={tx.ai_score:.3f}",
                    block_height=block.height,
                )
                # TRUST: penalize suspicious transactions
                self.trust_manager.on_suspicious_tx(sender_addr)

            # Quarterly node evaluation
            if block.height > 0 and block.height % EVALUATION_INTERVAL_BLOCKS == 0:
                self.node_ranking.evaluate_all(block.height)

            # Distribute game rewards (every 100 blocks)
            if block.height % 100 == 0:
                # Play-to-Earn engine rewards (built-in game)
                pending = self.game_engine.get_pending_rewards()
                for addr, amount in pending.items():
                    # Do NOT add_balance here — the system TX executor
                    # (_execute_reward) will credit the recipient when the
                    # block is processed, ensuring all nodes apply it once.
                    self._create_system_tx(
                        TxType.GAME_REWARD, b'\x00' * 20, addr, amount,
                        b"game_reward",
                    )
                if pending:
                    self.game_engine.clear_pending_rewards()

                # Game Bridge rewards (external games via SDK/API)
                bridge_pending = self.game_bridge.get_pending_rewards()
                for addr, amount in bridge_pending.items():
                    # Do NOT add_balance here — executor handles it via system TX.
                    self._create_system_tx(
                        TxType.GAME_REWARD, b'\x00' * 20, addr, amount,
                        b"game_bridge_reward",
                    )
                if bridge_pending:
                    self.game_bridge.clear_pending_rewards()

                # Play-to-Mine: check promotions after reward distribution
                self.promotion_manager.batch_check_promotions()

            # Clean old threats from immune system (every 1000 blocks)
            if block.height % 1000 == 0:
                self.immune_system.clear_old_threats()

            # Clean expired smart wallet session keys (every 100 blocks)
            if block.height % 100 == 0:
                self.smart_wallet_registry.clean_all_expired_sessions()

            # Check pending AI model activations
            if hasattr(self.model_governance, 'check_pending_activations'):
                self.model_governance.check_pending_activations(
                    block.height, self.ai_gate
                )

            # Review quarantine pool
            if block.height % 100 == 0:
                self._review_quarantine(block.height)

            return True

        except (DiskFullError, StorageFatalError) as e:
            self.state.revert(snapshot_id)
            try:
                self.db.safe_rollback()
            except Exception:
                pass
            self._storage_failed = True
            logger.critical(
                "STORAGE FAILURE at block %d: %s. "
                "Node halting block acceptance. Free disk space and restart.",
                block.height, e
            )
            return False
        except Exception as e:
            self.state.revert(snapshot_id)
            try:
                self.db.safe_rollback()
            except Exception:
                pass
            logging.getLogger(__name__).error("Block execution failed: %s", e)
            return False

    def execute_reorg(self, fork_blocks: List[Block]) -> bool:
        """
        Execute a chain reorganization to switch to a longer fork.

        Steps:
        1. Validate fork_blocks is non-empty
        2. Verify the fork base doesn't violate finality
        3. Verify the fork is actually longer (should_reorg)
        4. Create state snapshot for safety rollback
        5. Collect orphaned transactions from canonical blocks being reverted
        6. Re-execute each fork block's transactions on the snapshot state
        7. On success: commit snapshot, update height/chain_head, clear fork candidates
        8. On failure: revert snapshot
        9. Return orphaned txs to mempool (if available)

        Returns True if reorg was successful, False otherwise.
        """
        logger = logging.getLogger(__name__)

        if not fork_blocks:
            return False

        fork_base = fork_blocks[0].height
        fork_tip = fork_blocks[-1].height

        # Safety: never reorg past finalized height
        if fork_base <= self.fork_manager.finalized_height:
            logger.warning(
                "Reorg rejected: fork_base %d <= finalized_height %d",
                fork_base, self.fork_manager.finalized_height,
            )
            return False

        # Only reorg if the fork is strictly longer
        if not self.fork_manager.should_reorg(self.height, fork_tip):
            return False

        # Create state snapshot for rollback
        snapshot_id = self.state.snapshot()

        try:
            # Collect orphaned transactions from canonical blocks being reverted
            orphaned_txs = []
            for h in range(fork_base, self.height + 1):
                canonical_block = self.get_block(h)
                if canonical_block:
                    orphaned_txs.extend(canonical_block.transactions)

            # Re-execute each fork block's transactions
            for block in fork_blocks:
                validator_addr = (
                    address_from_pubkey(block.header.validator_pubkey)
                    if block.header.validator_pubkey else b""
                )
                for tx in block.transactions:
                    receipt = self.executor.execute(
                        tx,
                        block_height=block.height,
                        block_hash=block.hash,
                        validator_address=validator_addr,
                    )
                    if not receipt.status and tx.tx_type not in (
                        TxType.REWARD, TxType.AI_TREASURY
                    ):
                        # Non-system transaction failed during reorg replay
                        logger.warning(
                            "Reorg failed: tx %s failed at height %d",
                            tx.tx_hash.hex()[:16] if tx.tx_hash else "?",
                            block.height,
                        )
                        self.state.revert(snapshot_id)
                        return False

            # All fork blocks replayed successfully -- commit
            self.state.commit_snapshot(snapshot_id)
            self.chain_head = fork_blocks[-1]
            self.height = fork_tip

            # Persist fork blocks — atomic commit (FIX 3/4)
            for block in fork_blocks:
                self.chain_db.put_block(block, commit=False)
            self.state.save_to_db(self.state_db, commit=False)
            self.state_db.commit()  # single atomic commit for all fork blocks + state

            # Clear fork candidates for the reorg range
            self.fork_manager.clear_range(fork_base, fork_tip)

            # Return orphaned txs that are NOT in the fork chain to mempool
            fork_tx_hashes = set()
            for block in fork_blocks:
                for tx in block.transactions:
                    if tx.tx_hash:
                        fork_tx_hashes.add(tx.tx_hash)

            returnables = [
                tx for tx in orphaned_txs
                if tx.tx_hash and tx.tx_hash not in fork_tx_hashes
                and tx.tx_type not in (TxType.REWARD, TxType.AI_TREASURY)
            ]

            # If mempool is available, return orphaned transactions
            if hasattr(self, 'mempool') and self.mempool is not None:
                for tx in returnables:
                    try:
                        self.mempool.add(tx)
                    except Exception as e:
                        logger.debug("Failed to return orphaned tx to mempool: %s", e)

            logger.info(
                "Reorg successful: switched to fork chain heights %d-%d "
                "(orphaned %d txs, returned %d to mempool)",
                fork_base, fork_tip, len(orphaned_txs), len(returnables),
            )
            return True

        except Exception as e:
            self.state.revert(snapshot_id)
            logger.error("Reorg execution failed: %s", e)
            return False

    def create_block(
        self,
        transactions: List[Transaction],
        proposer_keypair: KeyPair,
        slot: int = 0,
        epoch: int = 0,
    ) -> Block:
        """
        Create a new block as the slot proposer.

        Steps:
        1. Verify this proposer is assigned to this slot (if DPoS active)
        2. Run transactions through AI validation gate
        3. Filter accepted transactions
        4. Add reward transaction
        5. Execute transactions to produce correct state root
        6. Finalize block with post-execution state root
        """
        proposer_addr = proposer_keypair.address

        # DPoS: ensure election exists for this slot's epoch, then
        # verify proposer assignment — ALL validators treated equally.
        if self.consensus and slot > 0:
            self._ensure_epoch_election(slot)
            expected_proposer = self.consensus.get_proposer(slot)
            if expected_proposer and expected_proposer != proposer_addr:
                # In testnet: accept blocks from known validators even if local
                # election disagrees (registry may not be synced across all nodes).
                # Log as debug instead of raising ValueError.
                logging.getLogger(__name__).debug(
                    "Proposer mismatch at slot %d: got %s, expected %s "
                    "(accepting — registry may differ across nodes)",
                    slot, proposer_addr.hex()[:16], expected_proposer.hex()[:16],
                )

        # Pre-filter: wallet registry + immune system checks
        pre_filtered = []
        for tx in transactions:
            sender_addr = address_from_pubkey(tx.sender) if tx.sender else b"\x00" * 20

            # Check if sender is blocked by immune system
            if self.immune_system.is_address_blocked(sender_addr):
                continue

            # Check wallet registration (allow system TXs and unknown wallets in early phase)
            wallet_allowed, reason = self.wallet_registry.check_transaction_allowed(
                sender_addr, tx.recipient, tx.value
            )
            if not wallet_allowed and reason != "Sender wallet not registered":
                # Block blacklisted/suspended, but allow unregistered (early phase)
                continue

            pre_filtered.append(tx)

        # Mempool transactions were already AI-validated at mempool.add() time.
        # Skip redundant AI gate here to prevent double-quarantine.
        # Only score for statistics (non-blocking).
        accepted_txs = list(pre_filtered)
        for tx in pre_filtered:
            try:
                sender_acc = self.state.get_account(
                    address_from_pubkey(tx.sender) if tx.sender else b"\x00" * 20
                )
                self.ai_gate.validate_transaction(
                    tx, sender_acc, pre_filtered, len(pre_filtered)
                )
            except Exception:
                pass  # Score for stats only — never block

        # Add block reward
        active_count = self.consensus.registry.active_count if self.consensus else 1
        reward_tx = self.reward_calculator.create_reward_transaction(
            self.height + 1, proposer_addr,
            active_validator_count=active_count,
        )

        # Include pending system transactions (admin transfers, etc.)
        if self._pending_system_txs:
            n_sys = len(self._pending_system_txs)
            logger.info("Including %d system TXs in block (accepted_txs before=%d)", n_sys, len(accepted_txs))
            for sys_tx in self._pending_system_txs:
                accepted_txs.append(sys_tx)
                logger.info("  SysTX: type=%s value=%d hash=%s", sys_tx.tx_type.name, sys_tx.value, sys_tx.tx_hash_hex[:20] if sys_tx.tx_hash_hex else "?")
            self._pending_system_txs = []
            logger.info("accepted_txs after system TXs=%d", len(accepted_txs))
        else:
            logger.debug("No pending system TXs (count=0)")

        # Dynamic gas limit: adjust based on parent block utilization
        # If utilization > 50%, increase; if < 50%, decrease (like Ethereum)
        parent_gas_limit = BLOCK_GAS_LIMIT
        parent_gas_used = 0
        if self.chain_head and hasattr(self.chain_head.header, 'gas_limit'):
            parent_gas_limit = self.chain_head.header.gas_limit or BLOCK_GAS_LIMIT
            parent_gas_used = getattr(self.chain_head.header, 'gas_used', 0)

        delta = parent_gas_limit // GAS_LIMIT_ADJUSTMENT_FACTOR
        if parent_gas_limit > 0 and parent_gas_used > 0:
            utilization = parent_gas_used / parent_gas_limit
            if utilization > TARGET_GAS_UTILIZATION:
                next_gas_limit = parent_gas_limit + delta
            else:
                next_gas_limit = parent_gas_limit - delta
        else:
            next_gas_limit = parent_gas_limit

        # Clamp to [MIN, MAX]
        next_gas_limit = max(MIN_BLOCK_GAS_LIMIT, min(next_gas_limit, MAX_BLOCK_GAS_LIMIT))

        # Enforce gas limit and prevent duplicate transactions
        total_gas = 0
        seen_hashes = set()
        final_txs = []
        if reward_tx is not None:
            final_txs.append(reward_tx)
            if reward_tx.tx_hash:
                seen_hashes.add(reward_tx.tx_hash)
        for tx in accepted_txs:
            # Skip duplicate transactions
            if tx.tx_hash and tx.tx_hash in seen_hashes:
                continue
            # System TXs bypass validation: identified by empty signature + zero gas
            is_system = (getattr(tx, 'signature', b'x') == b"" and getattr(tx, 'gas_price', 1) == 0 and getattr(tx, 'gas_limit', 1) == 0)
            if not is_system:
                # Re-validate: nonce, balance, signature may have changed since mempool
                vr = self.tx_validator.validate(tx, self.state, self.height)
                if not vr.valid:
                    continue
                # Enforce vesting limits on team/security wallets
                if not self._check_vesting_limit(tx):
                    continue
            if total_gas + tx.gas_limit <= next_gas_limit or is_system:
                final_txs.append(tx)
                total_gas += tx.gas_limit
                if tx.tx_hash:
                    seen_hashes.add(tx.tx_hash)

        n_sys = sum(1 for t in final_txs if t.tx_type in (TxType.STAKE, TxType.UNSTAKE, TxType.CLAIM_REWARDS))
        logger.info("BLOCK BUILD: final_txs=%d accepted_txs=%d (reward=%s, system=%d, types=%s)",
                    len(final_txs), len(accepted_txs),
                    "yes" if reward_tx else "no", n_sys,
                    [t.tx_type.name for t in final_txs])

        # Create block header
        header = BlockHeader(
            height=self.height + 1,
            timestamp=time.time(),
            previous_hash=self.chain_head.hash if self.chain_head else b"\x00" * 64,
            ai_model_version=self.ai_gate.model_version,
            slot=slot,
            epoch=epoch,
            gas_limit=next_gas_limit,
            chain_id=CHAIN_ID,
        )

        block = Block(header=header, transactions=final_txs)

        # Collect NVN addresses for fee distribution
        nvn_addrs = []
        if self.consensus:
            try:
                nvn_addrs = [v.address for v in self.consensus.registry.nvn_validators]
            except Exception as e:
                logging.getLogger(__name__).debug("NVN address collection: %s", e)
                nvn_addrs = []

        # Execute transactions in a snapshot to compute the correct
        # post-execution state root, then roll back (add_block will
        # re-execute during validation).
        snapshot_id = self.state.snapshot()
        try:
            for tx in final_txs:
                self.executor.execute(
                    tx,
                    block_height=block.height,
                    block_hash=b"\x00" * 64,  # Placeholder, hash not yet known
                    validator_address=proposer_addr,
                    nvn_addresses=nvn_addrs,
                    block_timestamp=block.header.timestamp,
                )
            state_root = self.state.compute_state_root()
        finally:
            self.state.revert(snapshot_id)

        # Finalize block with the correct post-execution state root
        block.finalize(proposer_keypair, state_root)

        return block

    def validate_transaction(self, tx: Transaction) -> bool:
        """Validate a single transaction (for mempool acceptance)."""
        result = self.tx_validator.validate(tx, self.state, self.height)
        if not result.valid:
            return False
        # Enforce vesting limits on team/security wallet transfers
        if not self._check_vesting_limit(tx):
            return False
        return True

    def _check_vesting_limit(self, tx: Transaction) -> bool:
        """Return False if tx sender is a vested wallet and amount exceeds available."""
        from positronic.crypto.address import TEAM_ADDRESS, SECURITY_ADDRESS
        from positronic.chain.validation import TransactionValidator
        # System TXs (admin_transfer) are checked separately in admin_transfer()
        if TransactionValidator.is_system_tx(tx):
            return True
        sender_addr = address_from_pubkey(tx.sender) if tx.sender else b""
        if sender_addr == TEAM_ADDRESS:
            vesting = self.get_team_vesting_status()
            if tx.value > vesting["available"]:
                return False
        elif sender_addr == SECURITY_ADDRESS:
            vesting = self.get_security_vesting_status()
            if tx.value > vesting["available"]:
                return False
        return True

    def _review_quarantine(self, current_block: int):
        """Review quarantined transactions."""
        def re_evaluate(tx):
            sender_acc = self.state.get_account(address_from_pubkey(tx.sender))
            result = self.ai_gate.validate_transaction(tx, sender_acc)
            return result.final_score

        released, expired = self.quarantine_pool.review(
            current_block, re_evaluate
        )
        # Released transactions can be added to the next block

    # === Query Methods ===

    def get_block(self, height: int) -> Optional[Block]:
        """Retrieve a block by height, or None if not found."""
        return self.chain_db.get_block_by_height(height)

    def get_block_by_hash(self, block_hash: bytes) -> Optional[Block]:
        """Retrieve a block by its hash, or None if not found."""
        return self.chain_db.get_block_by_hash(block_hash)

    def get_transaction(self, tx_hash: bytes) -> Optional[Transaction]:
        """Retrieve a transaction by its hash, or None if not found."""
        return self.chain_db.get_transaction(tx_hash)

    def get_receipt(self, tx_hash: bytes) -> Optional[TransactionReceipt]:
        """Retrieve a transaction receipt by tx hash, or None if not found."""
        return self.receipts.get(tx_hash)

    def get_balance(self, address: bytes) -> int:
        """Return the current balance for an address (0 if unset)."""
        return self.state.get_balance(address)

    def get_nonce(self, address: bytes) -> int:
        """Return the current nonce for an address (0 if unset)."""
        return self.state.get_nonce(address)

    # === Module Access Methods ===

    def register_wallet(self, address: bytes, source_type: str = "wallet_app"):
        """Register a wallet through the blockchain interface."""
        return self.wallet_registry.register_wallet(address, source_type)

    def verify_wallet(self, address: bytes, ai_trust_score: float):
        """Verify a wallet through AI analysis."""
        return self.wallet_registry.verify_wallet(address, ai_trust_score)

    def submit_game_result(self, result):
        """Submit a game result through the blockchain interface."""
        return self.game_engine.submit_game_result(result)

    def opt_in_auto_promotion(self, address: bytes, pubkey: bytes) -> bool:
        """Player opts in for auto-promotion to node/NVN."""
        return self.game_engine.opt_in_auto_promotion(address, pubkey)

    def get_promotion_status(self, address: bytes) -> dict:
        """Get a player's Play-to-Mine promotion status."""
        return self.promotion_manager.get_promotion_status(address)

    def submit_token_proposal(self, proposer: bytes, name: str, symbol: str,
                               supply: int, decimals: int = 18, description: str = "") -> Optional[dict]:
        """Submit a token creation proposal."""
        return self.token_governance.submit_proposal(
            proposer, name, symbol, supply, decimals, description
        )

    # === System Transaction Helper ===

    def _create_system_tx(self, tx_type, sender: bytes, recipient: bytes,
                          value: int = 0, data: bytes = b""):
        """Create a system transaction and queue it for the next block.

        This ensures off-chain state modifications (game rewards, faucet drips,
        bridge mints, treasury spends, etc.) are recorded on-chain so that
        other nodes can replay and verify every balance change.
        """
        import time as _time
        tx = Transaction(
            tx_type=tx_type if isinstance(tx_type, int) else TxType.TRANSFER,
            nonce=0,
            sender=sender,
            recipient=recipient,
            value=value,
            gas_price=0,
            gas_limit=0,
            data=data,
            signature=b"",
            timestamp=_time.time(),
            chain_id=CHAIN_ID,
            ai_score=0.0,
            ai_model_version=0,
        )
        if not hasattr(self, '_pending_system_txs'):
            self._pending_system_txs = []
        self._pending_system_txs.append(tx)
        return tx

    # === AI Model Governance (treasury spending) ===

    def execute_treasury_spend(self, proposal_id: int) -> Optional[TreasurySpendResult]:
        """
        Execute an approved TREASURY_SPEND governance proposal.
        Transfers funds from the treasury address to the proposal's recipient.

        Requires: proposal must be APPROVED and pass stricter validation
        (75% supermajority, 20% quorum, amount capped at 10M ASF).

        Returns dict with tx details on success, None on failure.
        """
        from positronic.crypto.address import TREASURY_ADDRESS

        # Execute (validates treasury-specific rules internally)
        proposal = self.model_governance.execute_proposal(
            proposal_id, self.height
        )
        if proposal is None:
            return None

        # Get validated transaction parameters
        tx_params = self.model_governance.get_treasury_spend_tx_params(
            proposal_id
        )
        if tx_params is None:
            return None

        # Check treasury has sufficient balance
        treasury_balance = self.state.get_balance(TREASURY_ADDRESS)
        if treasury_balance < tx_params["amount"]:
            return None

        # Create and execute the treasury transfer
        tx = Transaction(
            tx_type=TxType.AI_TREASURY,
            nonce=0,
            sender=b"\x00" * 32,  # System
            recipient=tx_params["recipient"],
            value=tx_params["amount"],
            gas_price=0,
            gas_limit=0,
        )

        # Do NOT mutate state directly — the system TX executor
        # (_execute_reward for AI_TREASURY type) will debit treasury
        # and credit recipient when the block is processed.
        self._create_system_tx(
            TxType.AI_TREASURY, TREASURY_ADDRESS, tx_params["recipient"],
            tx_params["amount"], b"treasury_spend",
        )

        return {
            "proposal_id": proposal_id,
            "recipient": tx_params["recipient"].hex(),
            "amount": tx_params["amount"],
            "title": tx_params["title"],
            "executed_at_block": self.height,
        }

    def get_team_vesting_status(self) -> TeamVestingStatus:
        """Calculate the current team token vesting status based on block height.

        12-month cliff: nothing unlocks before month 12, then linear over 96 months.
        """
        from positronic.crypto.address import TEAM_ADDRESS
        from positronic.constants import (
            TEAM_ALLOCATION, TEAM_VESTING_MONTHS, TEAM_MONTHLY_RELEASE,
            TEAM_CLIFF_MONTHS, BLOCK_TIME,
        )
        seconds_elapsed = self.height * BLOCK_TIME
        months_elapsed = int(seconds_elapsed / (30 * 24 * 3600))
        months_elapsed = min(months_elapsed, TEAM_VESTING_MONTHS)
        # Cliff enforcement: nothing unlocks before cliff period
        if months_elapsed < TEAM_CLIFF_MONTHS:
            total_unlocked = 0
        else:
            total_unlocked = min(months_elapsed * TEAM_MONTHLY_RELEASE, TEAM_ALLOCATION)
        current_balance = self.state.get_balance(TEAM_ADDRESS)
        total_withdrawn = TEAM_ALLOCATION - current_balance
        available = max(0, total_unlocked - total_withdrawn)
        return {
            "total_allocation": TEAM_ALLOCATION,
            "vesting_months": TEAM_VESTING_MONTHS,
            "cliff_months": TEAM_CLIFF_MONTHS,
            "monthly_release": TEAM_MONTHLY_RELEASE,
            "months_elapsed": months_elapsed,
            "total_unlocked": total_unlocked,
            "total_withdrawn": total_withdrawn,
            "available": available,
            "current_balance": current_balance,
        }

    def get_security_vesting_status(self) -> TeamVestingStatus:
        """Calculate the current security allocation vesting status based on block height.

        6-month cliff: nothing unlocks before month 6, then linear over 96 months.
        """
        from positronic.crypto.address import SECURITY_ADDRESS
        from positronic.constants import (
            SECURITY_ALLOCATION, SECURITY_VESTING_MONTHS, SECURITY_MONTHLY_RELEASE,
            SECURITY_CLIFF_MONTHS, BLOCK_TIME,
        )
        seconds_elapsed = self.height * BLOCK_TIME
        months_elapsed = int(seconds_elapsed / (30 * 24 * 3600))
        months_elapsed = min(months_elapsed, SECURITY_VESTING_MONTHS)
        # Cliff enforcement: nothing unlocks before cliff period
        if months_elapsed < SECURITY_CLIFF_MONTHS:
            total_unlocked = 0
        else:
            total_unlocked = min(months_elapsed * SECURITY_MONTHLY_RELEASE, SECURITY_ALLOCATION)
        current_balance = self.state.get_balance(SECURITY_ADDRESS)
        total_withdrawn = SECURITY_ALLOCATION - current_balance
        available = max(0, total_unlocked - total_withdrawn)
        return {
            "total_allocation": SECURITY_ALLOCATION,
            "vesting_months": SECURITY_VESTING_MONTHS,
            "cliff_months": SECURITY_CLIFF_MONTHS,
            "monthly_release": SECURITY_MONTHLY_RELEASE,
            "months_elapsed": months_elapsed,
            "total_unlocked": total_unlocked,
            "total_withdrawn": total_withdrawn,
            "available": available,
            "current_balance": current_balance,
        }

    def get_full_vesting_status(self) -> Dict[str, TeamVestingStatus]:
        """Return vesting status for all 4 treasury wallets.

        Returns a dict keyed by wallet name with vesting details for:
        - ai_treasury: 150M ASF, 96-month vesting
        - community: 50M ASF, 96-month vesting
        - team: 50M ASF, 96-month vesting (12-month cliff)
        - security: 50M ASF, 96-month vesting (6-month cliff)
        """
        from positronic.crypto.address import (
            TREASURY_ADDRESS, COMMUNITY_POOL_ADDRESS,
            TEAM_ADDRESS, SECURITY_ADDRESS,
        )
        from positronic.constants import (
            AI_TREASURY_ALLOCATION, AI_TREASURY_VESTING_MONTHS, AI_TREASURY_MONTHLY_RELEASE,
            COMMUNITY_ALLOCATION, COMMUNITY_VESTING_MONTHS, COMMUNITY_MONTHLY_RELEASE,
            TEAM_ALLOCATION, TEAM_VESTING_MONTHS, TEAM_MONTHLY_RELEASE, TEAM_CLIFF_MONTHS,
            SECURITY_ALLOCATION, SECURITY_VESTING_MONTHS, SECURITY_MONTHLY_RELEASE, SECURITY_CLIFF_MONTHS,
            BLOCK_TIME,
        )

        seconds_elapsed = self.height * BLOCK_TIME
        months_elapsed_raw = int(seconds_elapsed / (30 * 24 * 3600))

        wallets = [
            ("ai_treasury", TREASURY_ADDRESS, AI_TREASURY_ALLOCATION,
             AI_TREASURY_VESTING_MONTHS, AI_TREASURY_MONTHLY_RELEASE, 0),
            ("community", COMMUNITY_POOL_ADDRESS, COMMUNITY_ALLOCATION,
             COMMUNITY_VESTING_MONTHS, COMMUNITY_MONTHLY_RELEASE, 0),
            ("team", TEAM_ADDRESS, TEAM_ALLOCATION,
             TEAM_VESTING_MONTHS, TEAM_MONTHLY_RELEASE, TEAM_CLIFF_MONTHS),
            ("security", SECURITY_ADDRESS, SECURITY_ALLOCATION,
             SECURITY_VESTING_MONTHS, SECURITY_MONTHLY_RELEASE, SECURITY_CLIFF_MONTHS),
        ]

        result: Dict[str, TeamVestingStatus] = {}
        for name, addr, allocation, vest_months, monthly, cliff in wallets:
            months_eff = min(months_elapsed_raw, vest_months)
            # Apply cliff: nothing unlocks before the cliff period
            if months_eff < cliff:
                unlocked = 0
            else:
                unlocked = min(months_eff * monthly, allocation)
            current_balance = self.state.get_balance(addr)
            withdrawn = allocation - current_balance
            available = max(0, unlocked - withdrawn)
            result[name] = {
                "total_allocation": allocation,
                "vesting_months": vest_months,
                "cliff_months": cliff,
                "monthly_release": monthly,
                "months_elapsed": months_eff,
                "total_unlocked": unlocked,
                "total_withdrawn": withdrawn,
                "available": available,
                "current_balance": current_balance,
            }

        return result

    def get_treasury_balances(self) -> Dict[str, WalletBalanceInfo]:
        """Return current balances for all treasury wallets."""
        from positronic.crypto.address import TREASURY_WALLETS
        result = {}
        for name, addr in TREASURY_WALLETS.items():
            result[name] = {
                "address": "0x" + addr.hex(),
                "balance": self.state.get_balance(addr),
            }
        return result

    def _init_treasury_log_table(self) -> None:
        """Create the treasury_log table if it does not exist."""
        try:
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS treasury_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    from_wallet TEXT NOT NULL,
                    from_address TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    block_height INTEGER NOT NULL
                )"""
            )
            self.db.safe_commit()
        except Exception as e:
            logger.debug("treasury_log table init: %s", e)

    def _load_treasury_log_from_db(self) -> None:
        """Load existing treasury transaction log from the database."""
        try:
            rows = self.db.execute(
                "SELECT timestamp, from_wallet, from_address, to_address, amount, block_height "
                "FROM treasury_log ORDER BY id ASC"
            ).fetchall()
            for row in rows:
                self._treasury_tx_log.append({
                    "timestamp": row[0] if not isinstance(row, dict) else row["timestamp"],
                    "from_wallet": row[1] if not isinstance(row, dict) else row["from_wallet"],
                    "from_address": row[2] if not isinstance(row, dict) else row["from_address"],
                    "to_address": row[3] if not isinstance(row, dict) else row["to_address"],
                    "amount": int(row[4] if not isinstance(row, dict) else row["amount"]),
                    "block_height": row[5] if not isinstance(row, dict) else row["block_height"],
                })
            if self._treasury_tx_log:
                logger.info("Loaded %d treasury log entries from DB", len(self._treasury_tx_log))
        except Exception as e:
            logger.debug("treasury_log load: %s", e)

    def _persist_treasury_entry(self, entry: dict) -> None:
        """Write a single treasury log entry to the database."""
        try:
            self.db.execute(
                "INSERT INTO treasury_log (timestamp, from_wallet, from_address, to_address, amount, block_height) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry["timestamp"],
                    entry["from_wallet"],
                    entry["from_address"],
                    entry["to_address"],
                    str(entry["amount"]),
                    entry["block_height"],
                ),
            )
            self.db.safe_commit()
        except Exception as e:
            logger.warning("Failed to persist treasury log entry: %s", e)

    def admin_transfer(self, wallet_name: str, recipient: bytes, amount: int) -> AdminTransferResult:
        """Transfer funds from a treasury wallet or founder account to a recipient (admin only)."""
        from positronic.crypto.address import TREASURY_WALLETS, TEAM_ADDRESS, SECURITY_ADDRESS
        if wallet_name == "founder" and hasattr(self, 'founder_address') and self.founder_address:
            wallet_addr = self.founder_address
        elif wallet_name not in TREASURY_WALLETS:
            return {"success": False, "error": f"Unknown wallet: {wallet_name}"}
        else:
            wallet_addr = TREASURY_WALLETS[wallet_name]
        # Enforce vesting limits on team and security wallets
        if wallet_addr == TEAM_ADDRESS:
            vesting = self.get_team_vesting_status()
            if amount > vesting["available"]:
                return {
                    "success": False,
                    "error": f"Team vesting limit: only {vesting['available']} available "
                             f"(cliff: {vesting['cliff_months']}mo, elapsed: {vesting['months_elapsed']}mo)",
                }
        elif wallet_addr == SECURITY_ADDRESS:
            vesting = self.get_security_vesting_status()
            if amount > vesting["available"]:
                return {
                    "success": False,
                    "error": f"Security vesting limit: only {vesting['available']} available "
                             f"(cliff: {vesting['cliff_months']}mo, elapsed: {vesting['months_elapsed']}mo)",
                }
        balance = self.state.get_balance(wallet_addr)
        if amount > balance:
            return {"success": False, "error": "Insufficient balance"}
        if amount <= 0:
            return {"success": False, "error": "Amount must be positive"}
        # Create a system transaction (like REWARD) so it gets included in the next block
        # This ensures ALL nodes see the transfer when they sync
        from positronic.core.transaction import Transaction, TxType
        import time as _time
        tx = Transaction(
            tx_type=TxType.TRANSFER,
            nonce=0,
            sender=wallet_addr,
            recipient=recipient,
            value=amount,
            gas_price=0,
            gas_limit=0,
            data=f"admin:{wallet_name}".encode(),
            signature=b"",  # System TX — no signature needed
            timestamp=_time.time(),
            chain_id=CHAIN_ID,
            ai_score=0.0,
            ai_model_version=0,
        )
        # Do NOT modify state here — let the executor handle it when the TX
        # is included in the next block. This ensures all nodes (seed + peers)
        # apply the same state change exactly once.
        # Reserve the funds so they can't be double-spent before block creation
        self.state.sub_balance(wallet_addr, amount)
        try:
            self.state.save_to_db(self.state_db)
        except Exception as e:
            logger.warning("admin_transfer: state persist failed: %s", e)
        # Add to pending system TXs — executor will credit recipient in block
        if not hasattr(self, '_pending_system_txs'):
            self._pending_system_txs = []
        self._pending_system_txs.append(tx)
        entry = {
            "timestamp": _time.time(),
            "from_wallet": wallet_name,
            "from_address": "0x" + wallet_addr.hex(),
            "to_address": "0x" + recipient.hex(),
            "amount": amount,
            "block_height": self.height,
            "tx_hash": tx.tx_hash.hex() if hasattr(tx, 'tx_hash') else "",
        }
        self._treasury_tx_log.append(entry)
        self._persist_treasury_entry(entry)
        return {
            "success": True,
            "from_wallet": wallet_name,
            "from_address": "0x" + wallet_addr.hex(),
            "to_address": "0x" + recipient.hex(),
            "amount": amount,
            "block_height": self.height,
            "tx_hash": entry.get("tx_hash", ""),
        }

    def get_treasury_transactions(self, limit: int = 50) -> List[TreasuryTransactionRecord]:
        """Return recent treasury transactions."""
        return list(reversed(self._treasury_tx_log[-limit:]))

    def create_forensic_report(self, tx_hash: str, sender: str, recipient: str,
                                value: int, block_height: int, ai_score: float):
        """Create a forensic transaction trace report."""
        return self.forensic_reporter.create_transaction_trace(
            tx_hash, sender, recipient, value, block_height, ai_score
        )

    # === DPoS Consensus Methods ===

    def submit_attestation(
        self,
        slot: int,
        block_hash: bytes,
        validator_address: bytes,
        signature: bytes,
    ) -> bool:
        """Submit a validator attestation for BFT finality."""
        if self.consensus is None:
            return False
        return self.consensus.submit_attestation(
            slot, block_hash, validator_address, signature
        )

    def is_finalized(self, slot: int, block_hash: bytes) -> bool:
        """Check if a block has reached BFT finality (2/3 supermajority)."""
        if self.consensus is None:
            return False
        return self.consensus.is_finalized(slot, block_hash)

    def report_double_sign(
        self,
        validator_address: bytes,
        block_hash_a: bytes,
        block_hash_b: bytes,
        slot: int,
        signature_a: bytes,
        signature_b: bytes,
    ) -> bool:
        """Report and slash a double-signing validator."""
        if self.consensus is None:
            return False
        return self.consensus.report_double_sign(
            validator_address, block_hash_a, block_hash_b,
            slot, signature_a, signature_b,
        )

    def get_consensus_status(self) -> Optional[dict]:
        """Get DPoS consensus status."""
        if self.consensus is None:
            return None
        return self.consensus.get_status()

    def get_network_health(self) -> dict:
        """Get comprehensive network health status."""
        import time as _time
        # Peer/sync info from node if available
        peer_count = 0
        sync_status = "synced"
        uptime = "--"
        if hasattr(self, '_node') and self._node:
            try:
                peer_count = len(self._node.peer_manager.get_all_peers())
            except Exception:
                pass
            try:
                sync_status = "syncing" if self._node.sync.state.syncing else "synced"
            except Exception:
                pass
        if hasattr(self, '_start_time'):
            elapsed = int(_time.time() - self._start_time)
            h, m = divmod(elapsed // 60, 60)
            uptime = f"{h}h {m}m"
        status = "healthy" if peer_count > 0 or self.height > 0 else "unknown"
        return {
            "status": status,
            "peer_count": peer_count,
            "sync_status": sync_status,
            "uptime": uptime,
            "height": self.height,
            "immune_status": self.immune_system.get_status(),
            "node_ranking": self.node_ranking.get_stats(),
            "ai_ranks": self.ai_rank_manager.get_stats(),
            "wallet_registry": self.wallet_registry.get_stats(),
            "game_engine": self.game_engine.get_stats(),
            "governance": self.token_governance.get_stats(),
            "model_governance": self.model_governance.get_stats(),
            "forensics": self.forensic_reporter.get_stats(),
            "trust": self.trust_manager.get_stats(),
            "paymaster": self.paymaster_registry.get_stats(),
            "smart_wallets": self.smart_wallet_registry.get_stats(),
            "onchain_game": self.onchain_game.get_stats(),
            "ai_agents": self.agent_registry.get_stats(),
            "zk_privacy": self.zk_manager.get_stats(),
            "bridge": self.bridge.get_stats(),
            "bridge_v2": self.lock_mint_bridge.get_stats(),
            "depin": self.depin_registry.get_stats(),
            "post_quantum": self.pq_manager.get_stats(),
            "did": self.did_registry.get_stats(),
        }

    def get_stats(self) -> dict:
        """Return comprehensive blockchain statistics."""
        stats = {
            "chain_id": CHAIN_ID,
            "height": self.height,
            "head_hash": self.chain_head.hash_hex if self.chain_head else None,
            "state": self.state.get_stats(),
            "ai": self.ai_gate.get_stats(),
            "quarantine": self.quarantine_pool.get_stats(),
            "training": self.ai_trainer.get_stats(),
            "wallet_registry": self.wallet_registry.get_stats(),
            "immune_system": self.immune_system.get_status(),
            "node_ranking": self.node_ranking.get_stats(),
            "ai_ranks": self.ai_rank_manager.get_stats(),
            "game": self.game_engine.get_stats(),
            "play_to_mine": self.promotion_manager.get_stats(),
            "governance": self.token_governance.get_stats(),
            "model_governance": self.model_governance.get_stats(),
            "trust": self.trust_manager.get_stats(),
            "tokens": self.token_registry.get_stats(),
            "paymaster": self.paymaster_registry.get_stats(),
            "smart_wallets": self.smart_wallet_registry.get_stats(),
            "onchain_game": self.onchain_game.get_stats(),
            "ai_agents": self.agent_registry.get_stats(),
            "zk_privacy": self.zk_manager.get_stats(),
            "bridge": self.bridge.get_stats(),
            "bridge_v2": self.lock_mint_bridge.get_stats(),
            "depin": self.depin_registry.get_stats(),
            "post_quantum": self.pq_manager.get_stats(),
            "did": self.did_registry.get_stats(),
        }
        if self.consensus:
            stats["consensus"] = self.consensus.get_status()
        return stats
