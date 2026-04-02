"""
Positronic - Global Constants
All chain parameters defined in one place.
"""

# ===== Coin Parameters =====
COIN_NAME = "Positronic"
COIN_SYMBOL = "ASF"
DECIMALS = 18
BASE_UNIT = 10 ** DECIMALS  # 1 ASF = 10^18 base units (like wei)
TOTAL_SUPPLY = 1_000_000_000 * BASE_UNIT  # 1 Billion ASF
CHAIN_ID = 420420

# ===== Sub-denominations (Sci-Fi / Asimov Themed) =====
UNIT = 1                     # 1 base unit (smallest)
SPARK = 10**3                # 1,000 Unit = 1 Spark
FLUX = 10**6                 # 1,000 Spark = 1 Flux
CORE = 10**9                 # 1,000 Flux = 1 Core
NODE = 10**12                # 1,000 Core = 1 Node
LAYER = 10**15               # 1,000 Node = 1 Layer
POSI = 10**18                # 1,000 Layer = 1 ASF (= BASE_UNIT)



# Sub-denomination names mapping
DENOMINATIONS = {
    "unit": UNIT,
    "spark": SPARK,
    "flux": FLUX,
    "core": CORE,
    "node": NODE,
    "layer": LAYER,
    "posi": POSI,
    "asf": POSI,
}

# ===== Block Parameters =====
BLOCK_TIME = 12  # seconds (Ethereum-aligned)
SLOTS_PER_EPOCH = 32
EPOCH_DURATION = BLOCK_TIME * SLOTS_PER_EPOCH  # 384 seconds
MAX_BLOCK_SIZE = 2 * 1024 * 1024  # 2 MB
BLOCK_GAS_LIMIT = 30_000_000
MIN_GAS_PRICE = 1  # minimum gas price in base units
BLOCK_HEADER_VERSION = 1

# ===== Hashing =====
HASH_ALGORITHM = "sha512"  # SHA-512 for quantum resistance
HASH_SIZE = 64  # 64 bytes = 512 bits
ADDRESS_SIZE = 20  # 20 bytes for EVM compatibility

# ===== Consensus v2: Three-Layer Validator System =====
# Layer 1: Block Producers (21 weighted-random per epoch from ALL eligible)
# Layer 2: Attesters (ALL validators with 32+ ASF stake, vote on blocks)
# Layer 3: Node Operators (any participant, earn relay fees)
MIN_STAKE = 32 * BASE_UNIT  # 32 ASF to become validator (like ETH's 32 ETH)
MAX_VALIDATORS = 10_000  # Soft cap for performance (was 100 hard cap)
MAX_VALIDATORS_SOFT_CAP = 10_000  # Alias for clarity
MIN_VALIDATORS = 1  # Minimum 1 node - network must work even with a single validator
FINALITY_THRESHOLD = 2 / 3  # 66.7% supermajority for BFT finality
COMMITTEE_SIZE = 21  # Per-slot proposer committee
BLOCK_PRODUCER_COUNT = 21  # Block producers per epoch (same as committee)
MAX_VALIDATORS_HARD_CAP = 100_000  # Absolute max validators for future scaling

# ===== Rewards =====
INITIAL_BLOCK_REWARD = 24 * BASE_UNIT  # 24 ASF per block (Bitcoin-style mining)
HALVING_INTERVAL = 5_256_000  # Halving every ~5.2M blocks (~2 years at 12s blocks)
MINING_SUPPLY_CAP = 550_000_000 * BASE_UNIT  # Max 550M ASF from block rewards
TAIL_EMISSION = int(0.5 * BASE_UNIT)  # Minimum 0.5 ASF/block forever (perpetual validator incentive)
PLAY_MINING_SUPPLY_CAP = 200_000_000 * BASE_UNIT  # Max 200M ASF from play-to-mine

# === Block Reward Distribution (Four-Way Split) ===
# 24 ASF/block → producer 4 | attesters 12 | nvn/validators 4 | treasury 4
PRODUCER_REWARD_SHARE = 1 / 6      # ~16.67% → 4 ASF of 24 ASF block reward
ATTESTATION_REWARD_SHARE = 0.50    # 50% → 12 ASF to all active validators (pro-rata)
NODE_OPERATOR_REWARD_SHARE = 1 / 6 # ~16.67% → 4 ASF to NVN; fallback: active validators
TREASURY_REWARD_SHARE = 1 / 6     # ~16.67% → 4 ASF to DAO treasury

# === Fee Distribution (per-transaction) ===
FEE_BURN_SHARE = 0.20              # 20% burned (deflationary pressure)
FEE_PRODUCER_SHARE = 0.25          # 25% to block producer
FEE_ATTESTER_SHARE = 0.25          # 25% to attesters (pro-rata by stake)
FEE_NODE_SHARE = 0.20              # 20% to node operators / community
FEE_TREASURY_SHARE = 0.10          # 10% to AI Training Treasury (DAO-governed)

# Deprecated aliases (backward compatibility for existing code/tests)
FEE_VALIDATOR_SHARE = FEE_PRODUCER_SHARE   # Alias: was 0.30, now 0.25
FEE_NVN_SHARE = FEE_ATTESTER_SHARE        # Alias: was 0.20, now 0.25

# ===== AI Validation (PoNC) =====
AI_ACCEPT_THRESHOLD = 0.35  # Score < 0.35 = accepted (tightened for security)
AI_QUARANTINE_THRESHOLD = 0.65  # 0.35-0.65 = quarantined, > 0.65 = rejected
AI_KILL_SWITCH_FP_RATE = 0.05  # Disable AI if false positive > 5%
AI_MODEL_VERSION = 1
QUARANTINE_REVIEW_INTERVAL = 100  # Re-evaluate quarantined TXs every 100 blocks
MAX_QUARANTINE_TIME = 1000  # Max blocks a TX stays in quarantine

# ===== Quantized Integer Scoring (Cross-Node Determinism) =====
# All consensus-critical decisions use integer basis points (0-10000)
# to eliminate IEEE 754 floating-point divergence across CPUs.
AI_SCORE_SCALE = 10000           # 1.0 = 10000 basis points
AI_SCORE_DEAD_ZONE = 200         # ±2.0% dead zone around thresholds
AI_ACCEPT_THRESHOLD_Q = 5000     # 0.50 * 10000 — normal TXs score ~0.35, accept below 0.50
AI_QUARANTINE_THRESHOLD_Q = 6500  # 0.65 * 10000 (tightened)
AI_WEIGHT_TAD_Q = 30             # TAD weight (integer, sums to 100)
AI_WEIGHT_MSAD_Q = 25            # MSAD weight
AI_WEIGHT_SCRA_Q = 25            # SCRA weight
AI_WEIGHT_ESG_Q = 20             # ESG weight

# ===== VM (PositronicVM) =====
MAX_CODE_SIZE = 64 * 1024  # 64 KB max contract size
MAX_STACK_DEPTH = 1024
MAX_MEMORY = 1 * 1024 * 1024  # 1 MB
MAX_CALL_DEPTH = 256
TX_BASE_GAS = 21_000  # Base gas for simple transfer
CREATE_GAS = 53_000  # Gas for contract creation

# ===== Game Ecosystem Fees =====
GAME_REGISTRATION_FEE = 10 * BASE_UNIT         # 10 ASF one-time fee to register a game
GAME_EMISSION_TAX = 0.05                        # 5% of daily emissions → TEAM_ADDRESS

# ===== Network =====
DEFAULT_P2P_PORT = 9000
DEFAULT_RPC_PORT = 8545  # Same as Ethereum for MetaMask compatibility
MAX_PEERS = 0  # 0 = unlimited; peer_security.py manages capacity
TARGET_PEERS = 50  # Actively maintain 50 connections
MIN_PEERS = 8  # Higher minimum for resilience
PROTOCOL_VERSION = 1
NETWORK_MAGIC = b"POSTR"  # Network identifier

# ===== Peer Security =====
PEER_SEC_MAX_PER_IP = 300        # Max new connections per IP per hour (testnet: high restarts expected)
PEER_SEC_MAX_PER_SUBNET = 600    # Max new connections per /24 subnet per hour
PEER_SEC_RATE_WINDOW = 3600      # Rate-limit sliding window (seconds)
PEER_SEC_POW_DIFFICULTY = 0x40   # PoW: SHA-256 first byte must be < this
PEER_SEC_FLOOD_MSG_PER_MIN = 100 # Kick peer if > 100 messages/minute
PEER_SEC_IDLE_KICK_SECONDS = 300 # Kick peer if idle > 5 minutes
PEER_SEC_LATENCY_KICK_MS = 8000  # Kick peer if latency > 8 seconds
PEER_SEC_SCORE_KICK = 0          # Kick peer if score drops below 0
PEER_SEC_BAN_DURATION = 3600     # Ban IP for 1 hour after kick
PEER_SEC_VALIDATOR_SCORE = 1000  # Validators: max score, never kicked

# ===== Genesis (Bitcoin-style: 75% mined, 25% genesis allocation) =====
GENESIS_TIMESTAMP = 1750000000.0  # Fixed: Sat Jun 15 2025 10:13:20 UTC — all nodes share this
TESTNET_GENESIS_TIMESTAMP = 1750100000.0  # Testnet uses a separate genesis timestamp
GENESIS_FOUNDER_SEED = b"Positronic-Genesis-Founder-2025-"  # 32-byte deterministic seed
GENESIS_HASH = bytes.fromhex(
    "fd8f17a5c474e7e78b546c716f536b09d5574261e7d3467c67738ca9368003de"
    "2888a61dbf4899eea8b3b669735f62ce50b0dbc1f8146913d76367549f0932a4"
)  # Canonical genesis hash — all nodes must produce this exact hash (MPT state root)
TESTNET_GENESIS_HASH = bytes.fromhex(
    "38b4904c1aaafa2d13f2173d4f8a27a9c576d8b8599679c728e6287e2a7aa908"
    "b335614d0658b145baf09c789cec1f4294da0efb25652a4fd9c5e95263796bdb"
)  # Canonical testnet genesis hash — all testnet nodes must produce this exact hash
# Genesis allocations (250M total — all locked/restricted, NO founder pre-mine)
AI_TREASURY_ALLOCATION = 100_000_000 * BASE_UNIT  # 100M ASF (10%) — DAO-governed, locked
COMMUNITY_ALLOCATION = 100_000_000 * BASE_UNIT    # 100M ASF (10%) — faucet/airdrops
TEAM_ALLOCATION = 30_000_000 * BASE_UNIT          # 30M ASF (3%) — 36-month cliff, 96-month vesting
SECURITY_ALLOCATION = 20_000_000 * BASE_UNIT      # 20M ASF (2%) — 6-month cliff, 96-month vesting
# Mining allocations (750M total — earned through work)
MINING_ALLOCATION = 550_000_000 * BASE_UNIT       # 550M ASF (55%) — node mining
PLAY_MINING_ALLOCATION = 200_000_000 * BASE_UNIT  # 200M ASF (20%) — play-to-mine
# Genesis total (should equal 250M, NOT total supply)
GENESIS_SUPPLY = AI_TREASURY_ALLOCATION + COMMUNITY_ALLOCATION + TEAM_ALLOCATION + SECURITY_ALLOCATION
# Vesting schedules (96 months = 8 years for ALL treasury allocations)
TEAM_VESTING_MONTHS = 96  # 8 years (96 months linear after cliff)
TEAM_CLIFF_MONTHS = 36    # 36-month cliff (3 years) before any tokens unlock
TEAM_MONTHLY_RELEASE = TEAM_ALLOCATION // TEAM_VESTING_MONTHS
SECURITY_VESTING_MONTHS = 96  # 8 years
SECURITY_CLIFF_MONTHS = 6    # 6-month cliff before any tokens unlock
SECURITY_MONTHLY_RELEASE = SECURITY_ALLOCATION // SECURITY_VESTING_MONTHS
AI_TREASURY_VESTING_MONTHS = 96  # 8 years
AI_TREASURY_MONTHLY_RELEASE = AI_TREASURY_ALLOCATION // AI_TREASURY_VESTING_MONTHS
COMMUNITY_VESTING_MONTHS = 96  # 8 years
COMMUNITY_MONTHLY_RELEASE = COMMUNITY_ALLOCATION // COMMUNITY_VESTING_MONTHS
# Legacy compatibility
FOUNDER_ALLOCATION = 0  # No founder pre-mine (Bitcoin model)
STAKING_REWARDS_POOL = 0  # Removed — validators earn from mining + fees

# ===== Dynamic Block Parameters =====
MIN_BLOCK_GAS_LIMIT = 15_000_000  # Floor: 15M gas
MAX_BLOCK_GAS_LIMIT = 60_000_000  # Ceiling: 60M gas
GAS_LIMIT_ADJUSTMENT_FACTOR = 1024  # Max 1/1024 change per block (like Ethereum)
TARGET_GAS_UTILIZATION = 0.5  # Target 50% block utilization

# ===== Checkpoints =====
CHECKPOINT_INTERVAL = 10_000  # Create checkpoint every 10,000 blocks
SNAPSHOT_INTERVAL = 100_000  # Full state snapshot every 100,000 blocks

# ===== Multi-Signature =====
MAX_MULTISIG_SIGNERS = 20  # Maximum signers in a multisig
MIN_MULTISIG_THRESHOLD = 1  # Minimum threshold (1-of-N)

# ===== Gift System =====
GIFT_DEFAULT_AMOUNT = 50 * BASE_UNIT  # 50 ASF per gift (32 for staking + 18 for testing)
GIFT_MAX_DAILY = 1000  # Max gifts per day
GIFT_COOLDOWN = 0  # Effectively forever — one claim per IP/address

# ===== Neural AI Engine =====
AI_NEURAL_ACTIVATION_THRESHOLD = 500   # Min training samples before switching to neural
AI_PRETRAINING_EPOCHS = 50             # Contrastive pre-training epochs at genesis
AI_BATCH_SIZE = 64                     # Training batch size
AI_LEARNING_RATE = 0.001               # Base learning rate for Adam optimizer
AI_FEDERATED_ROUND_INTERVAL = 100      # Blocks between federated averaging rounds
AI_CONSENSUS_SEED = 42                 # Deterministic seed for consensus-safe inference
AI_MODEL_SAVE_INTERVAL = 10000         # Blocks between model checkpoint saves
AI_VAE_LATENT_DIM = 8                  # Latent dimension for VAE
AI_VAE_BETA = 1.0                      # Beta for VAE loss (KL weight)
AI_TEMPORAL_D_MODEL = 64               # Transformer model dimension
AI_TEMPORAL_HEADS = 4                  # Number of attention heads
AI_TEMPORAL_MAX_SEQ = 50               # Max sequence length for temporal model
AI_LSTM_HIDDEN = 32                    # LSTM hidden size for stability guardian
AI_LSTM_LAYERS = 2                     # Number of LSTM layers
AI_META_MIN_STEPS = 200                # Min training steps for meta-ensemble activation
AI_GRADIENT_CLIP_NORM = 1.0            # Gradient clipping max norm
AI_WARMUP_STEPS = 100                  # Learning rate warmup steps

# ===== AI Scalability =====
AI_CACHE_SIZE = 10_000                 # LRU inference cache size (up from 1000)
AI_MAX_LATENCY_MS = 2000               # Max AI scoring latency before kill switch
AI_TIERED_LIGHT_VALUE = 100 * BASE_UNIT  # Transactions below 100 ASF = light scoring
AI_TIERED_LIGHT_REPUTATION = 0.9       # Senders above 0.9 reputation = light scoring
AI_TIERED_MEDIUM_VALUE = 1000 * BASE_UNIT  # Transactions below 1000 ASF = medium scoring

# ===== Play-to-Mine =====
MINER_THRESHOLD = 4 * BASE_UNIT        # 4 ASF → "miner" status (informational)
NODE_THRESHOLD = MIN_STAKE              # 32 ASF → auto-stake + node + NVN
AUTO_STAKE_AMOUNT = MIN_STAKE           # Amount auto-staked on promotion (32 ASF)
GAME_REWARD_GAS = 0                     # Game reward transactions are gas-free

# ===== RPC Security =====
RPC_FAUCET_COOLDOWN_IP = 28800          # 8 hours between faucet drips per IP
RPC_FAUCET_MAX_PER_IP = 1000              # Max faucet drips per IP per day
RPC_GAME_MAX_PER_HOUR = 10             # Max game result submissions per address per hour
RPC_ACCESS_DENIED_CODE = -32403        # JSON-RPC error code for access denied

# ===== Phase 15: Security Hardening =====

# Graduated Kill-Switch Levels (replaces binary AI_KILL_SWITCH_FP_RATE)
KS_LEVEL_0_MAX_FP = 0.03              # Normal operation (FP < 3%)
KS_LEVEL_1_MAX_FP = 0.05              # Elevated: raise threshold to 0.90
KS_LEVEL_2_MAX_FP = 0.08              # Degraded: only TAD+ESG, threshold 0.92
KS_LEVEL_3_MAX_FP = 1.0               # Fallback: disable AI, use heuristic rules

KS_LEVEL_1_ACCEPT = 0.90              # More lenient acceptance at Level 1
KS_LEVEL_2_ACCEPT = 0.92              # Even more lenient at Level 2
KS_LEVEL_1_ACCEPT_Q = 9000            # Integer version (0.90 * 10000)
KS_LEVEL_2_ACCEPT_Q = 9200            # Integer version (0.92 * 10000)

# Fallback Validator Thresholds
FALLBACK_GAS_SPIKE_MULTIPLIER = 10     # Gas > 10x median = reject
FALLBACK_BALANCE_RATIO_LIMIT = 0.80    # Value > 80% balance = quarantine
FALLBACK_SENDER_BURST_LIMIT = 5        # >5 TXs in 10 blocks = reject
FALLBACK_BURST_WINDOW_BLOCKS = 10
FALLBACK_GAS_HISTORY_SIZE = 1000       # Rolling gas price window

# SCRA v2: Bytecode Analysis
SCRA_MAX_BYTECODE_SIZE = 49152         # 48KB
SCRA_DELEGATECALL_RISK = 0.30
SCRA_SSTORE_AFTER_CALL_RISK = 0.40     # Reentrancy pattern
SCRA_SELFDESTRUCT_RISK = 0.35
SCRA_PROXY_PATTERN_RISK = 0.25
SCRA_GAS_BOMB_THRESHOLD = 10_000_000   # Subcall gas > 10M

# Game Server Security
GAME_API_RATE_LIMIT = 100              # Max requests per minute per API key
GAME_API_RATE_WINDOW = 60              # Window in seconds
GAME_SCORE_MAX_PER_MINUTE = 10_000     # Max score per minute of play
GAME_TRUST_DECAY_BASE = 10             # Base trust loss
GAME_TRUST_DECAY_EXPONENT = 2          # trust_loss = base * exp^cheat_count
GAME_HEARTBEAT_INTERVAL = 3600         # 1 hour heartbeat requirement
GAME_HEARTBEAT_GRACE = 7200            # 2 hour grace before suspend

# Cross-Model Consensus
MODEL_VETO_THRESHOLD = 0.90            # Any model > 0.90 triggers veto
MODEL_VETO_MIN_SCORE = 0.85            # Veto raises floor to quarantine
MODEL_VETO_THRESHOLD_Q = 9000          # Integer version
MODEL_VETO_MIN_SCORE_Q = 8500          # Integer version
MODEL_CONFIDENCE_MIN = 0.3             # Below this, model score weighted less

# Validator Censorship Detection
CENSORSHIP_WINDOW_EPOCHS = 3           # Monitor over 3 epochs
CENSORSHIP_MIN_INCLUSION_RATE = 0.70   # Must include 70%+ of valid TXs
CENSORSHIP_HIGH_FEE_EXCLUSION = 5      # Flag if >5 high-fee TXs excluded

# ===== Phase 16: AI Intelligence Upgrade =====

# Feature dimensions
AI_FEATURE_DIM = 35                    # Was 26, now 35 with 9 new features
AI_FEATURE_DIM_V1 = 26                 # Old dimension for weight migration

# Cross-Model Attention
CROSS_ATTENTION_HEADS = 4              # Attention heads in cross-model layer
CROSS_ATTENTION_DIM = 32               # Hidden dimension
CROSS_ATTENTION_MIN_STEPS = 500        # Min training steps before activation
META_ENSEMBLE_GATE_INIT = 0.0          # Initial gate value (0 = old behavior)

# Curriculum Learning
CURRICULUM_PHASE1_END = 500            # Steps: only clear examples
CURRICULUM_PHASE2_END = 2000           # Steps: include quarantine zone
CURRICULUM_EASY_THRESHOLD = 0.50       # Below = clearly safe
CURRICULUM_HARD_THRESHOLD = 0.95       # Above = clearly malicious

# Plateau Detection
PLATEAU_WINDOW = 200                   # Steps to check for plateau
PLATEAU_MIN_IMPROVEMENT = 0.01         # 1% minimum improvement
PLATEAU_LR_BOOST = 1.5                # LR multiplier on plateau
PLATEAU_MAX_BOOST = 3.0               # Maximum LR boost

# Hard Negative Mining
HARD_NEGATIVE_BUFFER_SIZE = 500        # Capacity of hard negative buffer
HARD_NEGATIVE_MIN_SCORE = 0.80         # Lower bound of hard negative zone
HARD_NEGATIVE_MAX_SCORE = 0.90         # Upper bound of hard negative zone
HARD_NEGATIVE_BATCH_RATIO = 0.20       # 20% of each batch from hard negatives

# GAT Account Features
GAT_ACCOUNT_CACHE_SIZE = 5000          # Max cached account lookups

# ===== Phase 17: GOD CHAIN =====

# Gas Oracle (EIP-1559 style — advisory only, never rejects)
GAS_BASE_FEE_FLOOR = 1                # Minimum base fee (always at least 1)
GAS_BASE_FEE_CEILING = 1000           # Maximum base fee cap
GAS_BASE_FEE_CHANGE_DENOM = 8         # Max 12.5% change per block
GAS_FEE_HISTORY_SIZE = 256            # Fee history blocks to keep

# Transaction Priority Lanes
TX_LANE_FAST_MAX_GAS = 42_000         # Max gas for fast lane (simple transfers)
TX_LANE_HEAVY_MIN_GAS = 500_000       # Min gas threshold for heavy lane

# Parallel Validation
TX_VALIDATION_WORKERS = 4             # Thread pool size for parallel TX validation
TX_BATCH_SIZE = 50                    # Max batch size for parallel processing

# Compact Block Relay
COMPACT_BLOCK_TX_THRESHOLD = 10       # Use compact blocks when > 10 TXs

# Adaptive Peer Scoring Weights
PEER_SCORE_LATENCY_WEIGHT = 0.20      # 20% weight for latency score
PEER_SCORE_RELIABILITY_WEIGHT = 0.25  # 25% weight for reliability
PEER_SCORE_BANDWIDTH_WEIGHT = 0.15    # 15% weight for bandwidth
PEER_SCORE_CHAIN_WEIGHT = 0.25        # 25% weight for chain sync status
PEER_SCORE_BEHAVIOR_WEIGHT = 0.15     # 15% weight for message behavior
PEER_SCORE_DECAY_RATE = 0.99          # Score decay rate per interval

# Network Partition Detection
PARTITION_BLOCK_TIMEOUT_MULTIPLIER = 3  # No block for 3x BLOCK_TIME = degraded
PARTITION_MIN_PEERS_THRESHOLD = 3       # Below 3 peers = potential partition

# VM v2 Precompile Addresses
PRECOMPILE_BASE64_ADDR = 5            # Base64 encode/decode
PRECOMPILE_JSON_PARSE_ADDR = 6        # JSON field extraction
PRECOMPILE_BATCH_VERIFY_ADDR = 7      # Batch Ed25519 verification

# HD Wallet (BIP-32/44 style)
HD_WALLET_PURPOSE = 44                # BIP-44 purpose
HD_WALLET_COIN_TYPE = 420420          # Match CHAIN_ID
HD_WALLET_MAX_ACCOUNTS = 100          # Max derived accounts
HD_WALLET_MAX_ADDRESSES = 1000        # Max addresses per account

# ===== Phase 18: Audit Fixes =====

# Immune System — timed blocking (24h default, not permanent)
IMMUNE_BLOCK_DURATION = 24 * 3600     # 24 hours default block duration
IMMUNE_APPEAL_DEPOSIT = 10 * BASE_UNIT  # 10 ASF deposit for appeal

# TRUST — daily rate limit to prevent bot farming
TRUST_DAILY_CAP = 50                  # Max TRUST points earned per day per address

# Paymaster — minimum TRUST score for gasless transactions
MIN_TRUST_FOR_GASLESS = 10            # Minimum trust score for gasless TX

# AI Model Governance — delayed activation for safety
AI_MODEL_ACTIVATION_DELAY = 1000      # Blocks before model update activates

# DID — social recovery mechanism
RECOVERY_DELAY = 7 * 24 * 3600       # 7-day waiting period for DID recovery

# Play-to-Earn — consolidated emission caps (single source of truth)
P2E_PLAYER_DAILY_CAP = 50 * BASE_UNIT        # 50 ASF per player per day
P2E_GLOBAL_DAILY_CAP = 200 * BASE_UNIT       # 200 ASF total P2E per day
P2E_SESSION_MAX_REWARD = 10 * BASE_UNIT      # 10 ASF max per game session
P2E_DAILY_MAX_GAMES = 20                     # Max games per player per day

# ===== Phase 19: Real Post-Quantum Cryptography =====

PQ_HMAC_ITERATIONS = 3                            # HMAC chain depth for key derivation
PQ_BINDING_TAG = b"POSITRONIC_PQ_V2"              # Domain separator for PQ signatures
PQ_SIG_TAG = b"PQ_SIG_V2"                         # Tag for signature HMAC
PQ_VER_TAG = b"PQ_VER_V2"                         # Tag for verification derivation

# ===== Phase 24: zk-SNARK Privacy (Enhanced ZK) =====

ZK_PRIME = (1 << 255) - 19                        # Curve25519 field prime
ZK_RANGE_BITS = 64                                # Bits for range proofs
ZK_MERKLE_DEPTH = 16                              # Max Merkle tree depth for membership proofs
ZK_DOMAIN_BALANCE = b"ZK_BAL_V2"                  # Domain tag for balance proofs
ZK_DOMAIN_MEMBERSHIP = b"ZK_MEM_V2"               # Domain tag for membership proofs
ZK_DOMAIN_OWNERSHIP = b"ZK_OWN_V2"                # Domain tag for ownership proofs

# ===== Phase 21: AI Explainability (XAI) =====

XAI_MAX_EXPLANATION_LEN = 500                     # Max chars in explanation string
XAI_ATTENTION_THRESHOLD = 0.5                     # Min attention weight to report

# ===== Phase 22: DePIN Economic Layer =====

DEPIN_GPU_MULTIPLIER = 5                          # GPU devices earn 5x base
DEPIN_STORAGE_MULTIPLIER = 3                      # Storage devices earn 3x base
DEPIN_SENSOR_MULTIPLIER = 2                       # Sensor/Camera devices earn 2x base
DEPIN_NETWORK_MULTIPLIER = 1                      # Network devices earn 1x base
DEPIN_DAILY_DEVICE_CAP = 100                      # Max reward units per device per day
DEPIN_UPTIME_WEIGHT = 0.4                         # 40% weight for uptime in scoring
DEPIN_DATA_WEIGHT = 0.4                           # 40% weight for data quality
DEPIN_FRESHNESS_WEIGHT = 0.2                      # 20% weight for heartbeat freshness
DEPIN_FRESHNESS_DECAY_MINS = 60                   # Freshness decays to 0 at 60 min

# ===== Phase 23: Autonomous AI Agents =====

AGENT_DEFAULT_RATE_LIMIT = 60                     # Max actions per hour per agent
AGENT_MAX_SPEND_DEFAULT = 10 * BASE_UNIT          # 10 ASF max per action
AGENT_DAILY_LIMIT_DEFAULT = 100 * BASE_UNIT       # 100 ASF daily spending cap
AGENT_MIN_TRUST_FOR_AUTONOMOUS = 10               # Min trust score for autonomous mode

# ===== Phase 20: Cross-Chain Bridge v2 =====

BRIDGE_QUORUM = 3                                 # Required relayer confirmations
BRIDGE_RELAYERS_MIN = 5                           # Minimum relayers for bridge operation
BRIDGE_CHALLENGE_PERIOD = 86400                   # 24-hour fraud proof window (seconds)
BRIDGE_MIN_LOCK = 1 * BASE_UNIT                   # Minimum lock amount (1 ASF)
BRIDGE_FEE_BPS = 30                               # Bridge fee: 0.3% (30 basis points)

# API Versioning
API_VERSION = "1.0"

# Wallet Version (BIP-39 upgrade)
WALLET_VERSION = 2

# ===== Phase 25: Emergency Control System =====

# Timelock durations per multi-sig action type (seconds)
EMERGENCY_TIMELOCK_HALT = 0                   # Immediate (no delay for emergencies)
EMERGENCY_TIMELOCK_RESUME = 3600              # 1 hour cooldown before resuming
EMERGENCY_TIMELOCK_ROLLBACK = 21600           # 6 hours before rollback executes
EMERGENCY_TIMELOCK_UPGRADE = 86400            # 24 hours before upgrade approval
EMERGENCY_TIMELOCK_KEY_ROTATION = 172800      # 48 hours for key rotation

# Multi-sig parameters
EMERGENCY_MULTISIG_REQUIRED = 3               # 3-of-5 signatures required
EMERGENCY_MULTISIG_TOTAL = 5                  # Total authorized signers

# Upgrade Manager
UPGRADE_MONITOR_BLOCKS = 100                  # Monitor for 100 blocks after activation
UPGRADE_ROLLBACK_ERROR_THRESHOLD = 0.05       # Auto-rollback if error rate > 5%
UPGRADE_MIN_ACTIVATION_DELAY = 1000           # Min blocks ahead for scheduling

# ===== Phase 29: AI Agent Marketplace =====
AGENT_REGISTRATION_FEE = 50 * BASE_UNIT       # 50 ASF to register an AI agent
AGENT_TASK_FEE_MIN = 1 * BASE_UNIT            # Minimum 1 ASF per task submission
AGENT_QUALITY_THRESHOLD = 7000                # Min quality score (basis points) for rewards
AGENT_REWARD_SHARE = 0.85                     # 85% of task fee to agent developer
AGENT_PLATFORM_FEE = 0.10                     # 10% platform fee → AI Treasury
AGENT_BURN_FEE = 0.05                         # 5% burned (deflationary)
AGENT_CATEGORIES = [
    "ANALYSIS", "AUDIT", "GOVERNANCE", "CREATIVE", "DATA", "SECURITY",
]
MAX_AGENTS = 10_000                           # Soft cap on registered agents
AGENT_TASK_TIMEOUT = 300                      # 5 min max execution time per task
AGENT_COUNCIL_MIN_VOTES = 3                   # Min votes to decide agent approval
AGENT_COUNCIL_APPROVAL_PCT = 0.60             # 60% approval required
AGENT_AI_MAX_RISK = 0.60                      # Max AI risk score for auto-approval
AGENT_MIN_TRUST = 10                          # Below this, agent is auto-suspended

# ===== Phase 30: RWA Tokenization Engine (PRC-3643) =====
RWA_REGISTRATION_FEE = 200 * BASE_UNIT        # 200 ASF to register RWA token
RWA_COMPLIANCE_CHECK_FEE = 1 * BASE_UNIT      # 1 ASF per compliance check
RWA_DIVIDEND_FEE = 5 * BASE_UNIT              # 5 ASF per dividend distribution
RWA_MIN_KYC_LEVEL = 2                         # Min KYC level for RWA trading
RWA_ASSET_TYPES = [
    "REAL_ESTATE", "EQUITY", "COMMODITY", "BOND", "ART",
]
RWA_MAX_HOLDERS = 100_000                     # Max holders per RWA token
RWA_TRANSFER_COOLDOWN = 86400                 # 24h cooldown between large transfers
RWA_LARGE_TRANSFER_THRESHOLD = 0.01           # >1% of supply = large transfer
RWA_MAX_TOKENS = 10_000                       # Soft cap on RWA tokens
RWA_DIVIDEND_MIN_AMOUNT = 1 * BASE_UNIT       # Min 1 ASF dividend distribution
RWA_COUNCIL_MIN_VOTES = 3                     # Min votes to approve RWA token
RWA_COUNCIL_APPROVAL_PCT = 0.60               # 60% approval required
RWA_AI_MAX_RISK = 0.50                        # Max AI risk for auto-approval (stricter)

# ===== Phase 31: ZKML — Zero-Knowledge Machine Learning =====
ZKML_ENABLED = False                            # Feature flag (enable after testing)
ZKML_PROOF_TIMEOUT_MS = 2000                    # Max 2s for proof generation
ZKML_VERIFICATION_GAS = 50_000                  # Gas cost for ZK verification
ZKML_MODEL_COMMITMENT_INTERVAL = 100            # Commit model hash every 100 blocks
ZKML_MIN_PROOFS_PER_BLOCK = 0                   # Don't require proofs initially
ZKML_PROOF_FORMAT = "fiat-shamir-sha512"         # Fiat-Shamir hash-based commitment scheme
ZKML_QUANTIZATION_BITS = 16                     # Fixed-point quantization for circuits
ZKML_MAX_CIRCUIT_DEPTH = 8                      # Max layers in provable circuit
ZKML_CHALLENGE_ROUNDS = 3                       # Fiat-Shamir challenge rounds
ZKML_PROOF_CACHE_SIZE = 1000                    # LRU cache for recent proofs

# ===== Phase 32: Cold Start Management =====
COLD_START_PHASE_A_END = 100_000
COLD_START_PHASE_B_END = 500_000
COLD_START_PHASE_A_ACCEPT_Q = 10000              # Accept all in Phase A
COLD_START_PHASE_A_QUARANTINE_Q = 10001           # Never quarantine in Phase A
COLD_START_PHASE_B_START_ACCEPT_Q = 9500          # Phase B starts lenient
COLD_START_PHASE_B_START_QUARANTINE_Q = 9900
COLD_START_PHASE_B_KS_FP_RATE = 0.005            # 0.5% FP rate trigger in Phase B
COLD_START_FP_RATE_LIMIT_PER_HOUR = 10            # Max FP reports per reporter per hour

# ===== Phase 32b: Validator Pool Separation =====
QUARANTINE_JUDGE_COUNT = 7                         # Quarantine judges per epoch
POOL_CONCENTRATION_LIMIT = 0.15                    # 15% max per entity in any pool
MIN_VALIDATORS_FOR_POOLS = 28                      # 21 producers + 7 judges minimum

# ===== Phase 32c: Neural Self-Preservation =====
NSP_SNAPSHOT_INTERVAL = 500                          # Blocks between automatic snapshots
NSP_MAX_SNAPSHOTS = 50                               # Maximum snapshots to retain
NSP_SNAPSHOT_TIMEOUT_MS = 200                        # Warning if snapshot exceeds 200ms
NSP_P2P_BACKUP_PEERS = 5                             # Number of trusted peers for backup

# ===== Phase 32d: Graceful Degradation =====
DEGRAD_LEVEL_5_ACCEPT_Q = 8500                       # Full: all 4 models
DEGRAD_LEVEL_5_QUARANTINE_Q = 9500
DEGRAD_LEVEL_4_ACCEPT_Q = 8000                       # Reduced: TAD + ESG
DEGRAD_LEVEL_4_QUARANTINE_Q = 9200
DEGRAD_LEVEL_3_ACCEPT_Q = 7500                       # Minimal: TAD only
DEGRAD_LEVEL_3_QUARANTINE_Q = 9000
DEGRAD_LEVEL_2_ACCEPT_Q = 7000                       # Guardian: FallbackValidator
DEGRAD_LEVEL_2_QUARANTINE_Q = 8500
DEGRAD_LEVEL_1_ACCEPT_Q = 10000                      # Open: accept all
DEGRAD_LEVEL_1_QUARANTINE_Q = 10001

# ===== Phase 32e: Neural Watchdog =====
WATCHDOG_CHECK_INTERVAL = 0.5                        # Seconds between heartbeat checks
WATCHDOG_MISS_THRESHOLD = 3                          # Consecutive misses before action
WATCHDOG_SIGUSR1_WAIT = 5.0                          # Seconds to wait after SIGUSR1

# ===== Phase 32f: Pathway Memory =====
PATHWAY_WEIGHT_FLOOR = 0.1                           # Minimum pathway weight (Hebbian floor)
PATHWAY_WEIGHT_CEILING = 1.0                         # Maximum pathway weight (Hebbian ceiling)
PATHWAY_DECAY_FACTOR = 0.7                           # Weight multiplier on failure
PATHWAY_BOOST_FACTOR = 1.3                           # Weight multiplier on success
PATHWAY_ALT_BOOST_FACTOR = 1.1                       # Alternative pathway boost on neighbour failure
PATHWAY_FAILURE_BUFFER_SIZE = 100                    # Ring buffer size for failure timestamps
PATHWAY_PREEMPTIVE_THRESHOLD = 3                     # Failures in buffer before pre-emptive action
PATHWAY_CORRELATION_THRESHOLD = 0.7                  # Jaccard similarity threshold for correlation

# ===== Phase 32g: Neural Recovery =====
RECOVERY_MAX_ATTEMPTS = 3                               # Max recovery attempts before lockout

# ===== Phase 32h: Online Learning Extension =====
OLE_MIN_BATCH_SIZE = 100
OLE_MAX_BUFFER_SIZE = 10000
OLE_TRAIN_INTERVAL_BLOCKS = 1000
OLE_RATE_LIMIT_PER_HOUR = 10
OLE_QUALITY_GATE_DROP = 0.05
OLE_EWC_LAMBDA = 0.5

# ===== Phase 32i: Model Communication Bus =====
MODEL_BUS_BOOST_CAP = 500  # max ±500bp context boost

# ===== Phase 32j: Concept Drift Detection =====
DRIFT_WINDOW_SIZE = 500
DRIFT_LOW_THRESHOLD = 10.0     # percent
DRIFT_MEDIUM_THRESHOLD = 30.0  # percent
DRIFT_MAX_ALERTS = 50
DRIFT_WEIGHT_REDUCTION = 0.20  # 20% reduction

# ===== Phase 33: Tendermint BFT Timeouts =====
BFT_PROPOSE_TIMEOUT = 12.0                          # Seconds to wait for a proposal
BFT_PREVOTE_TIMEOUT = 6.0                           # Seconds to wait for 2/3 prevotes
BFT_PRECOMMIT_TIMEOUT = 6.0                         # Seconds to wait for 2/3 precommits
BFT_TIMEOUT_DELTA = 1.0                             # Per-round timeout increase (seconds)
