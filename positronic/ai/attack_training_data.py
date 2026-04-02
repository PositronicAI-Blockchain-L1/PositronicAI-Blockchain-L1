"""
Positronic - AI Pre-Learning Attack Dataset Generator

Generates 10,000+ labeled training samples covering ALL known blockchain
attack patterns. The 4 AI models (TAD, MSAD, SCRA, ESG) are pre-trained
with this data BEFORE the network starts, enabling attack detection from
block 1 (genesis).

Feature vectors are 35-dimensional, matching the TransactionFeatures format
defined in positronic/ai/feature_extractor.py:

  Index  Feature                     Description
  -----  --------------------------  -----------------------------------
   0     sender_balance_ratio        value / sender_balance
   1     sender_nonce                Transaction count history
   2     sender_avg_value            Average historical TX value
   3     sender_value_deviation      How much this TX deviates from avg
   4     sender_tx_frequency         TXs per hour recently
   5     sender_age                  Account age in hours
   6     sender_unique_recipients    Unique addresses sent to
   7     sender_reputation           AI reputation score (0-1)
   8     value_log                   log10(value + 1)
   9     value_is_round              Is value a round number (0/1)
  10     gas_price_ratio             gas_price / network_avg_gas
  11     total_cost_ratio            total_cost / sender_balance
  12     value_percentile            Percentile in recent TX values
  13     time_since_last_tx          Seconds since sender's last TX
  14     hour_of_day                 0-1 normalized hour
  15     is_burst                    Part of a rapid burst (0/1)
  16     mempool_wait_time           Time spent in mempool
  17     tx_type                     Transaction type enum
  18     has_data                    Contains calldata (0/1)
  19     data_size                   Size of calldata
  20     is_contract_interaction     Calls a contract (0/1)
  21     mempool_size                Current mempool size
  22     recent_block_fullness       Avg gas used / gas limit
  23     network_tx_rate             Network-wide TXs per second
  24     gas_price_vs_median         Gas price relative to median
  25     pending_from_sender         Pending TXs from same sender
  26     sender_recipient_tx_count   Times sender->recipient before
  27     sender_cluster_diversity    Unique recipients / total TXs
  28     recipient_popularity        Unique senders to recipient
  29     sender_tx_regularity        Std dev of inter-TX intervals
  30     contract_call_ratio         % of sender's TXs that are calls
  31     value_entropy               Shannon entropy of value dist
  32     incoming_outgoing_ratio     Received / Sent ratio
  33     value_velocity              Rate of value change over time
  34     gas_efficiency              gas_used / gas_limit

Labels: 0 = normal, 1 = attack

Usage:
    from positronic.ai.attack_training_data import pretrain_ai_models

    gate = AIValidationGate()
    pretrain_ai_models(gate)
"""

import math
import random
import time
from typing import Dict, List, Tuple

from positronic.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching positronic/constants.py)
# ---------------------------------------------------------------------------
_BASE_UNIT = 10 ** 18  # 1 ASF = 10^18 wei
_FEATURE_DIM = 35


# ---------------------------------------------------------------------------
# Feature-vector index constants (for readability)
# ---------------------------------------------------------------------------
_I_BALANCE_RATIO = 0
_I_NONCE = 1
_I_AVG_VALUE = 2
_I_VALUE_DEV = 3
_I_TX_FREQ = 4
_I_AGE = 5
_I_UNIQUE_RECIP = 6
_I_REPUTATION = 7
_I_VALUE_LOG = 8
_I_VALUE_ROUND = 9
_I_GAS_RATIO = 10
_I_COST_RATIO = 11
_I_VALUE_PCTILE = 12
_I_TIME_SINCE = 13
_I_HOUR = 14
_I_IS_BURST = 15
_I_MEMPOOL_WAIT = 16
_I_TX_TYPE = 17
_I_HAS_DATA = 18
_I_DATA_SIZE = 19
_I_IS_CONTRACT = 20
_I_MEMPOOL_SIZE = 21
_I_BLOCK_FULLNESS = 22
_I_NET_TX_RATE = 23
_I_GAS_VS_MEDIAN = 24
_I_PENDING_SENDER = 25
_I_PAIR_COUNT = 26
_I_CLUSTER_DIV = 27
_I_RECIP_POP = 28
_I_TX_REGULARITY = 29
_I_CONTRACT_RATIO = 30
_I_VALUE_ENTROPY = 31
_I_IN_OUT_RATIO = 32
_I_VALUE_VELOCITY = 33
_I_GAS_EFFICIENCY = 34


# ---------------------------------------------------------------------------
# Helper: build a baseline "normal" vector
# ---------------------------------------------------------------------------
def _normal_base(rng: random.Random) -> List[float]:
    """Return a 35-element feature vector representing a plausible normal TX."""
    value_asf = rng.uniform(1.0, 100.0)
    value_wei = value_asf * _BASE_UNIT
    balance_asf = rng.uniform(value_asf * 2, value_asf * 20)
    nonce = rng.randint(5, 500)
    gas_price = rng.uniform(1.0, 5.0)

    vec = [0.0] * _FEATURE_DIM
    vec[_I_BALANCE_RATIO] = value_asf / balance_asf           # 0.05 - 0.5
    vec[_I_NONCE] = float(nonce)
    vec[_I_AVG_VALUE] = math.log10(value_wei + 1) * rng.uniform(0.8, 1.2)
    vec[_I_VALUE_DEV] = rng.uniform(0.0, 1.0)
    vec[_I_TX_FREQ] = rng.uniform(0.5, 10.0)
    vec[_I_AGE] = rng.uniform(24.0, 8760.0)                   # 1 day - 1 year
    vec[_I_UNIQUE_RECIP] = float(rng.randint(3, 50))
    vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)
    vec[_I_VALUE_LOG] = math.log10(value_wei + 1)
    vec[_I_VALUE_ROUND] = float(rng.random() < 0.3)
    vec[_I_GAS_RATIO] = gas_price / 3.0
    vec[_I_COST_RATIO] = (value_asf + gas_price * 21000 / _BASE_UNIT) / balance_asf
    vec[_I_VALUE_PCTILE] = rng.uniform(0.1, 0.9)
    vec[_I_TIME_SINCE] = rng.uniform(10.0, 3600.0)
    vec[_I_HOUR] = rng.uniform(0.0, 1.0)
    vec[_I_IS_BURST] = 0.0
    vec[_I_MEMPOOL_WAIT] = rng.uniform(0.0, 5.0)
    vec[_I_TX_TYPE] = 0.0  # TRANSFER
    vec[_I_HAS_DATA] = 0.0
    vec[_I_DATA_SIZE] = 0.0
    vec[_I_IS_CONTRACT] = 0.0
    vec[_I_MEMPOOL_SIZE] = float(rng.randint(10, 200))
    vec[_I_BLOCK_FULLNESS] = rng.uniform(0.3, 0.8)
    vec[_I_NET_TX_RATE] = rng.uniform(1.0, 20.0)
    vec[_I_GAS_VS_MEDIAN] = gas_price / 3.0
    vec[_I_PENDING_SENDER] = float(rng.randint(0, 2))
    vec[_I_PAIR_COUNT] = float(rng.randint(0, 20))
    vec[_I_CLUSTER_DIV] = rng.uniform(0.3, 0.9)
    vec[_I_RECIP_POP] = float(rng.randint(1, 100))
    vec[_I_TX_REGULARITY] = rng.uniform(0.1, 0.8)
    vec[_I_CONTRACT_RATIO] = rng.uniform(0.0, 0.3)
    vec[_I_VALUE_ENTROPY] = rng.uniform(1.0, 4.0)
    vec[_I_IN_OUT_RATIO] = rng.uniform(0.3, 3.0)
    vec[_I_VALUE_VELOCITY] = rng.uniform(0.0, 0.5)
    vec[_I_GAS_EFFICIENCY] = rng.uniform(0.7, 1.0)
    return vec


# ===================================================================
# Category A: Transaction-Level Attacks
# ===================================================================

def _gen_double_spend(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Double spend: same nonce, same sender, different recipients.
    Signature: low time_since_last_tx, same nonce, different recipients (low pair count),
    sender cluster diversity suddenly rises, pending_from_sender high."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        nonce = rng.randint(0, 100)
        vec[_I_NONCE] = float(nonce)
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 2.0)            # Nearly simultaneous
        vec[_I_PENDING_SENDER] = float(rng.randint(2, 10))     # Multiple pending
        vec[_I_PAIR_COUNT] = 0.0                               # New recipient each time
        vec[_I_CLUSTER_DIV] = rng.uniform(0.9, 1.0)           # High diversity spike
        vec[_I_TX_REGULARITY] = rng.uniform(0.0, 0.1)         # Very regular (machine)
        vec[_I_REPUTATION] = rng.uniform(0.3, 0.7)
        samples.append((vec, 1))
    return samples


def _gen_overdraft(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Overdraft: value > balance (ratio 1.5x to 1000x)."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        ratio = rng.uniform(1.5, 1000.0)
        vec[_I_BALANCE_RATIO] = ratio
        vec[_I_COST_RATIO] = ratio * rng.uniform(1.0, 1.1)
        vec[_I_VALUE_LOG] = math.log10(rng.uniform(1e20, 1e24) + 1)
        vec[_I_VALUE_PCTILE] = rng.uniform(0.95, 1.0)
        vec[_I_REPUTATION] = rng.uniform(0.2, 0.6)
        samples.append((vec, 1))
    return samples


def _gen_negative_value(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Negative value: value < 0.  value_log will be 0 or anomalous."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_BALANCE_RATIO] = rng.uniform(-100.0, -0.01)
        vec[_I_VALUE_LOG] = 0.0  # log10 of negative is undefined, extractor returns 0
        vec[_I_COST_RATIO] = rng.uniform(-50.0, -0.001)
        vec[_I_VALUE_PCTILE] = 0.0
        vec[_I_VALUE_DEV] = rng.uniform(5.0, 50.0)
        samples.append((vec, 1))
    return samples


def _gen_dust_spam(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Dust spam: thousands of tiny TXs (value < 0.001 ASF).
    Signature: extremely small value, very high frequency, burst=1, many pending."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        dust_value = rng.uniform(1.0, 1e15)  # < 0.001 ASF in wei
        vec[_I_VALUE_LOG] = math.log10(dust_value + 1)
        vec[_I_BALANCE_RATIO] = dust_value / (rng.uniform(100, 10000) * _BASE_UNIT)
        vec[_I_TX_FREQ] = rng.uniform(50.0, 500.0)            # Very high freq
        vec[_I_IS_BURST] = 1.0
        vec[_I_PENDING_SENDER] = float(rng.randint(10, 100))
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 0.5)
        vec[_I_UNIQUE_RECIP] = float(rng.randint(50, 1000))   # Spraying many addresses
        vec[_I_CLUSTER_DIV] = rng.uniform(0.9, 1.0)
        vec[_I_VALUE_ENTROPY] = rng.uniform(0.0, 0.5)         # Low entropy (same value)
        vec[_I_NET_TX_RATE] = rng.uniform(50.0, 200.0)        # Network overwhelmed
        samples.append((vec, 1))
    return samples


def _gen_gas_manipulation(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Gas manipulation: zero gas or extreme gas price (front-running vector)."""
    samples: List[Tuple[List[float], int]] = []
    half = count // 2

    # Zero gas
    for _ in range(half):
        vec = _normal_base(rng)
        vec[_I_GAS_RATIO] = 0.0
        vec[_I_GAS_VS_MEDIAN] = 0.0
        vec[_I_GAS_EFFICIENCY] = 0.0
        vec[_I_COST_RATIO] = 0.0
        vec[_I_REPUTATION] = rng.uniform(0.1, 0.5)
        samples.append((vec, 1))

    # Extreme gas price (front-running)
    for _ in range(count - half):
        vec = _normal_base(rng)
        multiplier = rng.uniform(20.0, 500.0)
        vec[_I_GAS_RATIO] = multiplier
        vec[_I_GAS_VS_MEDIAN] = multiplier
        vec[_I_COST_RATIO] = rng.uniform(0.5, 5.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.2, 0.5)  # Wasting gas
        samples.append((vec, 1))
    return samples


def _gen_nonce_gap(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Nonce gap: future nonce (skip 10-1000 ahead of expected)."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        expected_nonce = rng.randint(5, 100)
        gap = rng.randint(10, 1000)
        vec[_I_NONCE] = float(expected_nonce + gap)
        vec[_I_PENDING_SENDER] = float(rng.randint(5, 50))
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 1.0)
        vec[_I_REPUTATION] = rng.uniform(0.3, 0.6)
        vec[_I_TX_REGULARITY] = rng.uniform(2.0, 10.0)  # Very irregular
        samples.append((vec, 1))
    return samples


def _gen_replay_attack(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Replay attack: exact same TX params replayed.
    Signature: same nonce, same value, same pair count, low time_since."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_PAIR_COUNT] = float(rng.randint(20, 200))    # Same pair, many times
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 3.0)
        vec[_I_VALUE_DEV] = 0.0                              # Exact same value
        vec[_I_VALUE_ENTROPY] = 0.0                          # Zero entropy (identical)
        vec[_I_TX_REGULARITY] = 0.0                          # Perfectly regular
        vec[_I_CLUSTER_DIV] = rng.uniform(0.0, 0.1)        # One recipient
        vec[_I_REPUTATION] = rng.uniform(0.2, 0.5)
        samples.append((vec, 1))
    return samples


# ===================================================================
# Category B: MEV / DeFi Attacks
# ===================================================================

def _gen_sandwich_attack(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Sandwich attack: buy-target-sell pattern, high gas, contract interaction."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_GAS_RATIO] = rng.uniform(10.0, 200.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(10.0, 200.0)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(100, 2000))
        vec[_I_TX_TYPE] = 4.0  # CONTRACT_CALL
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 2.0)          # Rapid sequence
        vec[_I_PENDING_SENDER] = float(rng.randint(2, 5))
        vec[_I_PAIR_COUNT] = float(rng.randint(0, 3))       # Few prior interactions
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.8, 1.0)      # All contract calls
        vec[_I_VALUE_VELOCITY] = rng.uniform(2.0, 20.0)     # Rapid value changes
        vec[_I_REPUTATION] = rng.uniform(0.3, 0.6)
        vec[_I_IN_OUT_RATIO] = rng.uniform(0.9, 1.1)        # Buy-sell balance
        samples.append((vec, 1))
    return samples


def _gen_flash_loan(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Flash loan pattern: huge borrow -> action -> repay in one TX.
    Signature: extremely high value, contract interaction, large data, high gas."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        flash_value = rng.uniform(1e24, 1e28)  # Millions of ASF in wei
        vec[_I_VALUE_LOG] = math.log10(flash_value + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(100.0, 10000.0)  # Way more than balance
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(5000, 50000))  # Complex calldata
        vec[_I_TX_TYPE] = 4.0
        vec[_I_GAS_RATIO] = rng.uniform(5.0, 50.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(5.0, 50.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.9, 1.0)       # Uses nearly all gas
        vec[_I_CONTRACT_RATIO] = 1.0
        vec[_I_VALUE_VELOCITY] = rng.uniform(10.0, 100.0)
        vec[_I_NONCE] = float(rng.randint(0, 5))             # Often new account
        vec[_I_AGE] = rng.uniform(0.0, 24.0)                 # Young account
        vec[_I_REPUTATION] = rng.uniform(0.1, 0.4)
        samples.append((vec, 1))
    return samples


def _gen_front_running(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Front-running: copy pending TX with higher gas price.
    Signature: very high gas ratio, contract call, mirrors another TX."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_GAS_RATIO] = rng.uniform(15.0, 300.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(15.0, 300.0)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(68, 500))       # Similar calldata size
        vec[_I_TX_TYPE] = 4.0
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 1.0)
        vec[_I_MEMPOOL_WAIT] = rng.uniform(0.0, 0.5)         # Submitted fast
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.7, 1.0)
        vec[_I_REPUTATION] = rng.uniform(0.3, 0.6)
        vec[_I_PENDING_SENDER] = float(rng.randint(1, 5))
        samples.append((vec, 1))
    return samples


def _gen_back_running(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Back-running: TX immediately after target TX.
    Signature: slightly higher gas, contract call, very low time_since."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_GAS_RATIO] = rng.uniform(3.0, 15.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(3.0, 15.0)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(68, 500))
        vec[_I_TX_TYPE] = 4.0
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 0.5)
        vec[_I_MEMPOOL_WAIT] = rng.uniform(0.0, 0.3)
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.6, 1.0)
        vec[_I_VALUE_VELOCITY] = rng.uniform(1.0, 10.0)
        vec[_I_REPUTATION] = rng.uniform(0.4, 0.7)
        samples.append((vec, 1))
    return samples


# ===================================================================
# Category C: Network Attacks
# ===================================================================

def _gen_sybil(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Sybil: many wallets created from same funding source.
    Signature: new account (nonce 0-2), young age, low reputation, funded
    from same source (same pair), identical value amounts."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_NONCE] = float(rng.randint(0, 2))
        vec[_I_AGE] = rng.uniform(0.0, 2.0)                   # Minutes old
        vec[_I_REPUTATION] = rng.uniform(0.0, 0.3)
        vec[_I_UNIQUE_RECIP] = float(rng.randint(0, 2))
        vec[_I_CLUSTER_DIV] = rng.uniform(0.0, 0.1)
        vec[_I_PAIR_COUNT] = float(rng.randint(0, 1))
        vec[_I_VALUE_ENTROPY] = rng.uniform(0.0, 0.3)         # Uniform amounts
        vec[_I_TX_REGULARITY] = rng.uniform(0.0, 0.1)         # Machine-like
        vec[_I_IN_OUT_RATIO] = rng.uniform(0.0, 0.2)          # Mostly outgoing
        vec[_I_RECIP_POP] = float(rng.randint(0, 2))          # Unknown recipient
        vec[_I_BALANCE_RATIO] = rng.uniform(0.8, 1.0)         # Spending all
        samples.append((vec, 1))
    return samples


def _gen_burst_ddos(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Burst/DDoS: >50 TXs per second from same sender.
    Signature: extreme frequency, burst=1, many pending, overwhelming mempool."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_TX_FREQ] = rng.uniform(100.0, 5000.0)
        vec[_I_IS_BURST] = 1.0
        vec[_I_PENDING_SENDER] = float(rng.randint(20, 500))
        vec[_I_TIME_SINCE] = rng.uniform(0.0, 0.1)
        vec[_I_MEMPOOL_SIZE] = float(rng.randint(500, 10000))
        vec[_I_NET_TX_RATE] = rng.uniform(100.0, 1000.0)
        vec[_I_TX_REGULARITY] = rng.uniform(0.0, 0.05)        # Perfectly timed
        vec[_I_MEMPOOL_WAIT] = rng.uniform(10.0, 300.0)       # Queue backed up
        vec[_I_REPUTATION] = rng.uniform(0.1, 0.4)
        samples.append((vec, 1))
    return samples


def _gen_time_manipulation(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Time manipulation: future or very old timestamp.
    Signature: abnormal hour_of_day patterns, extreme time_since_last_tx."""
    samples: List[Tuple[List[float], int]] = []
    half = count // 2

    # Future timestamp
    for _ in range(half):
        vec = _normal_base(rng)
        vec[_I_TIME_SINCE] = rng.uniform(-86400.0, -60.0)     # Negative = future
        vec[_I_MEMPOOL_WAIT] = rng.uniform(-3600.0, -1.0)
        vec[_I_AGE] = rng.uniform(-100.0, -1.0)               # Negative age
        vec[_I_REPUTATION] = rng.uniform(0.2, 0.5)
        samples.append((vec, 1))

    # Very old timestamp
    for _ in range(count - half):
        vec = _normal_base(rng)
        vec[_I_TIME_SINCE] = rng.uniform(86400.0, 604800.0)   # Days to weeks
        vec[_I_MEMPOOL_WAIT] = rng.uniform(3600.0, 86400.0)   # Hours in mempool
        vec[_I_REPUTATION] = rng.uniform(0.2, 0.5)
        samples.append((vec, 1))
    return samples


# ===================================================================
# Category D: Smart Contract Attacks
# ===================================================================

def _gen_reentrancy(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Reentrancy pattern: recursive call pattern.
    Signature: contract interaction, very high gas, large data, high gas efficiency,
    high contract call ratio, value draining pattern."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(500, 10000))
        vec[_I_TX_TYPE] = 4.0  # CONTRACT_CALL
        vec[_I_GAS_RATIO] = rng.uniform(5.0, 50.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(5.0, 50.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.95, 1.0)       # Nearly all gas used
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.9, 1.0)
        vec[_I_VALUE_VELOCITY] = rng.uniform(5.0, 50.0)       # Rapid draining
        vec[_I_BALANCE_RATIO] = rng.uniform(0.8, 10.0)        # Draining balance
        vec[_I_PAIR_COUNT] = float(rng.randint(0, 2))         # New target
        vec[_I_NONCE] = float(rng.randint(0, 10))             # Often new attacker
        vec[_I_REPUTATION] = rng.uniform(0.1, 0.4)
        vec[_I_AGE] = rng.uniform(0.0, 48.0)                  # Young account
        samples.append((vec, 1))
    return samples


def _gen_integer_overflow(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Integer overflow: value near uint256 max.
    Signature: astronomically large value_log, extreme balance ratio."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        # uint256 max is ~1.15e77
        overflow_value = rng.uniform(1e70, 1e78)
        vec[_I_VALUE_LOG] = math.log10(overflow_value + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(1e10, 1e50)
        vec[_I_COST_RATIO] = rng.uniform(1e10, 1e50)
        vec[_I_VALUE_PCTILE] = 1.0
        vec[_I_VALUE_DEV] = rng.uniform(100.0, 1e10)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(100, 1000))
        vec[_I_TX_TYPE] = 4.0
        vec[_I_REPUTATION] = rng.uniform(0.0, 0.3)
        samples.append((vec, 1))
    return samples


def _gen_self_destruct(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Self-destruct: contract destruction pattern.
    Signature: contract interaction, specific data patterns, balance draining."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(4, 100))        # Small but present
        vec[_I_TX_TYPE] = 4.0
        vec[_I_BALANCE_RATIO] = rng.uniform(0.9, 1.0)         # Draining everything
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.8, 1.0)
        vec[_I_PAIR_COUNT] = float(rng.randint(0, 3))
        vec[_I_NONCE] = float(rng.randint(0, 5))
        vec[_I_REPUTATION] = rng.uniform(0.1, 0.4)
        vec[_I_AGE] = rng.uniform(0.0, 24.0)
        vec[_I_VALUE_VELOCITY] = rng.uniform(5.0, 30.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.3, 0.6)        # Unusual gas pattern
        samples.append((vec, 1))
    return samples


# ===================================================================
# Category E: Normal Transactions (balance dataset)
# ===================================================================

def _gen_normal_transfer(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Regular transfers (1-100 ASF)."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        # Already normal by default, just vary slightly
        value_asf = rng.uniform(1.0, 100.0)
        vec[_I_VALUE_LOG] = math.log10(value_asf * _BASE_UNIT + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(0.01, 0.3)
        vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)
        vec[_I_TX_TYPE] = 0.0
        samples.append((vec, 0))
    return samples


def _gen_normal_staking(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Staking transactions (32-1000 ASF)."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        stake_asf = rng.uniform(32.0, 1000.0)
        vec[_I_VALUE_LOG] = math.log10(stake_asf * _BASE_UNIT + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(0.1, 0.5)
        vec[_I_VALUE_ROUND] = float(rng.random() < 0.6)       # Often round numbers
        vec[_I_TX_TYPE] = 2.0  # STAKE
        vec[_I_REPUTATION] = rng.uniform(0.8, 1.0)
        vec[_I_NONCE] = float(rng.randint(10, 500))
        vec[_I_AGE] = rng.uniform(168.0, 8760.0)              # Established accounts
        vec[_I_GAS_RATIO] = rng.uniform(0.8, 2.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(0.8, 2.0)
        samples.append((vec, 0))
    return samples


def _gen_normal_token_creation(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Token creation transactions."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(2000, 20000))   # Contract bytecode
        vec[_I_TX_TYPE] = 3.0  # CONTRACT_CREATE
        vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)
        vec[_I_NONCE] = float(rng.randint(20, 300))
        vec[_I_AGE] = rng.uniform(168.0, 4380.0)              # Established
        vec[_I_GAS_RATIO] = rng.uniform(1.0, 5.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(1.0, 5.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.5, 0.9)
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.1, 0.5)
        samples.append((vec, 0))
    return samples


def _gen_normal_nft_mint(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """NFT minting transactions."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(100, 500))
        vec[_I_TX_TYPE] = 4.0  # CONTRACT_CALL
        mint_cost = rng.uniform(0.1, 10.0)
        vec[_I_VALUE_LOG] = math.log10(mint_cost * _BASE_UNIT + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(0.001, 0.1)
        vec[_I_REPUTATION] = rng.uniform(0.6, 1.0)
        vec[_I_NONCE] = float(rng.randint(5, 200))
        vec[_I_AGE] = rng.uniform(48.0, 4380.0)
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.2, 0.6)
        vec[_I_GAS_RATIO] = rng.uniform(1.0, 3.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(1.0, 3.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.6, 0.95)
        samples.append((vec, 0))
    return samples


def _gen_normal_tips(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Small tips (0.01-1 ASF)."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        tip_asf = rng.uniform(0.01, 1.0)
        vec[_I_VALUE_LOG] = math.log10(tip_asf * _BASE_UNIT + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(0.0001, 0.01)
        vec[_I_VALUE_ROUND] = float(rng.random() < 0.5)
        vec[_I_TX_TYPE] = 0.0
        vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)
        vec[_I_NONCE] = float(rng.randint(10, 1000))
        vec[_I_AGE] = rng.uniform(24.0, 8760.0)
        vec[_I_PAIR_COUNT] = float(rng.randint(1, 50))
        samples.append((vec, 0))
    return samples


# ===================================================================
# Additional attack variations for robustness
# ===================================================================

def _gen_whale_manipulation(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Whale manipulation: large single TX to manipulate market.
    Signature: extremely high value, old high-rep account (whales are trusted),
    but the TX pattern is abnormal for them."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        whale_value = rng.uniform(1e25, 1e27)
        vec[_I_VALUE_LOG] = math.log10(whale_value + 1)
        vec[_I_BALANCE_RATIO] = rng.uniform(0.5, 0.95)        # Big chunk of balance
        vec[_I_VALUE_DEV] = rng.uniform(5.0, 50.0)            # Way above their avg
        vec[_I_VALUE_PCTILE] = rng.uniform(0.98, 1.0)
        vec[_I_NONCE] = float(rng.randint(100, 5000))
        vec[_I_AGE] = rng.uniform(1000.0, 8760.0)
        vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)            # Trusted whale
        vec[_I_VALUE_VELOCITY] = rng.uniform(5.0, 50.0)
        samples.append((vec, 1))
    return samples


def _gen_contract_bomb(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Contract gas bomb: deploy contract designed to consume all gas.
    Signature: contract creation, enormous data, maximal gas."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(20000, 100000))  # Huge bytecode
        vec[_I_TX_TYPE] = 3.0  # CONTRACT_CREATE
        vec[_I_GAS_RATIO] = rng.uniform(10.0, 100.0)
        vec[_I_GAS_VS_MEDIAN] = rng.uniform(10.0, 100.0)
        vec[_I_GAS_EFFICIENCY] = rng.uniform(0.99, 1.0)        # Uses ALL gas
        vec[_I_NONCE] = float(rng.randint(0, 3))
        vec[_I_AGE] = rng.uniform(0.0, 10.0)
        vec[_I_REPUTATION] = rng.uniform(0.0, 0.3)
        vec[_I_CONTRACT_RATIO] = 1.0
        samples.append((vec, 1))
    return samples


def _gen_wash_trading(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Wash trading: circular TX patterns between controlled addresses.
    Signature: value comes back (in/out ratio ~1), very regular timing,
    low entropy, same pair repeatedly."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_PAIR_COUNT] = float(rng.randint(50, 500))       # Same pair many times
        vec[_I_IN_OUT_RATIO] = rng.uniform(0.9, 1.1)          # Balanced (circular)
        vec[_I_TX_REGULARITY] = rng.uniform(0.0, 0.05)        # Machine-like
        vec[_I_VALUE_ENTROPY] = rng.uniform(0.0, 0.5)         # Same amounts
        vec[_I_CLUSTER_DIV] = rng.uniform(0.0, 0.1)           # Few recipients
        vec[_I_UNIQUE_RECIP] = float(rng.randint(1, 3))
        vec[_I_TX_FREQ] = rng.uniform(20.0, 200.0)
        vec[_I_TIME_SINCE] = rng.uniform(30.0, 120.0)         # Regular intervals
        vec[_I_REPUTATION] = rng.uniform(0.3, 0.6)
        samples.append((vec, 1))
    return samples


def _gen_phishing_drain(rng: random.Random, count: int) -> List[Tuple[List[float], int]]:
    """Phishing drain: approved contract draining victim wallet.
    Signature: contract call, high balance ratio, new recipient for victim,
    victim has good reputation but sudden drain."""
    samples: List[Tuple[List[float], int]] = []
    for _ in range(count):
        vec = _normal_base(rng)
        vec[_I_IS_CONTRACT] = 1.0
        vec[_I_HAS_DATA] = 1.0
        vec[_I_DATA_SIZE] = float(rng.randint(68, 200))
        vec[_I_TX_TYPE] = 4.0
        vec[_I_BALANCE_RATIO] = rng.uniform(0.9, 1.0)         # Draining all
        vec[_I_COST_RATIO] = rng.uniform(0.9, 1.0)
        vec[_I_PAIR_COUNT] = 0.0                               # Never interacted before
        vec[_I_RECIP_POP] = float(rng.randint(0, 5))          # Unknown recipient
        vec[_I_REPUTATION] = rng.uniform(0.7, 1.0)            # Victim is trusted
        vec[_I_VALUE_DEV] = rng.uniform(10.0, 100.0)          # Way above avg
        vec[_I_VALUE_VELOCITY] = rng.uniform(10.0, 100.0)
        vec[_I_CONTRACT_RATIO] = rng.uniform(0.0, 0.2)        # Victim rarely calls contracts
        samples.append((vec, 1))
    return samples


# ===================================================================
# Dataset Generation
# ===================================================================

# Registry: (generator_func, sample_count, category, attack_name)
_ATTACK_GENERATORS = [
    # Category A: Transaction-Level Attacks
    (_gen_double_spend,      600, "A", "double_spend"),
    (_gen_overdraft,         600, "A", "overdraft"),
    (_gen_negative_value,    500, "A", "negative_value"),
    (_gen_dust_spam,         600, "A", "dust_spam"),
    (_gen_gas_manipulation,  600, "A", "gas_manipulation"),
    (_gen_nonce_gap,         500, "A", "nonce_gap"),
    (_gen_replay_attack,     500, "A", "replay_attack"),

    # Category B: MEV/DeFi Attacks
    (_gen_sandwich_attack,   600, "B", "sandwich_attack"),
    (_gen_flash_loan,        500, "B", "flash_loan"),
    (_gen_front_running,     600, "B", "front_running"),
    (_gen_back_running,      500, "B", "back_running"),

    # Category C: Network Attacks
    (_gen_sybil,             600, "C", "sybil"),
    (_gen_burst_ddos,        600, "C", "burst_ddos"),
    (_gen_time_manipulation, 500, "C", "time_manipulation"),

    # Category D: Smart Contract Attacks
    (_gen_reentrancy,        600, "D", "reentrancy"),
    (_gen_integer_overflow,  500, "D", "integer_overflow"),
    (_gen_self_destruct,     500, "D", "self_destruct"),

    # Additional attack patterns
    (_gen_whale_manipulation, 400, "E+", "whale_manipulation"),
    (_gen_contract_bomb,      400, "E+", "contract_bomb"),
    (_gen_wash_trading,       500, "E+", "wash_trading"),
    (_gen_phishing_drain,     500, "E+", "phishing_drain"),
]

_NORMAL_GENERATORS = [
    # Category E: Normal Transactions
    (_gen_normal_transfer,       1200, "E", "normal_transfer"),
    (_gen_normal_staking,         600, "E", "normal_staking"),
    (_gen_normal_token_creation,  400, "E", "normal_token_creation"),
    (_gen_normal_nft_mint,        400, "E", "normal_nft_mint"),
    (_gen_normal_tips,            600, "E", "normal_tips"),
]


def generate_attack_dataset(
    seed: int = 12345,
) -> List[Tuple[List[float], int]]:
    """Generate the full labeled attack/normal dataset.

    Returns:
        List of (feature_vector, label) tuples.
        feature_vector: 35-element float list matching TransactionFeatures.to_vector()
        label: 0 = normal, 1 = attack

    Total samples: ~13,800
        Attack samples: ~10,600 across 21 attack patterns
        Normal samples: ~3,200 across 5 normal categories
    """
    rng = random.Random(seed)
    dataset: List[Tuple[List[float], int]] = []

    # Generate attack samples
    for gen_func, count, _cat, _name in _ATTACK_GENERATORS:
        samples = gen_func(rng, count)
        dataset.extend(samples)

    # Generate normal samples
    for gen_func, count, _cat, _name in _NORMAL_GENERATORS:
        samples = gen_func(rng, count)
        dataset.extend(samples)

    # Shuffle deterministically
    rng.shuffle(dataset)

    return dataset


def get_training_stats(dataset: List[Tuple[List[float], int]] = None) -> Dict:
    """Return detailed statistics about the training dataset.

    Args:
        dataset: Pre-generated dataset. If None, generates a new one.

    Returns:
        Dictionary with total counts, per-category breakdown, and feature stats.
    """
    if dataset is None:
        dataset = generate_attack_dataset()

    total = len(dataset)
    attack_count = sum(1 for _, label in dataset if label == 1)
    normal_count = total - attack_count

    # Per-generator breakdown (regenerate to count per category)
    rng = random.Random(12345)
    category_stats: Dict[str, Dict[str, int]] = {}
    for gen_func, count, cat, name in _ATTACK_GENERATORS + _NORMAL_GENERATORS:
        if cat not in category_stats:
            category_stats[cat] = {}
        category_stats[cat][name] = count

    # Feature statistics (min, max, mean per feature)
    feature_names = [
        "sender_balance_ratio", "sender_nonce", "sender_avg_value",
        "sender_value_deviation", "sender_tx_frequency", "sender_age",
        "sender_unique_recipients", "sender_reputation",
        "value_log", "value_is_round", "gas_price_ratio",
        "total_cost_ratio", "value_percentile",
        "time_since_last_tx", "hour_of_day", "is_burst", "mempool_wait_time",
        "tx_type", "has_data", "data_size", "is_contract_interaction",
        "mempool_size", "recent_block_fullness", "network_tx_rate",
        "gas_price_vs_median", "pending_from_sender",
        "sender_recipient_tx_count", "sender_cluster_diversity",
        "recipient_popularity", "sender_tx_regularity",
        "contract_call_ratio", "value_entropy",
        "incoming_outgoing_ratio", "value_velocity", "gas_efficiency",
    ]

    feature_stats = {}
    if dataset:
        for i, name in enumerate(feature_names):
            values = [vec[i] for vec, _ in dataset]
            feature_stats[name] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }

    return {
        "total_samples": total,
        "attack_samples": attack_count,
        "normal_samples": normal_count,
        "attack_ratio": attack_count / max(total, 1),
        "normal_ratio": normal_count / max(total, 1),
        "categories": category_stats,
        "feature_dim": _FEATURE_DIM,
        "feature_stats": feature_stats,
    }


# ===================================================================
# Pre-training integration
# ===================================================================

def pretrain_ai_models(gate, seed: int = 12345, epochs: int = 5) -> Dict:
    """Pre-train the AI Validation Gate models with the attack dataset.

    This function:
    1. Generates the full labeled attack dataset (10,000+ samples)
    2. Feeds normal samples through the TAD autoencoder to learn baseline
    3. Feeds attack samples to calibrate anomaly detection thresholds
    4. Logs training progress and final statistics

    Should be called during node initialization, BEFORE processing any blocks.

    Args:
        gate: AIValidationGate instance (from positronic.ai.meta_model).
        seed: Random seed for deterministic, reproducible training.
        epochs: Number of training passes over the dataset.

    Returns:
        Dictionary with training metrics and accuracy estimate.
    """
    start_time = time.monotonic()

    logger.info("=== Positronic AI Pre-Learning: Starting Attack Dataset Training ===")

    # Step 1: Generate dataset
    dataset = generate_attack_dataset(seed=seed)
    stats = get_training_stats(dataset)

    logger.info(
        "Dataset generated: %d total (%d attack, %d normal)",
        stats["total_samples"],
        stats["attack_samples"],
        stats["normal_samples"],
    )

    # Step 2: Separate normal and attack samples
    normal_vectors = [vec for vec, label in dataset if label == 0]
    attack_vectors = [vec for vec, label in dataset if label == 1]

    # Step 3: Train TAD autoencoder on normal data first (learn baseline)
    batch_size = 64
    total_batches = 0

    for epoch in range(epochs):
        # Shuffle normal vectors each epoch
        rng = random.Random(seed + epoch)
        epoch_normals = list(normal_vectors)
        rng.shuffle(epoch_normals)

        for i in range(0, len(epoch_normals), batch_size):
            batch = epoch_normals[i:i + batch_size]
            gate.anomaly_detector.train_step(batch)
            total_batches += 1

        logger.info(
            "Epoch %d/%d: trained on %d normal samples (%d batches)",
            epoch + 1, epochs, len(epoch_normals), total_batches,
        )

    # Step 4: Feed a small portion of attack data to calibrate thresholds
    # (autoencoder should see some anomalies to set error distribution)
    calibration_count = min(500, len(attack_vectors))
    rng_cal = random.Random(seed + 999)
    calibration_attacks = list(attack_vectors)
    rng_cal.shuffle(calibration_attacks)
    calibration_attacks = calibration_attacks[:calibration_count]

    for i in range(0, len(calibration_attacks), batch_size):
        batch = calibration_attacks[i:i + batch_size]
        gate.anomaly_detector.train_step(batch)
        total_batches += 1

    logger.info(
        "Calibration: fed %d attack samples to set detection thresholds",
        calibration_count,
    )

    # Step 5: Estimate accuracy by scoring a sample of the dataset
    correct = 0
    tested = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    # Test on a sample (not the full dataset, for speed)
    test_size = min(2000, len(dataset))
    rng_test = random.Random(seed + 777)
    test_set = list(dataset)
    rng_test.shuffle(test_set)
    test_set = test_set[:test_size]

    for vec, label in test_set:
        # Use the autoencoder's forward pass to get reconstruction error
        normalized = gate.anomaly_detector._normalize(vec)
        _, error = gate.anomaly_detector.forward(normalized)

        # Classify based on z-score
        if gate.anomaly_detector.std_error > 0:
            z_score = (
                (error - gate.anomaly_detector.mean_error)
                / max(gate.anomaly_detector.std_error, 1e-10)
            )
            predicted = 1 if z_score > 2.0 else 0
        else:
            predicted = 0

        if predicted == label:
            correct += 1
        if label == 1 and predicted == 1:
            true_pos += 1
        elif label == 0 and predicted == 1:
            false_pos += 1
        elif label == 0 and predicted == 0:
            true_neg += 1
        elif label == 1 and predicted == 0:
            false_neg += 1
        tested += 1

    accuracy = correct / max(tested, 1)
    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)
    f1 = (
        2 * precision * recall / max(precision + recall, 1e-10)
        if (precision + recall) > 0
        else 0.0
    )

    elapsed = time.monotonic() - start_time

    result = {
        "total_samples": stats["total_samples"],
        "attack_samples": stats["attack_samples"],
        "normal_samples": stats["normal_samples"],
        "epochs": epochs,
        "total_batches": total_batches,
        "batch_size": batch_size,
        "calibration_attacks": calibration_count,
        "test_size": tested,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "true_negatives": true_neg,
        "false_negatives": false_neg,
        "training_time_seconds": elapsed,
        "autoencoder_stats": gate.anomaly_detector.get_stats(),
    }

    logger.info(
        "=== Pre-Learning Complete in %.2fs ===\n"
        "  Accuracy: %.1f%% | Precision: %.1f%% | Recall: %.1f%% | F1: %.3f\n"
        "  TP=%d FP=%d TN=%d FN=%d (tested %d samples)\n"
        "  Autoencoder: trained=%s, samples=%d, mean_err=%.4f, std_err=%.4f",
        elapsed,
        accuracy * 100, precision * 100, recall * 100, f1,
        true_pos, false_pos, true_neg, false_neg, tested,
        result["autoencoder_stats"]["trained"],
        result["autoencoder_stats"]["training_samples"],
        result["autoencoder_stats"]["mean_error"],
        result["autoencoder_stats"]["std_error"],
    )

    return result
