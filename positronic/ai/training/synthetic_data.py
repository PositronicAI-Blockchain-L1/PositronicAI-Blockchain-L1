"""
Generate synthetic normal transaction feature vectors for TAD bootstrap training.

Creates Transaction + Account objects with safe parameters, extracts features
via FeatureExtractor (building up rolling statistics), and returns 35-dim
feature vectors suitable for anomaly_detector.train_step().
"""
import random
from typing import List

from positronic.constants import BASE_UNIT
from positronic.core.transaction import Transaction, TxType
from positronic.core.account import Account
from positronic.ai.feature_extractor import FeatureExtractor


def generate_normal_features(count: int = 500, seed: int = 42) -> List[List[float]]:
    """Generate synthetic normal transaction feature vectors for TAD training.

    Parameters mirror the safe test cases in test_ai_labeled_set.py:
    - Well-funded senders (balance >> value)
    - Normal gas prices (1-5)
    - Mixed nonce levels (0-200)
    - No contract deployments or suspicious data payloads

    Uses FeatureExtractor with update_stats() to build realistic rolling
    statistics across samples.

    Returns:
        List of 35-element float lists (one per synthetic transaction).
    """
    rng = random.Random(seed)
    extractor = FeatureExtractor()
    features: List[List[float]] = []

    for i in range(count):
        tx_type = TxType.STAKE if rng.random() < 0.15 else TxType.TRANSFER
        nonce = rng.randint(0, 200)
        balance = rng.randint(100, 100000)
        value_ratio = rng.uniform(0.001, 0.5)
        value = max(1, int(balance * value_ratio))
        gas_price = rng.randint(1, 5)
        gas_limit = 50000 if tx_type == TxType.STAKE else 21000

        sender_key = (i + 100).to_bytes(32, "big")
        recipient = (i + 200).to_bytes(20, "big")

        tx = Transaction(
            tx_type=tx_type, nonce=nonce, sender=sender_key,
            recipient=recipient, value=value * BASE_UNIT,
            gas_price=gas_price, gas_limit=gas_limit,
        )

        acct = Account(address=(i + 100).to_bytes(20, "big"))
        acct.balance = balance * BASE_UNIT
        acct.nonce = nonce

        tf = extractor.extract(tx, acct, mempool_size=rng.randint(10, 200))
        features.append(tf.to_vector())

        # Build up rolling statistics for subsequent extractions
        extractor.update_stats(tx)

    return features
