"""
Generate synthetic transaction data for AI model pre-training.

Creates a mix of:
- Normal transactions (transfers, contract calls)
- Suspicious transactions (high value to unknown addresses, rapid sequences)
- Malicious patterns (known attack signatures, sandwich attempts)

These are used to pre-train the AI models so they have reasonable
baseline weights instead of starting from random initialization.
"""
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict


def generate_normal_transactions(count: int = 5000) -> List[Dict]:
    """Generate synthetic normal transaction features."""
    transactions = []
    for i in range(count):
        tx = {
            "value_normalized": random.uniform(0.0, 0.3),  # Low-medium value
            "gas_ratio": random.uniform(0.8, 1.2),  # Normal gas
            "nonce_gap": 0,  # Sequential nonces
            "sender_age_blocks": random.randint(100, 100000),  # Established
            "recipient_known": 1,  # Known recipient
            "contract_call": random.choice([0, 1]),
            "data_size": random.randint(0, 500),
            "time_since_last_tx": random.uniform(10, 3600),
            "label": 0,  # Normal
        }
        transactions.append(tx)
    return transactions


def generate_suspicious_transactions(count: int = 2000) -> List[Dict]:
    """Generate synthetic suspicious transaction features."""
    transactions = []
    for i in range(count):
        tx = {
            "value_normalized": random.uniform(0.5, 0.9),  # Higher value
            "gas_ratio": random.uniform(1.5, 3.0),  # Elevated gas
            "nonce_gap": random.randint(0, 2),  # Some gaps
            "sender_age_blocks": random.randint(1, 50),  # New account
            "recipient_known": random.choice([0, 1]),
            "contract_call": 1,
            "data_size": random.randint(500, 5000),
            "time_since_last_tx": random.uniform(0.1, 5),  # Rapid
            "label": 1,  # Suspicious
        }
        transactions.append(tx)
    return transactions


def generate_malicious_transactions(count: int = 1000) -> List[Dict]:
    """Generate synthetic malicious transaction features."""
    transactions = []
    for i in range(count):
        tx = {
            "value_normalized": random.uniform(0.8, 1.0),  # Very high value
            "gas_ratio": random.uniform(3.0, 10.0),  # Very high gas
            "nonce_gap": random.randint(3, 50),  # Large gaps
            "sender_age_blocks": random.randint(0, 5),  # Brand new
            "recipient_known": 0,  # Unknown
            "contract_call": 1,
            "data_size": random.randint(5000, 50000),  # Large payload
            "time_since_last_tx": random.uniform(0, 0.5),  # Burst
            "label": 2,  # Malicious
        }
        transactions.append(tx)
    return transactions


def generate_dataset(seed: int = 42) -> Dict:
    """Generate complete synthetic training dataset."""
    random.seed(seed)
    normal = generate_normal_transactions(5000)
    suspicious = generate_suspicious_transactions(2000)
    malicious = generate_malicious_transactions(1000)

    all_data = normal + suspicious + malicious
    random.shuffle(all_data)

    return {
        "version": 1,
        "total_samples": len(all_data),
        "class_distribution": {
            "normal": len(normal),
            "suspicious": len(suspicious),
            "malicious": len(malicious),
        },
        "samples": all_data,
    }


def generate_and_save(output_dir: str = None):
    """Generate dataset and save to JSON."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "data")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()

    output_path = Path(output_dir) / "synthetic_training_data.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    print(f"Generated {dataset['total_samples']} samples -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_and_save()
