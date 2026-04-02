"""
Bootstrap training for the AI Validation Gate.

Trains the TAD autoencoder with synthetic normal transaction data so it can
detect anomalous patterns from genesis — before any real transaction history
is available.
"""
from positronic.ai.training.synthetic_data import generate_normal_features


def bootstrap_gate(gate, seed: int = 42, count: int = 500) -> None:
    """Bootstrap-train the AI validation gate with synthetic normal data.

    After training, the TAD autoencoder has baseline reconstruction error
    statistics. Anomalous transactions produce high reconstruction errors
    relative to the learned distribution, triggering the model veto mechanism
    (any model > 0.90 → final score floored at 0.85 → QUARANTINED).

    Args:
        gate: AIValidationGate instance.
        seed: Random seed for deterministic training data.
        count: Number of synthetic normal samples (≥100 for trained=True).
    """
    features = generate_normal_features(count=count, seed=seed)

    batch_size = 50
    for i in range(0, len(features), batch_size):
        batch = features[i:i + batch_size]
        gate.anomaly_detector.train_step(batch)
