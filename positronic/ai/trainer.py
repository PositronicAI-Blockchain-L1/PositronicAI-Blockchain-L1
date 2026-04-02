"""
Positronic - AI Model Trainer
Online learning system that continuously improves AI models
using confirmed transactions as training data.
"""

import time
from typing import List, Dict
from dataclasses import dataclass, field

from positronic.utils.logging import get_logger
from positronic.core.transaction import Transaction
from positronic.ai.feature_extractor import FeatureExtractor
from positronic.ai.anomaly_detector import Autoencoder

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""
    epoch: int
    samples_processed: int
    mean_loss: float
    training_time: float
    timestamp: float


class AITrainer:
    """
    Manages continuous training of AI models.

    Training strategy:
    - Collects confirmed transactions as positive examples
    - Collects known-fraud patterns as negative examples
    - Trains in mini-batches during low-activity periods
    - Validates on holdout set before deploying new model
    """

    def __init__(
        self,
        anomaly_detector: Autoencoder,
        feature_extractor: FeatureExtractor,
    ):
        self.anomaly_detector = anomaly_detector
        self.feature_extractor = feature_extractor

        # Training buffer
        self.training_buffer: List[List[float]] = []
        self.buffer_limit: int = 10000

        # Training state
        self.current_epoch: int = 0
        self.total_samples: int = 0
        self.metrics_history: List[TrainingMetrics] = []

        # Training config
        self.batch_size: int = 64
        self.learning_rate: float = 0.001
        self.min_samples_for_training: int = 100
        self.training_interval: int = 1000  # Train every N transactions

        # NEW: Neural training pipeline
        try:
            from positronic.ai.training.pretraining import ContrastivePretrainer
            from positronic.ai.training.online_learner import OnlineLearner
            # Create pretrainer if anomaly detector has a VAE
            if hasattr(anomaly_detector, '_vae') and anomaly_detector._vae is not None:
                self._pretrainer = ContrastivePretrainer(anomaly_detector._vae)
            else:
                self._pretrainer = None
            self._online_learner = OnlineLearner(
                vae=getattr(anomaly_detector, '_vae', None)
            )
            self._neural_available = True
        except ImportError:
            self._pretrainer = None
            self._online_learner = None
            self._neural_available = False
        self._neural_errors: int = 0

    def add_training_data(self, transactions: List[Transaction]):
        """Add confirmed transactions to training buffer."""
        for tx in transactions:
            features = self.feature_extractor.extract(tx)
            self.training_buffer.append(features.to_vector())

            if len(self.training_buffer) > self.buffer_limit:
                self.training_buffer = self.training_buffer[-self.buffer_limit:]

        self.total_samples += len(transactions)

        # NEW: Feed neural training pipeline
        if self._neural_available:
            try:
                import numpy as np
                vectors = [self.feature_extractor.extract(tx).to_vector() for tx in transactions]
                fv_arrays = [np.array(v, dtype=np.float32) for v in vectors]

                if self._pretrainer and not self._pretrainer.is_complete:
                    for fv in fv_arrays:
                        self._pretrainer.add_sample(fv)

                if self._online_learner:
                    self._online_learner.add_confirmed_transactions(fv_arrays)
            except Exception as e:
                logger.debug("Neural training data pipeline error: %s", e)
                self._neural_errors += 1  # Neural fallback: add_training_data neural pipeline

    def should_train(self) -> bool:
        """Check if we have enough data for a training step."""
        return (
            len(self.training_buffer) >= self.min_samples_for_training
            and self.total_samples % self.training_interval < self.batch_size
        )

    def train_step(self) -> TrainingMetrics:
        """Execute one training step."""
        start_time = time.time()

        # Select batch
        batch = self.training_buffer[-self.batch_size:]

        # Train anomaly detector
        self.anomaly_detector.train_step(batch, self.learning_rate)

        self.current_epoch += 1
        elapsed = time.time() - start_time

        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            samples_processed=len(batch),
            mean_loss=self.anomaly_detector.mean_error,
            training_time=elapsed,
            timestamp=time.time(),
        )

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        # NEW: Trigger neural training
        if self._neural_available:
            try:
                # Pre-training phase
                if self._pretrainer and self._pretrainer.should_pretrain():
                    self._pretrainer.pretrain()

                # Online learning phase
                if self._online_learner and self._online_learner.should_train():
                    self._online_learner.train_step()
            except Exception as e:
                logger.debug("Neural train_step error: %s", e)
                self._neural_errors += 1  # Neural fallback: train_step neural pipeline

        return metrics

    def pretrain_if_ready(self):
        """Trigger contrastive pre-training if enough samples have been collected."""
        if self._neural_available and self._pretrainer and self._pretrainer.should_pretrain():
            self._pretrainer.pretrain()

    def get_stats(self) -> dict:
        stats = {
            "current_epoch": self.current_epoch,
            "total_samples": self.total_samples,
            "buffer_size": len(self.training_buffer),
            "model_trained": self.anomaly_detector.trained,
            "mean_loss": self.anomaly_detector.mean_error,
        }
        if self._neural_available:
            stats["neural_training"] = {
                "pretraining_complete": self._pretrainer.is_complete if self._pretrainer else False,
                "online_learner_step": self._online_learner._step if self._online_learner else 0,
            }
        stats["neural_errors"] = self._neural_errors
        return stats
