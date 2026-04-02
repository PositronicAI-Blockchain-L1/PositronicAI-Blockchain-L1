"""
Positronic - Self-Supervised Pre-Training

Contrastive pre-training that learns normal transaction patterns without labels.
Used at genesis before any labeled fraud data is available.

Strategy:
    1. Collect first N transactions (normal by assumption at genesis)
    2. Create augmented views (mask features + add noise)
    3. Train encoder using NT-Xent contrastive loss
    4. After pre-training, encoder understands "normal" transaction manifold
    5. Switch to supervised/reconstruction fine-tuning

The pre-training phase is critical because at genesis there are zero confirmed
fraud examples. By learning the structure of normal transactions through
self-supervised contrastive learning, the model can immediately detect
out-of-distribution (potentially fraudulent) transactions once the chain
begins processing real traffic.

Dependencies:
    - VAE from positronic.ai.models.vae
    - ContrastiveLearner from positronic.ai.models.contrastive
    - Adam from positronic.ai.engine.optimizers
"""

import numpy as np
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class PretrainingConfig:
    """Configuration for the self-supervised pre-training phase.

    Attributes:
        epochs: Number of full passes through collected data during pre-training.
        batch_size: Number of samples per training batch. Must be >= 4 for
            contrastive learning to produce meaningful positive/negative pairs.
        learning_rate: Base learning rate for the Adam optimizer.
        min_samples: Minimum number of collected transaction samples before
            pre-training can begin. Below this threshold, should_pretrain()
            returns False.
        mask_ratio: Fraction of input features to mask when creating augmented
            views for contrastive learning. Higher values force the encoder
            to learn more robust representations.
        noise_std: Standard deviation of Gaussian noise added to augmented
            views. Provides continuous perturbation complementing discrete
            feature masking.
        temperature: NT-Xent temperature parameter controlling the sharpness
            of the contrastive similarity distribution. Lower values make the
            model more discriminative between similar and dissimilar pairs.
        warmup_epochs: Number of epochs with linearly increasing learning rate
            before reaching the base learning rate.
        reconstruction_finetune_epochs: Number of epochs to fine-tune the VAE
            with reconstruction loss after contrastive pre-training completes.
    """
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001
    min_samples: int = 100
    mask_ratio: float = 0.15
    noise_std: float = 0.05
    temperature: float = 0.5
    warmup_epochs: int = 5
    reconstruction_finetune_epochs: int = 10


@dataclass
class PretrainingMetrics:
    """Metrics captured at the end of each pre-training epoch.

    Attributes:
        epoch: Zero-indexed epoch number.
        loss: Average contrastive loss for this epoch.
        learning_rate: Learning rate used during this epoch.
        elapsed_time: Wall-clock time in seconds for this epoch.
        timestamp: Unix timestamp when the epoch completed.
    """
    epoch: int
    loss: float
    learning_rate: float
    elapsed_time: float
    timestamp: float


class ContrastivePretrainer:
    """Manages the self-supervised pre-training phase at genesis.

    This class orchestrates the full pre-training lifecycle:

    1. **Collection phase**: Transaction feature vectors are accumulated via
       ``add_sample()``. These are assumed to be normal transactions since
       genesis blocks contain only legitimate activity.

    2. **Readiness check**: ``should_pretrain()`` returns True once enough
       samples have been collected (controlled by ``config.min_samples``).

    3. **Contrastive pre-training**: ``pretrain()`` runs the full training
       loop, using the ContrastiveLearner to train the VAE encoder with
       NT-Xent loss on augmented views of the collected transactions.

    4. **Reconstruction fine-tuning**: After contrastive learning, a short
       VAE reconstruction fine-tuning phase aligns the full encoder-decoder
       pipeline for anomaly detection via reconstruction error.

    5. **Completion**: ``is_complete`` becomes True, signaling the system
       to transition to online learning.

    Args:
        vae_model: A VAE instance from positronic.ai.models.vae. Must have
            ``encoder``, ``decoder``, ``train_step(batch, optimizer)``,
            and ``parameters()`` methods.
        config: Pre-training configuration. Uses defaults if not provided.

    Example::

        from positronic.ai.models.vae import VAE
        from positronic.ai.training.pretraining import ContrastivePretrainer, PretrainingConfig

        vae = VAE(input_dim=32, latent_dim=16)
        config = PretrainingConfig(epochs=30, min_samples=50)
        pretrainer = ContrastivePretrainer(vae, config)

        # Collect genesis transactions
        for tx_features in genesis_transactions:
            pretrainer.add_sample(tx_features)

        if pretrainer.should_pretrain():
            metrics = pretrainer.pretrain()
            print(f"Pre-training complete. Final loss: {metrics[-1].loss:.4f}")
    """

    def __init__(self, vae_model, config: PretrainingConfig = None):
        self.config = config or PretrainingConfig()
        self.vae = vae_model

        # Build contrastive learner wrapping the VAE encoder.
        # The ContrastiveLearner creates augmented views internally and
        # computes NT-Xent loss across the batch.
        from positronic.ai.models.contrastive import ContrastiveLearner
        self.learner = ContrastiveLearner(
            encoder=self.vae.encoder,
            temperature=self.config.temperature,
        )
        self.learner.mask_ratio = self.config.mask_ratio
        self.learner.noise_std = self.config.noise_std

        # Data collection buffer
        self._samples: List[np.ndarray] = []

        # State tracking
        self._is_complete: bool = False
        self._metrics: List[PretrainingMetrics] = []

    @property
    def is_complete(self) -> bool:
        """Whether pre-training has finished successfully."""
        return self._is_complete

    @property
    def num_samples(self) -> int:
        """Number of transaction feature vectors collected so far."""
        return len(self._samples)

    @property
    def metrics_history(self) -> List[PretrainingMetrics]:
        """Full list of per-epoch metrics from the most recent pre-training run."""
        return list(self._metrics)

    def add_sample(self, feature_vector: np.ndarray) -> None:
        """Add a transaction feature vector for pre-training.

        Feature vectors are copied to prevent external mutation.

        Args:
            feature_vector: 1-D numpy array of transaction features.
        """
        self._samples.append(feature_vector.copy())

    def add_samples(self, feature_vectors: List[np.ndarray]) -> None:
        """Add multiple transaction feature vectors at once.

        Args:
            feature_vectors: List of 1-D numpy arrays of transaction features.
        """
        for fv in feature_vectors:
            self._samples.append(fv.copy())

    def should_pretrain(self) -> bool:
        """Check if we have enough samples to begin pre-training.

        Returns True only if pre-training has not yet completed and the
        number of collected samples meets or exceeds ``config.min_samples``.

        Returns:
            True if pre-training should be initiated.
        """
        return (
            not self._is_complete
            and len(self._samples) >= self.config.min_samples
        )

    def pretrain(self) -> List[PretrainingMetrics]:
        """Run the full contrastive pre-training pipeline.

        Executes the following stages:

        1. **Contrastive learning**: Trains the encoder to map similar
           transaction patterns close together in embedding space using
           NT-Xent loss with augmented views.

        2. **Reconstruction fine-tuning**: Trains the full VAE (encoder +
           decoder) to reconstruct normal transactions. This enables
           anomaly detection via reconstruction error.

        After completion, ``is_complete`` is set to True.

        Returns:
            List of PretrainingMetrics, one per contrastive training epoch.

        Raises:
            ValueError: If not enough samples have been collected.
        """
        if len(self._samples) < self.config.min_samples:
            raise ValueError(
                f"Not enough samples for pre-training. "
                f"Have {len(self._samples)}, need {self.config.min_samples}."
            )

        from positronic.ai.engine.optimizers import Adam

        data = np.array(self._samples, dtype=np.float32)

        # Single optimizer covers both contrastive learner and VAE parameters
        # since the learner wraps the VAE encoder.
        optimizer = Adam(
            self.learner.parameters() + self.vae.parameters(),
            lr=self.config.learning_rate,
        )

        metrics: List[PretrainingMetrics] = []

        for epoch in range(self.config.epochs):
            start = time.time()

            # Linear learning rate warmup
            if epoch < self.config.warmup_epochs:
                lr = self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
                optimizer.lr = lr

            # Shuffle data indices for this epoch
            indices = np.random.permutation(len(data))
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(data), self.config.batch_size):
                batch_idx = indices[i : i + self.config.batch_size]
                if len(batch_idx) < 4:
                    # Contrastive learning requires a minimum batch size to
                    # form meaningful positive/negative pairs.
                    continue

                batch = data[batch_idx]
                loss = self.learner.train_step(batch, optimizer)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            elapsed = time.time() - start

            m = PretrainingMetrics(
                epoch=epoch,
                loss=avg_loss,
                learning_rate=optimizer.lr,
                elapsed_time=elapsed,
                timestamp=time.time(),
            )
            metrics.append(m)
            self._metrics.append(m)

        # Stage 2: Reconstruction fine-tuning
        self._finetune_reconstruction(data, optimizer)

        self._is_complete = True
        return metrics

    def _finetune_reconstruction(self, data: np.ndarray, optimizer) -> None:
        """Fine-tune the VAE with reconstruction loss after contrastive pre-training.

        This short phase aligns the full encoder-decoder pipeline so that
        normal transactions produce low reconstruction error, enabling
        anomaly detection from the start of online operations.

        Args:
            data: Full training dataset as a 2-D numpy array.
            optimizer: The Adam optimizer (reused from contrastive phase).
        """
        for epoch in range(self.config.reconstruction_finetune_epochs):
            indices = np.random.permutation(len(data))
            for i in range(0, len(data), self.config.batch_size):
                batch = data[indices[i : i + self.config.batch_size]]
                if len(batch) < 2:
                    continue
                self.vae.train_step(batch, optimizer)

    def get_stats(self) -> Dict:
        """Return a summary dictionary of pre-training state.

        Returns:
            Dictionary with keys: is_complete, samples_collected,
            epochs_completed, final_loss.
        """
        return {
            "is_complete": self._is_complete,
            "samples_collected": len(self._samples),
            "epochs_completed": len(self._metrics),
            "final_loss": self._metrics[-1].loss if self._metrics else None,
        }
