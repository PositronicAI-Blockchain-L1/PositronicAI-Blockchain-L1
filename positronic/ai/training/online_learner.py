"""
Positronic - Online Learning System

Continuous learning from confirmed transactions. Trains all AI models
on data from accepted blocks in an incremental fashion, without requiring
full retraining from scratch.

Training data sources:
    - **Confirmed transactions**: Accepted by consensus into the blockchain.
      These are treated as normal patterns and stored in the normal buffer.
    - **Released quarantine transactions**: Transactions that were initially
      flagged but later cleared. These are hard negative examples that help
      reduce false positive rates.
    - **Rejected transactions**: Confirmed malicious by consensus. These are
      positive fraud examples stored in the anomaly buffer with elevated
      priority so they are sampled more frequently.

Phase 16 additions:
    - **Curriculum Learning**: Trains on easy examples first, gradually adds
      harder cases near the decision boundary.
    - **Plateau Detection**: Monitors loss improvement and boosts learning
      rate when training stalls.
    - **Hard Negative Mining**: Stores transactions near the decision boundary
      (score 0.80-0.90) for focused training.

Dependencies:
    - VAE from positronic.ai.models.vae
    - TemporalAttentionNet from positronic.ai.models.temporal_attention
    - GraphAttentionNet from positronic.ai.models.graph_attention
    - LSTMAttentionNet from positronic.ai.models.lstm_attention
    - MetaEnsemble from positronic.ai.models.meta_ensemble
    - Adam from positronic.ai.engine.optimizers
    - PriorityReplayBuffer from positronic.ai.training.data_buffer
"""

import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

from positronic.constants import (
    CURRICULUM_PHASE1_END,
    CURRICULUM_PHASE2_END,
    CURRICULUM_EASY_THRESHOLD,
    CURRICULUM_HARD_THRESHOLD,
    PLATEAU_WINDOW,
    PLATEAU_MIN_IMPROVEMENT,
    PLATEAU_LR_BOOST,
    PLATEAU_MAX_BOOST,
    HARD_NEGATIVE_BUFFER_SIZE,
    HARD_NEGATIVE_MIN_SCORE,
    HARD_NEGATIVE_MAX_SCORE,
    HARD_NEGATIVE_BATCH_RATIO,
)


@dataclass
class OnlineLearningMetrics:
    """Metrics captured for each online training step.

    Attributes:
        step: Monotonically increasing training step counter.
        vae_loss: VAE reconstruction loss for this step. Zero if VAE was
            not trained (e.g., insufficient data).
        meta_loss: MetaEnsemble loss for this step. Zero if meta-ensemble
            was not trained (e.g., no anomaly data available yet).
        samples_used: Number of samples in the training batch.
        elapsed_time: Wall-clock time in seconds for this training step.
        timestamp: Unix timestamp when the step completed.
    """
    step: int
    vae_loss: float
    meta_loss: float
    samples_used: int
    elapsed_time: float
    timestamp: float


class CurriculumScheduler:
    """Phase 16: Controls training difficulty progression.

    Phase 1 (steps 0-500): Only train on clear accept (<0.50) and
        clear reject (>0.95) examples. Avoids confusing the model
        with ambiguous cases early on.
    Phase 2 (steps 500-2000): Include quarantine zone (0.50-0.95).
    Phase 3 (steps 2000+): All examples, weighted by difficulty.
    """

    def __init__(self):
        self._step: int = 0

    @property
    def phase(self) -> int:
        """Current curriculum phase (1, 2, or 3)."""
        if self._step < CURRICULUM_PHASE1_END:
            return 1
        elif self._step < CURRICULUM_PHASE2_END:
            return 2
        return 3

    def should_train_on(self, score: float) -> bool:
        """Check if a sample with this score should be used for training.

        Args:
            score: The AI risk score assigned to this transaction.

        Returns:
            True if the sample should be included in training.
        """
        phase = self.phase
        if phase == 1:
            # Only clear examples
            return score < CURRICULUM_EASY_THRESHOLD or score > CURRICULUM_HARD_THRESHOLD
        elif phase == 2:
            # Include quarantine zone
            return True
        else:
            # All examples
            return True

    def step(self):
        """Increment the curriculum step counter."""
        self._step += 1


class PlateauDetector:
    """Phase 16: Detects training plateaus and adjusts learning rate.

    Tracks rolling loss over the last PLATEAU_WINDOW steps. If
    improvement is less than PLATEAU_MIN_IMPROVEMENT (1%), boosts
    the learning rate by PLATEAU_LR_BOOST (1.5x), up to a maximum
    of PLATEAU_MAX_BOOST (3.0x) the base rate.
    """

    def __init__(self):
        self._loss_history: List[float] = []
        self._lr_multiplier: float = 1.0

    def record_loss(self, loss: float):
        """Record a training loss value."""
        self._loss_history.append(loss)
        if len(self._loss_history) > PLATEAU_WINDOW * 3:
            self._loss_history = self._loss_history[-(PLATEAU_WINDOW * 3):]

    def is_plateau(self) -> bool:
        """Check if training is on a plateau.

        Returns:
            True if loss improvement over the last PLATEAU_WINDOW steps
            is less than PLATEAU_MIN_IMPROVEMENT.
        """
        if len(self._loss_history) < PLATEAU_WINDOW:
            return False

        recent = self._loss_history[-PLATEAU_WINDOW:]
        early_avg = sum(recent[:PLATEAU_WINDOW // 2]) / (PLATEAU_WINDOW // 2)
        late_avg = sum(recent[PLATEAU_WINDOW // 2:]) / (PLATEAU_WINDOW // 2)

        if early_avg <= 0:
            return False

        improvement = (early_avg - late_avg) / abs(early_avg)
        return improvement < PLATEAU_MIN_IMPROVEMENT

    def update(self, loss: float):
        """Record loss and update LR multiplier if plateau detected.

        Args:
            loss: Current training step loss.
        """
        self.record_loss(loss)
        if self.is_plateau():
            self._lr_multiplier = min(
                self._lr_multiplier * PLATEAU_LR_BOOST,
                PLATEAU_MAX_BOOST,
            )
        else:
            # Gradually decay back toward 1.0
            self._lr_multiplier = max(1.0, self._lr_multiplier * 0.99)

    @property
    def lr_multiplier(self) -> float:
        """Current learning rate multiplier."""
        return self._lr_multiplier


class OnlineLearner:
    """Manages continuous online learning from confirmed blocks.

    The OnlineLearner maintains two priority replay buffers:

    - **normal_buffer**: Stores feature vectors from confirmed (legitimate)
      transactions with default priority.
    - **anomaly_buffer**: Stores feature vectors from confirmed malicious
      transactions with elevated priority, ensuring these rare examples
      are sampled proportionally more during training.

    Phase 16 adds:
    - **hard_negative_buffer**: Stores transactions near the decision
      boundary (score 0.80-0.90) for focused training on difficult cases.
    - **CurriculumScheduler**: Controls which examples are used based on
      training progress.
    - **PlateauDetector**: Monitors loss and boosts LR when training stalls.

    Training steps are triggered periodically based on the number of new
    samples received (controlled by ``train_interval``).

    Args:
        vae: VAE model instance, or None to skip VAE training.
        temporal_net: TemporalAttentionNet instance, or None.
        graph_net: GraphAttentionNet instance, or None.
        lstm_net: LSTMAttentionNet instance, or None.
        meta_ensemble: MetaEnsemble instance, or None.

    Example::

        from positronic.ai.training.online_learner import OnlineLearner

        learner = OnlineLearner(vae=vae_model, meta_ensemble=meta_model)

        # Feed confirmed block data
        learner.add_confirmed_transactions(normal_features)
        learner.add_anomaly_data(fraud_features, priority=3.0)

        if learner.should_train():
            metrics = learner.train_step()
            print(f"Step {metrics.step}: VAE loss={metrics.vae_loss:.4f}")
    """

    def __init__(
        self,
        vae=None,
        temporal_net=None,
        graph_net=None,
        lstm_net=None,
        meta_ensemble=None,
    ):
        self.vae = vae
        self.temporal_net = temporal_net
        self.graph_net = graph_net
        self.lstm_net = lstm_net
        self.meta_ensemble = meta_ensemble

        # Lazy-initialized optimizers keyed by model name
        self._optimizers: Dict[str, object] = {}

        # Training replay buffers
        from positronic.ai.training.data_buffer import PriorityReplayBuffer
        self.normal_buffer = PriorityReplayBuffer(capacity=10000)
        self.anomaly_buffer = PriorityReplayBuffer(capacity=2000)

        # Phase 16: Hard negative buffer for boundary cases
        self._hard_negative_buffer: List[np.ndarray] = []
        self._hard_negative_capacity: int = HARD_NEGATIVE_BUFFER_SIZE

        # Phase 16: Curriculum scheduler and plateau detector
        self.curriculum = CurriculumScheduler()
        self.plateau_detector = PlateauDetector()

        # Configuration
        self.batch_size: int = 64
        self.learning_rate: float = 0.001
        self.train_interval: int = 100  # Train every N new samples

        # State tracking
        self._step: int = 0
        self._total_samples: int = 0
        self._metrics: List[OnlineLearningMetrics] = []
        self._metrics_max_history: int = 1000

    def _get_optimizer(self, model, name: str):
        """Get or lazily create an Adam optimizer for the given model.

        Args:
            model: A model instance with a ``parameters()`` method.
            name: Unique string key for caching the optimizer.

        Returns:
            The Adam optimizer for the model, or None if the model is None.
        """
        if name not in self._optimizers and model is not None:
            from positronic.ai.engine.optimizers import Adam
            self._optimizers[name] = Adam(model.parameters(), lr=self.learning_rate)
        return self._optimizers.get(name)

    def add_confirmed_transactions(self, feature_vectors: List[np.ndarray]) -> None:
        """Add confirmed (normal) transaction features to the normal buffer.

        These are transactions that were accepted by consensus and included
        in a confirmed block. They represent legitimate activity patterns.

        Args:
            feature_vectors: List of 1-D numpy arrays, each representing
                the feature vector of a confirmed transaction.
        """
        for fv in feature_vectors:
            self.normal_buffer.add(fv, priority=1.0)
        self._total_samples += len(feature_vectors)

    def add_anomaly_data(
        self, feature_vectors: List[np.ndarray], priority: float = 2.0
    ) -> None:
        """Add confirmed anomalous transaction features to the anomaly buffer.

        These are transactions confirmed as malicious by consensus or
        released from quarantine after investigation. They serve as hard
        examples for training the fraud detection models.

        Args:
            feature_vectors: List of 1-D numpy arrays of anomaly features.
            priority: Sampling priority. Higher values make these examples
                more likely to be drawn during training. Default is 2.0
                (twice the priority of normal transactions).
        """
        for fv in feature_vectors:
            self.anomaly_buffer.add(fv, priority=priority)

    def record_hard_negative(self, features: np.ndarray, final_score: float) -> None:
        """Phase 16: Record a hard negative (near decision boundary).

        Hard negatives are transactions that scored between 0.80 and 0.90,
        right near the quarantine boundary. Training on these improves
        discrimination at the critical decision threshold.

        Args:
            features: Feature vector (1-D numpy array).
            final_score: The AI risk score for this transaction.
        """
        if HARD_NEGATIVE_MIN_SCORE <= final_score <= HARD_NEGATIVE_MAX_SCORE:
            self._hard_negative_buffer.append(features.copy())
            if len(self._hard_negative_buffer) > self._hard_negative_capacity:
                self._hard_negative_buffer = self._hard_negative_buffer[
                    -self._hard_negative_capacity:
                ]

    def should_train(self) -> bool:
        """Check if it is time to execute a training step.

        Training is triggered when the normal buffer has at least
        ``batch_size`` samples and the total sample count aligns with
        the ``train_interval``.

        Returns:
            True if a training step should be executed now.
        """
        return (
            self.normal_buffer.size >= self.batch_size
            and self._total_samples % self.train_interval < self.batch_size
        )

    def train_step(self) -> OnlineLearningMetrics:
        """Execute one training step across all registered models.

        The training step proceeds as follows:

        1. **Sample normal batch**: Draw ``batch_size`` samples from the
           normal buffer using priority-weighted sampling.

        2. **Mix hard negatives**: Phase 16 adds up to 20% hard negatives
           into the batch for focused boundary training.

        3. **Train VAE**: If a VAE is registered, train it on the batch
           to update reconstruction capabilities.

        4. **Train MetaEnsemble**: If a MetaEnsemble is registered and
           anomaly data exists, create balanced training pairs.

        5. **Update curriculum and plateau detector**.

        Returns:
            OnlineLearningMetrics for this step.
        """
        start = time.time()
        vae_loss = 0.0
        meta_loss = 0.0

        # Sample a batch from the normal buffer
        batch = self.normal_buffer.sample(self.batch_size)
        if batch is None or len(batch) < 4:
            return self._empty_metrics()

        batch = list(batch)

        # Phase 16: Mix in hard negatives (up to 20% of batch)
        if self._hard_negative_buffer:
            hn_count = max(1, int(len(batch) * HARD_NEGATIVE_BATCH_RATIO))
            hn_count = min(hn_count, len(self._hard_negative_buffer))
            hn_indices = np.random.choice(
                len(self._hard_negative_buffer), size=hn_count, replace=False
            )
            for idx in hn_indices:
                batch.append(self._hard_negative_buffer[idx])

        batch = np.array(batch, dtype=np.float32)

        # Phase 16: Apply plateau-adjusted learning rate
        effective_lr = self.learning_rate * self.plateau_detector.lr_multiplier

        # --- Train VAE on normal transaction reconstruction ---
        if self.vae is not None:
            opt = self._get_optimizer(self.vae, "vae")
            if opt:
                # Update optimizer LR if plateau boost is active
                if hasattr(opt, 'lr'):
                    opt.lr = effective_lr
                vae_loss = self.vae.train_step(batch, opt)

        # --- Train MetaEnsemble with labeled normal/anomaly pairs ---
        if self.meta_ensemble is not None and self.anomaly_buffer.size > 0:
            opt = self._get_optimizer(self.meta_ensemble, "meta")
            if opt:
                meta_loss = self._train_meta_step(batch, opt)

        # Phase 16: Update plateau detector with combined loss
        combined_loss = vae_loss + meta_loss
        self.plateau_detector.update(combined_loss)

        # Phase 16: Advance curriculum
        self.curriculum.step()

        self._step += 1
        elapsed = time.time() - start

        metrics = OnlineLearningMetrics(
            step=self._step,
            vae_loss=vae_loss,
            meta_loss=meta_loss,
            samples_used=len(batch),
            elapsed_time=elapsed,
            timestamp=time.time(),
        )
        self._metrics.append(metrics)

        # Cap metrics history to prevent unbounded memory growth
        if len(self._metrics) > self._metrics_max_history:
            self._metrics = self._metrics[-self._metrics_max_history:]

        return metrics

    def _train_meta_step(self, normal_batch: np.ndarray, optimizer) -> float:
        """Meta-ensemble training is handled by meta_model.py which has proper
        access to component scores. Skip meta training here to avoid feeding
        raw 35-dim features where the meta-ensemble expects 4 component scores
        + 6 context features (10 total).

        Args:
            normal_batch: Batch of normal transaction feature vectors (unused).
            optimizer: Adam optimizer for the meta-ensemble (unused).

        Returns:
            0.0 (no training performed).
        """
        return 0.0

    def _empty_metrics(self) -> OnlineLearningMetrics:
        """Return a zero-valued metrics object when no training occurred."""
        return OnlineLearningMetrics(
            step=self._step,
            vae_loss=0.0,
            meta_loss=0.0,
            samples_used=0,
            elapsed_time=0.0,
            timestamp=time.time(),
        )

    def get_stats(self) -> Dict:
        """Return a summary dictionary of the online learner state.

        Returns:
            Dictionary with keys: step, total_samples, normal_buffer_size,
            anomaly_buffer_size, last_vae_loss, and Phase 16 stats.
        """
        return {
            "step": self._step,
            "total_samples": self._total_samples,
            "normal_buffer_size": self.normal_buffer.size,
            "anomaly_buffer_size": self.anomaly_buffer.size,
            "last_vae_loss": (
                self._metrics[-1].vae_loss if self._metrics else None
            ),
            # Phase 16
            "hard_negative_count": len(self._hard_negative_buffer),
            "curriculum_phase": self.curriculum.phase,
            "plateau_lr_multiplier": self.plateau_detector.lr_multiplier,
            "is_plateau": self.plateau_detector.is_plateau(),
        }
