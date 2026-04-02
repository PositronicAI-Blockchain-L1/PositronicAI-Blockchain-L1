"""
Positronic - Curriculum Learning

Orders training examples from easy to hard for stable, efficient training.
In the context of fraud detection:

    - **Easy examples**: Typical, common transaction patterns that the model
      can learn to reconstruct with low error early in training.
    - **Hard examples**: Edge cases, unusual but legitimate transactions, and
      anomalous patterns that require more model capacity to handle.

By starting with easy examples and gradually introducing harder ones, the
model builds a stable representation of normal behavior before encountering
confusing edge cases. This prevents catastrophic forgetting and improves
convergence stability during online learning.

Difficulty scoring methods:
    - **reconstruction_error**: Uses the VAE reconstruction error as a proxy
      for difficulty. High reconstruction error = harder example.
    - **value_deviation**: Uses the feature-space distance from the dataset
      mean. Outliers in feature space are considered harder.

Dependencies:
    - numpy for numerical operations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class CurriculumConfig:
    """Configuration for the curriculum learning schedule.

    Attributes:
        initial_difficulty: Fraction of easiest examples to include at the
            start of training. Value between 0 and 1, where 0.3 means only
            the easiest 30% of examples are eligible initially.
        ramp_steps: Number of training steps over which the difficulty
            threshold linearly increases from ``initial_difficulty`` to 1.0.
        difficulty_metric: Method for scoring example difficulty. One of
            "reconstruction_error" or "value_deviation".
    """
    initial_difficulty: float = 0.3
    ramp_steps: int = 1000
    difficulty_metric: str = "reconstruction_error"


class CurriculumManager:
    """Manages curriculum learning schedule for training data selection.

    The curriculum manager controls which training examples are eligible
    for sampling at each training step. It assigns a difficulty score
    (0 = easiest, 1 = hardest) to each example and maintains a difficulty
    threshold that increases over time.

    Only examples whose difficulty score is below the current threshold
    are eligible for batch selection. As training progresses, the threshold
    rises, gradually exposing the model to harder examples.

    Args:
        config: Curriculum configuration. Uses defaults if not provided.

    Example::

        from positronic.ai.training.curriculum import CurriculumManager, CurriculumConfig

        config = CurriculumConfig(initial_difficulty=0.2, ramp_steps=500)
        curriculum = CurriculumManager(config)

        # Score difficulties based on reconstruction errors
        difficulties = curriculum.score_difficulty(data, reconstruction_errors)

        # Select a curriculum-appropriate batch
        batch = curriculum.select_batch(data, batch_size=32, difficulties=difficulties)

        # The curriculum automatically advances its internal step counter
    """

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self._step: int = 0
        self._difficulties: List[Tuple[int, float]] = []

    @property
    def current_difficulty(self) -> float:
        """Current difficulty threshold (0 to 1).

        At step 0, returns ``initial_difficulty``. Linearly increases to
        1.0 over ``ramp_steps``. After ``ramp_steps``, remains at 1.0
        (all examples eligible).

        Returns:
            The current difficulty threshold.
        """
        progress = min(self._step / max(self.config.ramp_steps, 1), 1.0)
        return (
            self.config.initial_difficulty
            + (1.0 - self.config.initial_difficulty) * progress
        )

    def score_difficulty(
        self,
        features: np.ndarray,
        reconstruction_errors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Score the difficulty of each example in the dataset.

        If ``reconstruction_errors`` are provided, they are used directly
        (normalized to [0, 1]). Otherwise, feature-space deviation from
        the dataset mean is used as a proxy.

        Args:
            features: 2-D array of shape (num_samples, num_features).
            reconstruction_errors: Optional 1-D array of per-sample
                reconstruction errors from the VAE.

        Returns:
            1-D array of difficulty scores in [0, 1], one per sample.
        """
        if reconstruction_errors is not None:
            # Higher reconstruction error = harder example
            if np.std(reconstruction_errors) > 0:
                min_err = np.min(reconstruction_errors)
                max_err = np.max(reconstruction_errors)
                scores = (reconstruction_errors - min_err) / (
                    max_err - min_err + 1e-8
                )
            else:
                scores = np.zeros_like(reconstruction_errors)
        else:
            # Use feature deviation from mean as proxy for difficulty
            mean = np.mean(features, axis=0)
            deviations = np.linalg.norm(features - mean, axis=1)
            if np.std(deviations) > 0:
                min_dev = np.min(deviations)
                max_dev = np.max(deviations)
                scores = (deviations - min_dev) / (max_dev - min_dev + 1e-8)
            else:
                scores = np.zeros(len(features))

        return scores

    def select_batch(
        self,
        data: np.ndarray,
        batch_size: int,
        difficulties: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Select a curriculum-appropriate batch from the dataset.

        Only examples whose difficulty score is at or below the current
        difficulty threshold are eligible. If fewer than ``batch_size``
        examples are eligible, all examples become eligible as a fallback.

        The internal step counter is automatically incremented after each
        batch selection.

        Args:
            data: 2-D array of shape (num_samples, num_features).
            batch_size: Number of samples to select.
            difficulties: Optional pre-computed difficulty scores. If not
                provided, scores are computed from ``data`` using feature
                deviation.

        Returns:
            2-D array of shape (selected_size, num_features) containing
            the selected batch. ``selected_size`` is
            ``min(batch_size, num_eligible)``.
        """
        if difficulties is None:
            difficulties = self.score_difficulty(data)

        threshold = self.current_difficulty
        eligible = np.where(difficulties <= threshold)[0]

        if len(eligible) < batch_size:
            # Fallback: use all examples when not enough are below threshold
            eligible = np.arange(len(data))

        selected = np.random.choice(
            eligible,
            size=min(batch_size, len(eligible)),
            replace=False,
        )

        self._step += 1

        return data[selected]

    def step(self) -> None:
        """Manually advance the curriculum step counter.

        This is useful when the curriculum threshold should advance even
        without a ``select_batch()`` call (e.g., during validation steps).
        """
        self._step += 1

    def reset(self) -> None:
        """Reset the curriculum to its initial state.

        Sets the step counter back to zero, restarting the difficulty ramp.
        """
        self._step = 0

    def get_stats(self) -> Dict:
        """Return a summary dictionary of the curriculum state.

        Returns:
            Dictionary with keys: step, current_difficulty, config.
        """
        return {
            "step": self._step,
            "current_difficulty": self.current_difficulty,
            "config": {
                "initial_difficulty": self.config.initial_difficulty,
                "ramp_steps": self.config.ramp_steps,
                "difficulty_metric": self.config.difficulty_metric,
            },
        }
