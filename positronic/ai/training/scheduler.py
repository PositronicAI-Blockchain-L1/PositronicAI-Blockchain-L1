"""
Positronic - Learning Rate Schedulers

Provides learning rate scheduling strategies for the AI training pipeline.
Each scheduler wraps an optimizer and adjusts its learning rate on each
call to ``step()``.

Available schedulers:
    - **LearningRateScheduler**: Base class with constant learning rate.
    - **WarmupCosineScheduler**: Linear warmup followed by cosine decay.
      Standard in modern transformer-based architectures.
    - **StepScheduler**: Multiplicative step decay at fixed intervals.
    - **CyclicScheduler**: Triangular cyclic learning rate that oscillates
      between a minimum and maximum, helping escape local minima.

Dependencies:
    - Optimizer from positronic.ai.engine.optimizers (must have a writable ``lr``
      attribute).
"""

import math
import numpy as np
from typing import Optional


class LearningRateScheduler:
    """Base learning rate scheduler with constant rate.

    Subclasses override ``get_lr()`` to implement specific schedules.
    The ``step()`` method increments the step counter and applies the
    computed learning rate to the optimizer.

    Args:
        optimizer: An optimizer instance with a writable ``lr`` attribute.
        base_lr: The base (initial) learning rate.
    """

    def __init__(self, optimizer, base_lr: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_count: int = 0

    def step(self) -> float:
        """Advance one step and update the optimizer's learning rate.

        Returns:
            The new learning rate after this step.
        """
        self.step_count += 1
        lr = self.get_lr()
        self.optimizer.lr = lr
        return lr

    def get_lr(self) -> float:
        """Compute the learning rate for the current step.

        Returns:
            The learning rate. Base implementation returns constant ``base_lr``.
        """
        return self.base_lr


class WarmupCosineScheduler(LearningRateScheduler):
    """Linear warmup followed by cosine annealing to a minimum learning rate.

    During the warmup phase (steps 0 to ``warmup_steps``), the learning rate
    increases linearly from 0 to ``base_lr``. After warmup, it decays
    following a cosine curve from ``base_lr`` down to ``min_lr`` over the
    remaining steps.

    This schedule is the standard in modern transformer training and provides
    smooth, stable convergence.

    Args:
        optimizer: An optimizer instance with a writable ``lr`` attribute.
        base_lr: Peak learning rate reached at the end of warmup.
        warmup_steps: Number of steps for the linear warmup phase.
        total_steps: Total number of training steps (warmup + decay).
        min_lr: Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        min_lr: float = 1e-6,
    ):
        super().__init__(optimizer, base_lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self) -> float:
        """Compute learning rate with warmup + cosine decay.

        Returns:
            The learning rate for the current step.
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup: ramp from 0 to base_lr
            return self.base_lr * (self.step_count / max(self.warmup_steps, 1))
        else:
            # Cosine decay: anneal from base_lr to min_lr
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )


class StepScheduler(LearningRateScheduler):
    """Step decay scheduler that reduces the learning rate by a fixed factor
    at regular intervals.

    Every ``step_size`` steps, the learning rate is multiplied by ``gamma``.
    After k intervals, the rate is ``base_lr * gamma^k``.

    Args:
        optimizer: An optimizer instance with a writable ``lr`` attribute.
        base_lr: The initial learning rate.
        step_size: Number of steps between each decay.
        gamma: Multiplicative decay factor applied at each interval.
            Must be in (0, 1) for decay behavior.
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        step_size: int = 1000,
        gamma: float = 0.5,
    ):
        super().__init__(optimizer, base_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        """Compute learning rate with step decay.

        Returns:
            The learning rate for the current step.
        """
        return self.base_lr * (self.gamma ** (self.step_count // self.step_size))


class CyclicScheduler(LearningRateScheduler):
    """Cyclic learning rate scheduler that oscillates between a minimum
    and maximum learning rate.

    The triangular cycling pattern helps the optimizer escape local minima
    and saddle points by periodically increasing the learning rate. This
    is particularly useful during online learning where the data distribution
    shifts over time.

    Args:
        optimizer: An optimizer instance with a writable ``lr`` attribute.
        base_lr: Minimum learning rate (bottom of cycle).
        max_lr: Maximum learning rate (peak of cycle). Defaults to
            10x the base learning rate.
        cycle_length: Number of steps per full cycle (bottom -> top -> bottom).
    """

    def __init__(
        self,
        optimizer,
        base_lr: float,
        max_lr: float = None,
        cycle_length: int = 500,
    ):
        super().__init__(optimizer, base_lr)
        self.max_lr = max_lr if max_lr is not None else base_lr * 10
        self.cycle_length = cycle_length

    def get_lr(self) -> float:
        """Compute learning rate with triangular cycling.

        Returns:
            The learning rate for the current step, oscillating between
            ``base_lr`` and ``max_lr``.
        """
        cycle = math.floor(self.step_count / self.cycle_length)
        x = abs(self.step_count / self.cycle_length - cycle - 0.5) * 2
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x)
