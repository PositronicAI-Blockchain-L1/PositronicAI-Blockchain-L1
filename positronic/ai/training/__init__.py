"""
Positronic - AI Training Pipeline
Handles self-supervised pre-training, online learning, and federated gossip.
"""

from positronic.ai.training.pretraining import ContrastivePretrainer
from positronic.ai.training.online_learner import OnlineLearner
from positronic.ai.training.federated import FederatedAverager
from positronic.ai.training.scheduler import LearningRateScheduler, WarmupCosineScheduler
from positronic.ai.training.curriculum import CurriculumManager
from positronic.ai.training.data_buffer import PriorityReplayBuffer

__all__ = [
    "ContrastivePretrainer",
    "OnlineLearner",
    "FederatedAverager",
    "LearningRateScheduler",
    "WarmupCosineScheduler",
    "CurriculumManager",
    "PriorityReplayBuffer",
]
