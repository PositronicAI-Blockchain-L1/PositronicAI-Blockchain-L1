"""
Positronic - Model Registry
Manages model versions, rollback, and A/B testing (canary deployments).

The registry provides a central catalog of all trained model versions, allowing
the system to track model evolution, roll back to previous versions if a new
model underperforms, and gradually route traffic to a new model version via
canary deployments before full promotion.

Dependencies:
    - numpy: Array operations for state dict handling
    - positronic.ai.engine.model.Model: Base model interface with state_dict()
"""

import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelVersion:
    """A registered model version with metadata.

    Attributes:
        name: Logical model name (e.g., 'vae', 'meta_ensemble').
        version: Sequential version number starting from 1.
        state_dict: Snapshot of model parameters as {name: numpy array}.
        metrics: Training/validation metrics at registration time
            (e.g., {'loss': 0.05, 'accuracy': 0.97}).
        timestamp: Unix timestamp when this version was registered.
        checksum: SHA-256 checksum (first 16 hex chars) of the serialized
            state for integrity verification.
        is_active: Whether this is the currently active (primary) version.
        is_canary: Whether this version is deployed as a canary for A/B testing.
        canary_traffic: Fraction of traffic (0.0 to 1.0) routed to this canary.
    """
    name: str
    version: int
    state_dict: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    timestamp: float
    checksum: str
    is_active: bool = False
    is_canary: bool = False
    canary_traffic: float = 0.0


class ModelRegistry:
    """
    Model version registry with versioning, rollback, and A/B testing.

    The registry stores snapshots of model state dictionaries along with
    metadata such as training metrics and integrity checksums. It supports:

        - **Version tracking**: Each call to register() creates a new numbered
          version with a timestamp and checksum.
        - **Activation**: One version per model name can be marked as active
          (the primary version used for inference).
        - **Rollback**: Revert to a previous version by number or by step count.
        - **Canary deployments**: Route a fraction of traffic to a new version
          for A/B testing before full promotion.
        - **Checksum verification**: Each version's state is checksummed to
          detect corruption during storage or transfer.
    """

    def __init__(self):
        self._models: Dict[str, List[ModelVersion]] = {}
        self._active: Dict[str, int] = {}

    def register(self, name: str, model, metrics: dict = None) -> ModelVersion:
        """Register a new model version.

        Takes a snapshot of the model's current state_dict and stores it
        as a new version in the registry.

        Args:
            name: Logical model name (e.g., 'vae', 'temporal_attention').
            model: Model instance that exposes a state_dict() method returning
                a dictionary of {param_name: numpy.ndarray}.
            metrics: Optional dictionary of metrics to associate with this
                version (e.g., training loss, validation accuracy).

        Returns:
            The newly created ModelVersion instance.
        """
        if name not in self._models:
            self._models[name] = []

        version_num = len(self._models[name]) + 1
        state = model.state_dict()

        # Compute integrity checksum over sorted parameter bytes
        all_bytes = b""
        for k in sorted(state.keys()):
            all_bytes += state[k].tobytes()
        checksum = hashlib.sha256(all_bytes).hexdigest()[:16]

        mv = ModelVersion(
            name=name,
            version=version_num,
            state_dict={k: v.copy() for k, v in state.items()},
            metrics=metrics or {},
            timestamp=time.time(),
            checksum=checksum,
        )

        self._models[name].append(mv)
        return mv

    def activate(self, name: str, version: int = None) -> ModelVersion:
        """Activate a model version, making it the primary for inference.

        Deactivates all other versions of the same model name and marks the
        specified version as active.

        Args:
            name: Logical model name.
            version: Version number to activate. If None, activates the latest
                registered version.

        Returns:
            The activated ModelVersion instance.

        Raises:
            KeyError: If the model name is not found or the specified version
                does not exist.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")

        if version is None:
            version = len(self._models[name])  # Latest

        # Deactivate all versions of this model
        for mv in self._models[name]:
            mv.is_active = False

        # Activate the specified version
        for mv in self._models[name]:
            if mv.version == version:
                mv.is_active = True
                self._active[name] = version
                return mv

        raise KeyError(
            f"Version {version} not found for model '{name}'. "
            f"Available versions: 1-{len(self._models[name])}"
        )

    def get_active(self, name: str) -> Optional[ModelVersion]:
        """Get the currently active version of a model.

        If no version has been explicitly activated, returns the latest
        registered version. Returns None if no versions exist.

        Args:
            name: Logical model name.

        Returns:
            The active ModelVersion, or the latest version, or None.
        """
        if name not in self._active:
            # Return latest if any versions exist
            if name in self._models and self._models[name]:
                return self._models[name][-1]
            return None

        version = self._active[name]
        for mv in self._models[name]:
            if mv.version == version:
                return mv
        return None

    def rollback(self, name: str, steps: int = 1) -> Optional[ModelVersion]:
        """Rollback to a previous model version.

        Moves the active version back by the specified number of steps.
        The minimum version is 1 (the first registered version).

        Args:
            name: Logical model name.
            steps: Number of versions to roll back. Defaults to 1.

        Returns:
            The newly activated ModelVersion after rollback, or None if
            the model has no versions or no active version.
        """
        if name not in self._active or name not in self._models:
            return None

        current = self._active[name]
        target = max(1, current - steps)
        return self.activate(name, target)

    def set_canary(self, name: str, version: int, traffic_percent: float):
        """Set a canary deployment for A/B testing.

        Marks the specified version as a canary and configures the fraction
        of inference traffic that should be routed to it.

        Args:
            name: Logical model name.
            version: Version number to designate as canary.
            traffic_percent: Percentage of traffic (0-100) to route to
                the canary version. Clamped to [0, 100].
        """
        for mv in self._models.get(name, []):
            if mv.version == version:
                mv.is_canary = True
                mv.canary_traffic = min(max(traffic_percent / 100.0, 0.0), 1.0)
                return

    def should_use_canary(self, name: str) -> bool:
        """Determine whether the current request should use the canary version.

        Uses random sampling based on the configured canary traffic fraction.

        Args:
            name: Logical model name.

        Returns:
            True if the request should be routed to the canary version.
        """
        for mv in self._models.get(name, []):
            if mv.is_canary:
                return np.random.random() < mv.canary_traffic
        return False

    def promote_canary(self, name: str):
        """Promote the canary version to active.

        Removes the canary flag and activates the canary version as the
        new primary. This completes the A/B test by fully deploying the
        canary version.

        Args:
            name: Logical model name.
        """
        for mv in self._models.get(name, []):
            if mv.is_canary:
                mv.is_canary = False
                mv.canary_traffic = 0.0
                self.activate(name, mv.version)
                return

    def get_version_history(self, name: str) -> List[dict]:
        """Get the full version history for a model.

        Args:
            name: Logical model name.

        Returns:
            List of dictionaries, one per version, containing version number,
            checksum, metrics, timestamp, active status, and canary status.
            Returns an empty list if the model name is not found.
        """
        return [
            {
                "version": mv.version,
                "checksum": mv.checksum,
                "metrics": mv.metrics,
                "timestamp": mv.timestamp,
                "is_active": mv.is_active,
                "is_canary": mv.is_canary,
            }
            for mv in self._models.get(name, [])
        ]

    def get_stats(self) -> dict:
        """Get registry-wide statistics.

        Returns:
            Dictionary containing:
                - registered_models: List of model names in the registry.
                - total_versions: Total number of versions across all models.
                - active_versions: Mapping of model name to active version number.
        """
        return {
            "registered_models": list(self._models.keys()),
            "total_versions": sum(len(v) for v in self._models.values()),
            "active_versions": dict(self._active),
        }
