"""
Positronic - AI Model Service API
Internal Python API for all AI operations.

This module provides the central ModelService class that serves as the single
entry point for all AI operations within the Positronic system. Other components
such as the blockchain layer, neural validator, and RPC interface interact with
the AI subsystem exclusively through this service.

Dependencies:
    - positronic.ai.server.inference: InferencePipeline for cached, batched inference
    - positronic.ai.server.registry: ModelRegistry for version management
    - positronic.ai.server.health: HealthMonitor for system health tracking
    - positronic.ai.engine.serialization: State serialization for model export/import
    - positronic.ai.models.vae.VAE: Variational autoencoder for anomaly detection
    - positronic.ai.models.temporal_attention.TemporalAttentionNet: Temporal patterns
    - positronic.ai.models.graph_attention.GraphAttentionNet: Graph-based detection
    - positronic.ai.models.lstm_attention.LSTMAttentionNet: Sequence modeling
    - positronic.ai.models.meta_ensemble.MetaEnsemble: Ensemble scoring
"""

import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from positronic.utils.logging import get_logger
from positronic.ai.server.inference import InferencePipeline, InferenceResult
from positronic.ai.server.registry import ModelRegistry
from positronic.ai.server.health import HealthMonitor

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about the current AI model state.

    Attributes:
        version: Current active model version number.
        is_healthy: Whether the AI system is operating within healthy parameters.
        total_inferences: Total number of inferences performed since startup.
        avg_latency_ms: Average inference latency in milliseconds.
        anomaly_rate: Current fraction of transactions flagged as anomalous.
        models_registered: List of model names registered in the registry.
    """
    version: int
    is_healthy: bool
    total_inferences: int
    avg_latency_ms: float
    anomaly_rate: float
    models_registered: List[str]


@dataclass
class TrainingResult:
    """Result of a training trigger operation.

    Attributes:
        success: Whether the training completed successfully.
        loss: Final training loss value.
        samples_used: Number of training samples processed.
        elapsed_ms: Total training time in milliseconds.
    """
    success: bool
    loss: float
    samples_used: int
    elapsed_ms: float


class ModelService:
    """
    Central AI service API for Positronic.

    This is the main entry point for all AI operations. Other components
    (blockchain, validator, RPC) interact with the AI subsystem exclusively
    through this service, which coordinates the inference pipeline, model
    registry, and health monitor.

    Capabilities:
        - Score individual transactions for anomaly detection.
        - Score transaction batches for block validation.
        - Trigger model training updates.
        - Export/import model weights for peer-to-peer gossip sharing.
        - Monitor system health and performance metrics.
        - Manage model versions with rollback and A/B testing.

    Usage:
        >>> service = ModelService()
        >>> service.set_models(vae=my_vae, meta_ensemble=my_ensemble)
        >>> result = service.score_transaction(features, context)
        >>> print(result.score, result.latency_ms)
    """

    def __init__(self):
        self.inference = InferencePipeline()
        self.registry = ModelRegistry()
        self.health = HealthMonitor()

        # Model references (set by the blockchain integration layer)
        self._vae = None
        self._temporal_net = None
        self._graph_net = None
        self._lstm_net = None
        self._meta_ensemble = None
        self._model_version = 1

    def set_models(
        self,
        vae=None,
        temporal_net=None,
        graph_net=None,
        lstm_net=None,
        meta_ensemble=None,
        version: int = 1,
    ):
        """Configure model references used for inference.

        This method is called by the blockchain integration layer after
        models have been initialized or updated. It provides the service
        with references to the live model instances.

        Args:
            vae: VAE model instance for transaction anomaly detection.
            temporal_net: TemporalAttentionNet instance for temporal pattern
                analysis.
            graph_net: GraphAttentionNet instance for graph-based anomaly
                detection.
            lstm_net: LSTMAttentionNet instance for sequence modeling.
            meta_ensemble: MetaEnsemble instance that combines all component
                scores into a final anomaly score.
            version: Model version number to associate with these models.
        """
        self._vae = vae
        self._temporal_net = temporal_net
        self._graph_net = graph_net
        self._lstm_net = lstm_net
        self._meta_ensemble = meta_ensemble
        self._model_version = version

    def score_transaction(
        self, features: np.ndarray, context: dict
    ) -> InferenceResult:
        """Score a single transaction for anomaly detection.

        This is the primary method called during transaction validation.
        It runs the feature vector through the inference pipeline and
        records the result with the health monitor.

        Args:
            features: Numpy array of extracted transaction features.
            context: Dictionary of contextual information, which may include
                pre-computed component scores and transaction metadata.

        Returns:
            An InferenceResult containing the anomaly score, component
            scores, model version, and latency.
        """
        result = self.inference.infer_single(
            features,
            context,
            vae=self._vae,
            meta_ensemble=self._meta_ensemble,
            model_version=self._model_version,
        )
        self.health.record_inference(result.score, result.latency_ms)
        return result

    def score_batch(
        self, feature_list: List[np.ndarray], contexts: List[dict]
    ) -> List[InferenceResult]:
        """Score a batch of transactions for anomaly detection.

        Used during block validation to efficiently process all transactions
        in a proposed block.

        Args:
            feature_list: List of numpy feature arrays, one per transaction.
            contexts: List of context dictionaries, one per transaction.

        Returns:
            A list of InferenceResult objects in the same order as the inputs.
        """
        results = self.inference.infer_batch(
            feature_list,
            contexts,
            vae=self._vae,
            meta_ensemble=self._meta_ensemble,
            model_version=self._model_version,
        )
        for r in results:
            self.health.record_inference(r.score, r.latency_ms)
        return results

    def get_model_info(self) -> ModelInfo:
        """Get current model information and health status.

        Runs a health check and collects inference statistics to provide
        a comprehensive view of the AI system's current state.

        Returns:
            A ModelInfo dataclass with version, health status, inference
            counts, latency, anomaly rate, and registered model names.
        """
        health = self.health.check_health(self._model_version)
        stats = self.inference.get_stats()
        return ModelInfo(
            version=self._model_version,
            is_healthy=health.is_healthy,
            total_inferences=stats["total_inferences"],
            avg_latency_ms=stats["avg_latency_ms"],
            anomaly_rate=health.anomaly_rate,
            models_registered=list(self.registry._models.keys()),
        )

    def export_model(self, model_name: str) -> Optional[bytes]:
        """Export model weights for peer-to-peer gossip sharing.

        Retrieves the active version of the named model from the registry
        and serializes its state dictionary for network transmission.

        Args:
            model_name: Name of the model to export (e.g., 'vae').

        Returns:
            Serialized bytes of the model state, or None if the model
            is not found or serialization fails.
        """
        version = self.registry.get_active(model_name)
        if version is None:
            return None
        try:
            from positronic.ai.engine.serialization import serialize_state
            return serialize_state(version.state_dict)
        except Exception as e:
            logger.warning("Model export failed for %s: %s", model_name, e)
            return None

    def import_model(self, model_name: str, data: bytes) -> bool:
        """Import model weights received from a peer node.

        Deserializes the state dictionary from bytes received via the
        gossip protocol. The actual loading of weights into a live model
        instance is handled by the blockchain integration layer.

        Args:
            model_name: Name of the model being imported (e.g., 'vae').
            data: Serialized bytes of the model state dictionary.

        Returns:
            True if deserialization succeeded, False otherwise.
        """
        try:
            from positronic.ai.engine.serialization import deserialize_state
            state = deserialize_state(data)
            # State is deserialized successfully; actual model loading
            # is handled by the blockchain integration layer which has
            # access to the live model instances.
            return True
        except Exception as e:
            logger.warning("Model import failed for %s: %s", model_name, e)
            return False

    def get_stats(self) -> dict:
        """Get comprehensive statistics from all subsystems.

        Returns:
            Dictionary containing:
                - model_version: Current active model version.
                - inference: Inference pipeline statistics.
                - registry: Model registry statistics.
                - health: Health monitor statistics.
        """
        return {
            "model_version": self._model_version,
            "inference": self.inference.get_stats(),
            "registry": self.registry.get_stats(),
            "health": self.health.get_stats(),
        }
