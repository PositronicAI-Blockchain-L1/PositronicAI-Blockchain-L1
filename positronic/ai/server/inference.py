"""
Positronic - Inference Pipeline
Efficient inference with batching, caching, and deterministic mode for consensus.

This module provides the core inference pipeline used by the AI server to score
transactions. It supports result caching for repeated inputs, batch processing
for throughput, and deterministic execution for consensus agreement across nodes.

Dependencies:
    - numpy: Numerical computation for feature processing
    - positronic.ai.models.vae.VAE: Variational autoencoder for anomaly scoring
    - positronic.ai.models.meta_ensemble.MetaEnsemble: Ensemble scoring across components
"""

import logging
import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass

from positronic.constants import AI_CACHE_SIZE


@dataclass
class InferenceResult:
    """Result of a single inference operation.

    Attributes:
        score: Final anomaly score in [0.0, 1.0] where higher means more anomalous.
        component_scores: Individual scores from each detection component
            (tad, msad, scra, esg).
        model_version: Version number of the model that produced this result.
        latency_ms: Time taken for inference in milliseconds.
        cached: Whether this result was served from cache.
    """
    score: float
    component_scores: Dict[str, float]
    model_version: int
    latency_ms: float
    cached: bool = False


class InferencePipeline:
    """
    Manages inference with caching, batching, and latency tracking.

    The pipeline sits between the ModelService API and the underlying AI models,
    providing performance optimizations that are transparent to callers.

    Features:
        - Result caching: Identical feature vectors return cached results without
          re-running inference, using SHA-256 hashing for cache keys.
        - Batch accumulation: Multiple feature vectors can be processed together
          for improved throughput.
        - Deterministic mode: When the same features are provided, the same score
          is returned (via cache), which is essential for consensus agreement.
        - Latency tracking: All inference times are recorded for health monitoring.

    Args:
        cache_size: Maximum number of cached results to retain. Oldest entries
            are evicted when the limit is exceeded (LRU policy).
    """

    def __init__(self, cache_size: int = AI_CACHE_SIZE):
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._total_inferences = 0
        self._cache_hits = 0
        self._total_latency = 0.0
        self._batch_queue: List[Tuple[np.ndarray, dict]] = []

    def _cache_key(self, features: np.ndarray) -> str:
        """Compute a deterministic cache key from a feature vector.

        Uses SHA-256 over the raw byte representation of the array to ensure
        that identical numerical values always produce the same key.

        Args:
            features: Input feature array.

        Returns:
            A 16-character hexadecimal hash string.
        """
        return hashlib.sha256(features.tobytes()).hexdigest()[:16]

    def infer_single(
        self,
        features: np.ndarray,
        context: dict,
        vae=None,
        meta_ensemble=None,
        model_version: int = 1,
    ) -> InferenceResult:
        """Run inference on a single feature vector.

        This is the primary inference entry point. It checks the cache first,
        then runs the VAE anomaly scorer and meta-ensemble if available, falling
        back to a static weighted average if the ensemble is not ready.

        Args:
            features: Numpy array of transaction features.
            context: Dictionary of contextual information that may include
                pre-computed component scores ('msad_score', 'scra_score',
                'esg_score') and other metadata.
            vae: Optional VAE model instance for anomaly scoring. If None,
                a default TAD score of 0.3 is used.
            meta_ensemble: Optional MetaEnsemble instance for final scoring.
                If None or not ready, a static weighted average is used.
            model_version: Version number to tag the result with.

        Returns:
            An InferenceResult containing the final score, component scores,
            model version, latency, and cache status.
        """
        start = time.time()

        # Check cache for previously computed result
        cache_key = self._cache_key(features)
        if cache_key in self._cache:
            self._cache_hits += 1
            self._total_inferences += 1
            cached_result = self._cache[cache_key]
            cached_result.cached = True
            return cached_result

        # Run VAE anomaly scoring (Transaction Anomaly Detection)
        tad_score = 0.3  # default fallback
        if vae is not None:
            try:
                tad_score = vae.compute_anomaly_score(features)
            except Exception as e:
                logging.getLogger(__name__).warning("VAE anomaly scoring failed: %s", e)
                tad_score = 0.3

        # Assemble component scores from context and computed values
        component_scores = {
            "tad": tad_score,
            "msad": context.get("msad_score", 0.0),
            "scra": context.get("scra_score", 0.0),
            "esg": context.get("esg_score", 0.0),
        }

        # Compute final score via meta-ensemble or static fallback
        if meta_ensemble is not None and meta_ensemble.is_ready:
            final_score = meta_ensemble.score(component_scores, context)
        else:
            # Static weighted average fallback when ensemble is unavailable
            weights = {"tad": 0.30, "msad": 0.25, "scra": 0.25, "esg": 0.20}
            final_score = sum(component_scores[k] * weights[k] for k in weights)

        latency = (time.time() - start) * 1000
        self._total_latency += latency
        self._total_inferences += 1

        result = InferenceResult(
            score=float(np.clip(final_score, 0.0, 1.0)),
            component_scores=component_scores,
            model_version=model_version,
            latency_ms=latency,
        )

        # Update cache with LRU eviction
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return result

    def infer_batch(
        self,
        feature_list: List[np.ndarray],
        contexts: List[dict],
        **kwargs,
    ) -> List[InferenceResult]:
        """Run inference on a batch of feature vectors.

        Processes each feature vector individually through infer_single,
        benefiting from caching for any repeated inputs within the batch.

        Args:
            feature_list: List of numpy feature arrays, one per transaction.
            contexts: List of context dictionaries, one per transaction.
                Must be the same length as feature_list.
            **kwargs: Additional keyword arguments forwarded to infer_single
                (e.g., vae, meta_ensemble, model_version).

        Returns:
            A list of InferenceResult objects in the same order as the inputs.

        Raises:
            ValueError: If feature_list and contexts have different lengths.
        """
        if len(feature_list) != len(contexts):
            raise ValueError(
                f"feature_list length ({len(feature_list)}) must match "
                f"contexts length ({len(contexts)})"
            )
        return [
            self.infer_single(f, c, **kwargs)
            for f, c in zip(feature_list, contexts)
        ]

    def clear_cache(self):
        """Clear the entire inference cache.

        This should be called after model updates to ensure stale results
        are not served.
        """
        self._cache.clear()

    def get_stats(self) -> dict:
        """Get inference pipeline statistics.

        Returns:
            Dictionary containing:
                - total_inferences: Total number of inference calls.
                - cache_hits: Number of cache hits.
                - cache_hit_rate: Ratio of cache hits to total inferences.
                - cache_size: Current number of cached entries.
                - avg_latency_ms: Average inference latency in milliseconds.
        """
        avg_latency = self._total_latency / max(self._total_inferences, 1)
        return {
            "total_inferences": self._total_inferences,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(self._total_inferences, 1),
            "cache_size": len(self._cache),
            "avg_latency_ms": avg_latency,
        }
