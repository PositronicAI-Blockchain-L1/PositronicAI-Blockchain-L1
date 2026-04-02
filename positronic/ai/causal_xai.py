"""
Positronic - Causal XAI v2 (Phase 2.5.C)

Enhanced explainability for AI transaction scoring.
Provides feature attribution (permutation importance), counterfactual
reasoning, and human-readable narratives in English and Persian.

This module operates alongside the existing ``_generate_explanation()``
method in ``AIValidationGate`` (meta_model.py) without modifying it.
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from positronic.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default feature names (subset of the 35 features in feature_extractor.py
# that are most interpretable for end-user explanations).
# ---------------------------------------------------------------------------
_DEFAULT_FEATURE_NAMES: List[str] = [
    "value_log",
    "gas_price_ratio",
    "sender_nonce",
    "sender_reputation",
    "is_contract",
    "mempool_ratio",
    "gas_limit_ratio",
]

# Valid status strings
_VALID_STATUSES = {"ACCEPTED", "QUARANTINED", "REJECTED"}

# Accept threshold for counterfactual reasoning: score below this -> ACCEPTED
_ACCEPT_THRESHOLD = 0.845

# ---------------------------------------------------------------------------
# Per-model explanation templates
# ---------------------------------------------------------------------------
_TEMPLATES_EN = {
    "tad": {
        "high": "Transaction anomaly score: {score:.2f} — high deviation from learned patterns",
        "normal": "Transaction anomaly score: {score:.2f} — normal deviation from learned patterns",
    },
    "msad": {
        "high": "MEV analysis: {score:.2f} — detected sandwich or frontrun patterns",
        "normal": "MEV analysis: {score:.2f} — no sandwich or frontrun patterns",
    },
    "scra": {
        "high": "Contract risk: {score:.2f} — high risk bytecode patterns",
        "normal": "Contract risk: {score:.2f} — low risk bytecode patterns",
    },
    "esg": {
        "high": "Network stability: {score:.2f} — unstable market conditions",
        "normal": "Network stability: {score:.2f} — stable market conditions",
    },
}

_TEMPLATES_FA = {
    "tad": {
        "high": "\u0627\u0645\u062a\u06cc\u0627\u0632 \u0646\u0627\u0647\u0646\u062c\u0627\u0631\u06cc \u062a\u0631\u0627\u06a9\u0646\u0634: {score:.2f} \u2014 \u0628\u0627\u0644\u0627 \u0627\u0646\u062d\u0631\u0627\u0641 \u0627\u0632 \u0627\u0644\u06af\u0648\u0647\u0627\u06cc \u06cc\u0627\u062f\u06af\u0631\u0641\u062a\u0647\u200c\u0634\u062f\u0647",
        "normal": "\u0627\u0645\u062a\u06cc\u0627\u0632 \u0646\u0627\u0647\u0646\u062c\u0627\u0631\u06cc \u062a\u0631\u0627\u06a9\u0646\u0634: {score:.2f} \u2014 \u0639\u0627\u062f\u06cc \u0627\u0646\u062d\u0631\u0627\u0641 \u0627\u0632 \u0627\u0644\u06af\u0648\u0647\u0627\u06cc \u06cc\u0627\u062f\u06af\u0631\u0641\u062a\u0647\u200c\u0634\u062f\u0647",
    },
    "msad": {
        "high": "\u062a\u062d\u0644\u06cc\u0644 MEV: {score:.2f} \u2014 \u0634\u0646\u0627\u0633\u0627\u06cc\u06cc \u0627\u0644\u06af\u0648\u06cc \u0633\u0627\u0646\u062f\u0648\u06cc\u0686 \u06cc\u0627 \u0641\u0631\u0627\u0646\u062a\u200c\u0631\u0627\u0646",
        "normal": "\u062a\u062d\u0644\u06cc\u0644 MEV: {score:.2f} \u2014 \u0639\u062f\u0645 \u0634\u0646\u0627\u0633\u0627\u06cc\u06cc \u0627\u0644\u06af\u0648\u06cc \u0633\u0627\u0646\u062f\u0648\u06cc\u0686 \u06cc\u0627 \u0641\u0631\u0627\u0646\u062a\u200c\u0631\u0627\u0646",
    },
    "scra": {
        "high": "\u0631\u06cc\u0633\u06a9 \u0642\u0631\u0627\u0631\u062f\u0627\u062f: {score:.2f} \u2014 \u0627\u0644\u06af\u0648\u0647\u0627\u06cc \u0628\u0627\u06cc\u062a\u200c\u06a9\u062f \u067e\u0631\u062e\u0637\u0631",
        "normal": "\u0631\u06cc\u0633\u06a9 \u0642\u0631\u0627\u0631\u062f\u0627\u062f: {score:.2f} \u2014 \u0627\u0644\u06af\u0648\u0647\u0627\u06cc \u0628\u0627\u06cc\u062a\u200c\u06a9\u062f \u06a9\u0645\u200c\u062e\u0637\u0631",
    },
    "esg": {
        "high": "\u062b\u0628\u0627\u062a \u0634\u0628\u06a9\u0647: {score:.2f} \u2014 \u0634\u0631\u0627\u06cc\u0637 \u0628\u0627\u0632\u0627\u0631 \u0646\u0627\u067e\u0627\u06cc\u062f\u0627\u0631",
        "normal": "\u062b\u0628\u0627\u062a \u0634\u0628\u06a9\u0647: {score:.2f} \u2014 \u0634\u0631\u0627\u06cc\u0637 \u0628\u0627\u0632\u0627\u0631 \u067e\u0627\u06cc\u062f\u0627\u0631",
    },
}

_SUMMARY_EN = {
    "ACCEPTED": "Transaction accepted with risk score {score:.2f} — within safe operational bounds.",
    "QUARANTINED": "Transaction quarantined with risk score {score:.2f} — requires further review.",
    "REJECTED": "Transaction rejected with risk score {score:.2f} — exceeds safety thresholds.",
}

_SUMMARY_FA = {
    "ACCEPTED": "\u062a\u0631\u0627\u06a9\u0646\u0634 \u0628\u0627 \u0627\u0645\u062a\u06cc\u0627\u0632 \u0631\u06cc\u0633\u06a9 {score:.2f} \u067e\u0630\u06cc\u0631\u0641\u062a\u0647 \u0634\u062f.",
    "QUARANTINED": "\u062a\u0631\u0627\u06a9\u0646\u0634 \u0628\u0627 \u0627\u0645\u062a\u06cc\u0627\u0632 \u0631\u06cc\u0633\u06a9 {score:.2f} \u0642\u0631\u0646\u0637\u06cc\u0646\u0647 \u0634\u062f.",
    "REJECTED": "\u062a\u0631\u0627\u06a9\u0646\u0634 \u0628\u0627 \u0627\u0645\u062a\u06cc\u0627\u0632 \u0631\u06cc\u0633\u06a9 {score:.2f} \u0631\u062f \u0634\u062f.",
}

# Score threshold for high/normal classification in templates
_MODEL_HIGH_THRESHOLD = 0.5


class CausalExplainer:
    """Enhanced XAI engine for AI-validated transaction scoring.

    Provides three capabilities:
    1. **Feature Attribution** — permutation importance to identify the
       top features driving a risk score.
    2. **Counterfactual Reasoning** — greedy search for minimal feature
       changes that would flip the transaction status.
    3. **Human-Readable Narrative** — per-model explanations in English
       and Persian (Farsi).

    Usage::

        explainer = CausalExplainer()
        importance = explainer.compute_feature_importance(features, score_fn)
        cf = explainer.generate_counterfactual(features, score_fn)
        narrative = explainer.explain(features, model_scores, "QUARANTINED")
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names: List[str] = (
            list(feature_names) if feature_names is not None
            else list(_DEFAULT_FEATURE_NAMES)
        )

    # ------------------------------------------------------------------
    # 1. Feature Attribution (Permutation Importance)
    # ------------------------------------------------------------------

    def compute_feature_importance(
        self,
        features: dict,
        score_fn: Callable[[dict], float],
        n_permutations: int = 10,
    ) -> List[Tuple[str, float]]:
        """Compute permutation-based feature importance.

        For each feature in *features* that is also in ``self.feature_names``,
        the feature value is shuffled (perturbed) and the score function is
        re-evaluated.  The importance is the mean absolute change in score
        across *n_permutations* trials.

        Args:
            features: Feature name -> value mapping.
            score_fn: Callable that maps a feature dict to a float risk score.
            n_permutations: Number of random perturbations per feature.

        Returns:
            Sorted list of ``(feature_name, importance_score)`` tuples,
            limited to the top 3 features, sorted descending by importance.
        """
        if not features:
            return []

        rng = np.random.RandomState(42)  # deterministic for reproducibility
        baseline_score = score_fn(features)

        importance: Dict[str, float] = {}
        active_features = [f for f in self.feature_names if f in features]

        for fname in active_features:
            total_impact = 0.0
            original_val = features[fname]

            # Zero-valued features have zero variance — no perturbation needed
            if original_val == 0.0 or original_val == 0:
                importance[fname] = 0.0
                continue

            for _ in range(n_permutations):
                perturbed = dict(features)
                # Perturb: add Gaussian noise scaled to the value magnitude
                noise = rng.normal(0, abs(original_val) * 0.5)
                perturbed[fname] = original_val + noise
                perturbed_score = score_fn(perturbed)
                total_impact += abs(perturbed_score - baseline_score)

            importance[fname] = total_impact / n_permutations

        # Sort descending and take top 3
        sorted_importance = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_importance[:3]

    # ------------------------------------------------------------------
    # 2. Counterfactual Reasoning
    # ------------------------------------------------------------------

    def generate_counterfactual(
        self,
        features: dict,
        score_fn: Callable[[dict], float],
        target_status: str = "ACCEPTED",
    ) -> dict:
        """Generate a counterfactual explanation via greedy feature adjustment.

        Finds the minimal set of feature changes that would bring the
        score below the acceptance threshold (for ``target_status="ACCEPTED"``).

        Args:
            features: Current feature values.
            score_fn: Callable producing a risk score from features.
            target_status: Desired status (currently only ``"ACCEPTED"``
                triggers score-lowering search).

        Returns:
            Dict with keys:
            - ``changes``: ``{feature: (old_val, new_val)}`` for modified features
            - ``new_score``: score after applying changes
            - ``achievable``: bool — whether the target status was reached
        """
        current_score = score_fn(features)

        # If already at target, no changes needed
        if target_status == "ACCEPTED" and current_score < _ACCEPT_THRESHOLD:
            return {"changes": {}, "new_score": current_score, "achievable": True}

        # Compute importance to know which features to adjust first
        active_features = [f for f in self.feature_names if f in features]
        if not active_features:
            return {"changes": {}, "new_score": current_score, "achievable": False}

        # Get full importance ranking (not just top 3)
        rng = np.random.RandomState(42)
        baseline_score = current_score
        importance: Dict[str, float] = {}

        for fname in active_features:
            total_impact = 0.0
            original_val = features[fname]
            for _ in range(5):  # fewer permutations for speed
                perturbed = dict(features)
                noise = rng.normal(0, max(abs(original_val) * 0.5, 0.1))
                perturbed[fname] = original_val + noise
                perturbed_score = score_fn(perturbed)
                total_impact += abs(perturbed_score - baseline_score)
            importance[fname] = total_impact / 5

        # Sort by importance descending — adjust highest-impact features first
        sorted_features = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )

        modified = dict(features)
        changes: Dict[str, Tuple[float, float]] = {}

        for fname, _ in sorted_features:
            old_val = features[fname]
            # Try reducing the feature value toward 0 (greedy)
            for factor in [0.0, 0.1, 0.25, 0.5]:
                trial = dict(modified)
                new_val = old_val * factor
                trial[fname] = new_val
                new_score = score_fn(trial)
                if new_score < _ACCEPT_THRESHOLD:
                    if old_val != new_val:
                        changes[fname] = (old_val, new_val)
                    modified = trial
                    return {
                        "changes": changes,
                        "new_score": new_score,
                        "achievable": True,
                    }
            # Even if not yet achieved, apply the best reduction
            best_val = old_val * 0.0
            if old_val != best_val:
                changes[fname] = (old_val, best_val)
            modified[fname] = best_val

        # After adjusting all features, check if we reached target
        final_score = score_fn(modified)
        achievable = final_score < _ACCEPT_THRESHOLD

        return {
            "changes": changes,
            "new_score": final_score,
            "achievable": achievable,
        }

    # ------------------------------------------------------------------
    # 3. Human-Readable Narrative
    # ------------------------------------------------------------------

    def explain(
        self,
        features: dict,
        scores: dict,
        status: str,
        language: str = "en",
    ) -> dict:
        """Generate a complete human-readable explanation.

        Args:
            features: Feature name -> value mapping.
            scores: Per-model scores ``{"tad": float, "msad": float, ...}``.
            status: Transaction status (``"ACCEPTED"``, ``"QUARANTINED"``,
                or ``"REJECTED"``).
            language: ``"en"`` for English, ``"fa"`` for Persian (Farsi).
                Unknown languages fall back to English.

        Returns:
            Dict with keys:
            - ``summary``: one-sentence overall explanation
            - ``per_model``: ``{model_name: explanation_string}``
            - ``top_features``: top 3 feature attributions (empty if no
              features provided)
            - ``counterfactual``: counterfactual dict or ``None`` (only
              for QUARANTINED / REJECTED)
            - ``language``: actual language used (after fallback)
        """
        start = time.perf_counter()

        # Validate status
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Must be one of: {_VALID_STATUSES}"
            )

        # Language fallback
        if language not in ("en", "fa"):
            language = "en"

        # Select templates
        templates = _TEMPLATES_FA if language == "fa" else _TEMPLATES_EN
        summaries = _SUMMARY_FA if language == "fa" else _SUMMARY_EN

        # Compute overall score from model scores for summary
        overall_score = sum(scores.values()) / max(len(scores), 1)

        # Summary
        summary = summaries[status].format(score=overall_score)

        # Per-model explanations
        per_model: Dict[str, str] = {}
        for model_key in ("tad", "msad", "scra", "esg"):
            model_score = scores.get(model_key, 0.0)
            level = "high" if model_score >= _MODEL_HIGH_THRESHOLD else "normal"
            template = templates.get(model_key, {}).get(level, "")
            per_model[model_key] = template.format(score=model_score)

        # Top features (using a simple score function derived from model scores)
        top_features: List[Tuple[str, float]] = []
        if features:
            def _internal_score_fn(feats: dict) -> float:
                """Lightweight score function for attribution."""
                s = 0.0
                s += feats.get("value_log", 0) * 0.05
                s += feats.get("gas_price_ratio", 0) * 0.1
                s -= feats.get("sender_reputation", 0) * 0.2
                s += feats.get("is_contract", 0) * 0.15
                s += feats.get("mempool_ratio", 0) * 0.05
                s += feats.get("gas_limit_ratio", 0) * 0.02
                s += feats.get("sender_nonce", 0) * 0.001
                return max(0.0, min(1.0, s))

            top_features = self.compute_feature_importance(
                features, _internal_score_fn, n_permutations=5
            )

        # Counterfactual (only for non-accepted)
        counterfactual = None
        if status in ("QUARANTINED", "REJECTED") and features:
            def _cf_score_fn(feats: dict) -> float:
                s = 0.0
                s += feats.get("value_log", 0) * 0.05
                s += feats.get("gas_price_ratio", 0) * 0.1
                s -= feats.get("sender_reputation", 0) * 0.2
                s += feats.get("is_contract", 0) * 0.15
                s += feats.get("mempool_ratio", 0) * 0.05
                s += feats.get("gas_limit_ratio", 0) * 0.02
                s += feats.get("sender_nonce", 0) * 0.001
                return max(0.0, min(1.0, s))

            counterfactual = self.generate_counterfactual(
                features, _cf_score_fn, target_status="ACCEPTED"
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 50:
            logger.warning(
                "CausalExplainer.explain took %.1fms (>50ms limit)", elapsed_ms
            )

        return {
            "summary": summary,
            "per_model": per_model,
            "top_features": top_features,
            "counterfactual": counterfactual,
            "language": language,
        }
