"""
Positronic - AI Agent Quality Scoring (Phase 29)
Evaluates agent output quality using PoNC-derived scoring primitives.

Two modes:
  1. Heuristic (default) — deterministic, consensus-safe, no ML dependency
  2. PoNC-assisted — uses AIValidationGate ensemble for richer scoring
     (enabled when an AIValidationGate instance is provided)
"""

import hashlib
from typing import Optional

from positronic.constants import (
    AI_SCORE_SCALE,
    AGENT_QUALITY_THRESHOLD,
)


def score_agent_output(
    result_data: str,
    input_data: str,
    agent_category: int,
    ai_gate=None,
) -> int:
    """Score an agent's output quality.

    Returns a quality score in basis points (0-10000).
    Higher is better.

    When *ai_gate* (an AIValidationGate instance) is provided, the PoNC
    ensemble contributes to the final score.  Otherwise falls back to
    deterministic heuristics that are safe for cross-node consensus.
    """
    heuristic = _heuristic_score(result_data, input_data, agent_category)

    if ai_gate is not None:
        ponc = _ponc_score(result_data, input_data, ai_gate)
        # Weighted blend: 60% PoNC, 40% heuristic
        combined = int(ponc * 0.6 + heuristic * 0.4)
        return max(0, min(AI_SCORE_SCALE, combined))

    return heuristic


# ---- Heuristic scoring (deterministic, consensus-safe) ----

def _heuristic_score(result_data: str, input_data: str, agent_category: int) -> int:
    """Deterministic scoring using content heuristics."""
    score = 5000  # Start neutral

    # Factor 1: Result length (non-trivial output)
    result_len = len(result_data)
    if result_len == 0:
        return 0  # Empty result = worst score
    if result_len < 10:
        score -= 2000  # Too short
    elif result_len < 100:
        score += 500
    elif result_len < 5000:
        score += 1500  # Good detail
    else:
        score += 1000  # Very long, slightly less (might be padding)

    # Factor 2: Input/output relevance (simple overlap check)
    input_words = set(input_data.lower().split())
    result_words = set(result_data.lower().split())
    if input_words:
        overlap = len(input_words & result_words) / len(input_words)
        score += int(overlap * 1500)  # Up to +1500 for keyword match

    # Factor 3: Structure (JSON-like or well-formatted output)
    if result_data.strip().startswith("{") or result_data.strip().startswith("["):
        score += 500  # Structured output bonus

    # Factor 4: Content diversity (penalize repetitive padding)
    unique_words = len(result_words)
    total_words = len(result_data.split())
    if total_words > 20:
        diversity = unique_words / total_words
        if diversity < 0.3:
            score -= 1000  # Too repetitive
        elif diversity > 0.6:
            score += 300   # Good diversity

    # Clamp to valid range
    return max(0, min(AI_SCORE_SCALE, score))


# ---- PoNC-assisted scoring ----

def _ponc_score(result_data: str, input_data: str, ai_gate) -> int:
    """Use AIValidationGate ensemble for quality scoring.

    Creates a synthetic transaction-like feature set from the agent output
    and runs it through the PoNC anomaly detector. A LOW risk score from
    PoNC maps to a HIGH quality score here (inverted).
    """
    try:
        # Use the anomaly detector's feature extractor to get a risk signal
        # from the result content. We hash the result to create a
        # deterministic "transaction fingerprint".
        result_hash = hashlib.sha512(result_data.encode()).digest()
        input_hash = hashlib.sha512(input_data.encode()).digest()

        # Extract byte-level features for the anomaly detector
        features = []
        for i in range(0, min(64, len(result_hash)), 4):
            features.append(int.from_bytes(result_hash[i:i+4], "big") / (2**32))
        for i in range(0, min(32, len(input_hash)), 4):
            features.append(int.from_bytes(input_hash[i:i+4], "big") / (2**32))

        # Pad/truncate to expected feature dimension
        while len(features) < 24:
            features.append(0.5)
        features = features[:24]

        # Use the anomaly detector if available
        if hasattr(ai_gate, 'anomaly_detector'):
            try:
                risk = ai_gate.anomaly_detector.compute_anomaly_score(features)
                # Invert: low risk (0.0) → high quality (10000)
                quality = int((1.0 - risk) * AI_SCORE_SCALE)
                return max(0, min(AI_SCORE_SCALE, quality))
            except Exception:
                pass

        # Fallback: derive score from hash entropy
        entropy_score = sum(features[:16]) / 16.0
        return int(entropy_score * AI_SCORE_SCALE)

    except Exception:
        # If PoNC fails, return neutral score
        return 5000


# ---- Threshold and trend helpers ----

def meets_quality_threshold(score: int) -> bool:
    """Check if a quality score meets the minimum threshold for rewards."""
    return score >= AGENT_QUALITY_THRESHOLD


def calculate_quality_trend(scores: list) -> str:
    """Determine quality trend from recent scores.

    Returns: 'improving', 'stable', or 'declining'.
    """
    if len(scores) < 3:
        return "stable"

    recent = scores[-3:]
    older = scores[-6:-3] if len(scores) >= 6 else scores[:3]

    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)

    diff = recent_avg - older_avg
    if diff > 200:  # Improved by >2%
        return "improving"
    elif diff < -200:  # Declined by >2%
        return "declining"
    return "stable"
