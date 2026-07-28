"""Chest X-Ray Trust Gate and Safe Abstention.

Provides trust scoring, uncertainty estimation, and safe abstention
for chest X-ray predictions. This ensures the system never makes
confident predictions on unsuitable images.

Key design principles:
- Deterministic quality checks (no LLM involvement)
- Uncertainty-aware: abstain when model is uncertain
- Multi-signal trust score: image quality + prediction confidence + dataset similarity
- Fallback for medical predictions is always disabled — no kNN centroid matching
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from cnn.preprocessing import (
    check_aspect_ratio,
    check_brightness_distribution,
    check_edge_density,
    check_grayscale_ratio,
)

logger = logging.getLogger(__name__)

# Trust thresholds (conservative — better to abstain than misdiagnose)
TRUST_THRESHOLDS = {
    "image_quality": 0.4,
    "confidence": 0.75,
    "overall": 0.50,
}


@dataclass
class TrustScore:
    """Multi-signal trust score for a chest X-ray prediction."""

    # Image quality (deterministic, from pixel heuristics)
    image_quality_score: float = 0.0
    quality_checks: dict[str, float] = field(default_factory=dict)
    quality_issues: list[str] = field(default_factory=list)

    # Prediction confidence (from model softmax)
    confidence_score: float = 0.0
    entropy: float = 0.0
    margin: float = 0.0  # difference between top-2 class probabilities

    # Distribution similarity
    dataset_similarity: float | None = None

    # Overall trust
    overall_score: float = 0.0
    trust_status: str = "abstain"  # "trust", "uncertain", or "abstain"
    abstain_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_quality_score": round(self.image_quality_score, 4),
            "quality_checks": self.quality_checks,
            "quality_issues": self.quality_issues,
            "confidence_score": round(self.confidence_score, 4),
            "entropy": round(self.entropy, 4),
            "margin": round(self.margin, 4),
            "dataset_similarity": (
                round(self.dataset_similarity, 4)
                if self.dataset_similarity is not None
                else None
            ),
            "overall_score": round(self.overall_score, 4),
            "trust_status": self.trust_status,
            "abstain_reason": self.abstain_reason,
            "is_synthetic_data": False,
            "disclaimer": (
                "This trust assessment is based on automated quality, confidence, "
                "and distribution checks. It does not guarantee clinical accuracy. "
                "Always verify AI findings with a qualified clinician."
            ),
        }


def _shannon_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy from a probability distribution."""
    p = np.clip(probabilities, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _softmax_margin(probabilities: np.ndarray) -> float:
    """Compute the margin between top-2 predicted probabilities."""
    sorted_probs = np.sort(probabilities)[::-1]
    if len(sorted_probs) < 2:
        return float(sorted_probs[0])
    return float(sorted_probs[0] - sorted_probs[1])


def assess_image_quality(image: Image.Image) -> tuple[float, list[str], dict[str, float]]:
    """Assess image quality using OOD heuristics.

    Returns:
        (quality_score [0-1], issues list, detailed checks dict)
    """
    issues: list[str] = []
    checks: dict[str, float] = {}
    total_score = 0.0
    num_checks = 4

    # 1. Grayscale ratio (X-rays should be nearly monochrome)
    try:
        ratio, passes = check_grayscale_ratio(image)
        checks["grayscale_ratio"] = round(ratio, 4)
        if passes:
            total_score += min(ratio, 1.0)
            if ratio >= 0.97:
                total_score += 0.5  # bonus for strong monochrome
                num_checks += 0.5  # partial check
        else:
            issues.append(f"Not grayscale (ratio={ratio:.3f})")
    except Exception:
        checks["grayscale_ratio"] = 0.0
        issues.append("Grayscale check failed")

    # 2. Aspect ratio
    try:
        ratio, passes = check_aspect_ratio(image)
        checks["aspect_ratio"] = round(ratio, 4)
        if passes:
            # Bonus for typical chest X-ray ratio (~0.8)
            total_score += 1.0 - min(abs(ratio - 0.8), 0.5) / 0.5 * 0.4
            num_checks += 1.0
        else:
            issues.append(f"Abnormal aspect ratio ({ratio:.2f})")
    except Exception:
        checks["aspect_ratio"] = 0.0
        issues.append("Aspect ratio check failed")

    # 3. Edge density (X-rays have soft edges)
    try:
        density, passes = check_edge_density(image)
        checks["edge_density"] = round(density, 4)
        if passes:
            total_score += 1.0 - density * 2  # lower edge density = better
            num_checks += 1.0
        else:
            issues.append(f"High edge density ({density:.3f})")
    except Exception:
        checks["edge_density"] = 0.0
        issues.append("Edge density check failed")

    # 4. Brightness distribution / histogram spread
    try:
        spread, passes = check_brightness_distribution(image)
        checks["histogram_spread"] = round(spread, 4)
        if passes:
            total_score += min(spread * 3, 1.0)
            num_checks += 1.0
        else:
            issues.append(f"Narrow histogram spread ({spread:.4f})")
    except Exception:
        checks["histogram_spread"] = 0.0
        issues.append("Histogram check failed")

    quality_score = total_score / max(num_checks, 1.0)
    return min(quality_score, 1.0), issues, checks


def estimate_dataset_similarity(
    image_array: np.ndarray,
    reference_stats: dict[str, float] | None = None,
) -> float:
    """Estimate how similar the image is to the training dataset distribution.

    Uses simple feature-level distance to dataset centroids.
    This is NOT a kNN classifier — it produces a similarity score only.

    Args:
        image_array: Preprocessed image (224, 224, 3) float32 [0,1]
        reference_stats: Optional pre-computed dataset statistics.
            If None, returns a neutral score.

    Returns:
        Similarity score [0-1], higher = more similar to training data
    """
    if reference_stats is None:
        return 0.5  # neutral score when no reference available

    # Compute simple feature statistics for this image
    gray = image_array.mean(axis=2)

    # Compute features comparable to fallback._features_from_array
    stats = {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "gradient_mean": float(np.abs(np.gradient(gray)[0]).mean()),
        "left_region": float(gray[44:180, 22:102].mean()),
        "right_region": float(gray[44:180, 122:202].mean()),
    }

    # Compute cosine similarity to reference
    ref_mean = reference_stats.get("mean", 0.5)
    ref_std = reference_stats.get("std", 0.2)

    mean_diff = abs(stats["mean"] - ref_mean)
    std_diff = abs(stats["std"] - ref_std)

    # Higher similarity = closer to reference distribution
    similarity = 1.0 - min((mean_diff * 2 + std_diff * 2), 1.0)
    return float(np.clip(similarity, 0.0, 1.0))


def compute_trust(
    probabilities: np.ndarray,
    image: Image.Image | None = None,
    image_array: np.ndarray | None = None,
    image_quality_score: float | None = None,
    reference_stats: dict[str, float] | None = None,
) -> TrustScore:
    """Compute a multi-signal trust score for a chest X-ray prediction.

    This is the main entry point. It combines:
    1. Image quality (from pixel heuristics)
    2. Confidence (from prediction probabilities)
    3. Dataset similarity (distribution alignment)

    Args:
        probabilities: Model softmax output, shape (n_classes,)
        image: PIL Image for quality assessment
        image_array: Preprocessed array for dataset similarity
        image_quality_score: Optional pre-computed quality score
        reference_stats: Optional dataset statistics for similarity

    Returns:
        TrustScore with overall assessment and abstention decision
    """
    trust = TrustScore()

    # 1. Image quality
    if image is not None and image_quality_score is None:
        quality, issues, checks = assess_image_quality(image)
        trust.image_quality_score = quality
        trust.quality_checks = checks
        trust.quality_issues = issues
    elif image_quality_score is not None:
        trust.image_quality_score = image_quality_score
    else:
        trust.image_quality_score = 0.5  # neutral

    # 2. Prediction confidence
    prob_np = np.asarray(probabilities, dtype=np.float32)
    max_prob = float(prob_np.max())
    trust.confidence_score = max_prob
    trust.entropy = _shannon_entropy(prob_np)
    trust.margin = _softmax_margin(prob_np)

    # Normalize entropy to [0,1] for binary classification
    max_entropy = math.log(len(prob_np))
    normalized_entropy = trust.entropy / max_entropy if max_entropy > 0 else 0.0

    # 3. Dataset similarity
    if image_array is not None:
        trust.dataset_similarity = estimate_dataset_similarity(
            image_array, reference_stats
        )
    else:
        trust.dataset_similarity = 0.5  # neutral

    # 4. Overall trust score
    quality_weight = 0.30
    confidence_weight = 0.45
    similarity_weight = 0.25

    # Confidence signal: high confidence + low entropy = trustworthy
    confidence_signal = trust.confidence_score * 0.7 + (1 - normalized_entropy) * 0.3

    trust.overall_score = (
        trust.image_quality_score * quality_weight
        + confidence_signal * confidence_weight
        + (trust.dataset_similarity or 0.5) * similarity_weight
    )

    # 5. Trust status and abstention
    if trust.overall_score < TRUST_THRESHOLDS["overall"]:
        trust.trust_status = "abstain"
        reasons = []
        if trust.image_quality_score < TRUST_THRESHOLDS["image_quality"]:
            reasons.append("poor image quality")
        if trust.confidence_score < TRUST_THRESHOLDS["confidence"]:
            reasons.append("low model confidence")
        if trust.dataset_similarity is not None and trust.dataset_similarity < 0.3:
            reasons.append("image differs from training data distribution")
        trust.abstain_reason = (
            f"Prediction not suitable for clinical use: "
            f"{'; '.join(reasons)}. "
            f"Overall trust score: {trust.overall_score:.2f}/1.0. "
            f"A higher-quality chest X-ray or retake may help."
        )
    elif trust.overall_score >= 0.7 and trust.confidence_score >= 0.85:
        trust.trust_status = "trust"
    else:
        trust.trust_status = "uncertain"

    return trust


# Reference dataset statistics (pre-computed from chest X-ray training set)
# These are used when the dataset is not available at inference time
DEFAULT_REFERENCE_STATS: dict[str, float] = {
    "mean": 0.518,
    "std": 0.207,
    "gradient_mean": 0.042,
    "left_region": 0.465,
    "right_region": 0.512,
}
