"""
SHAP explainability for the Heart Disease ANN.

Uses KernelExplainer (model-agnostic) to compute per-feature SHAP values
for a given prediction.  Generates a bar chart PNG showing feature
contributions.

KernelExplainer is chosen over DeepExplainer for Keras 3 compatibility —
DeepExplainer relies on TF1/TF2 session internals that can break with
newer Keras versions.
"""
from __future__ import annotations

import io
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import keras

REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))

# Module-level cache for model + background data
_model: Optional[keras.Model] = None
_explainer: Optional[shap.KernelExplainer] = None
_feature_names: Optional[list[str]] = None


def _load_model() -> keras.Model:
    """Load the trained ANN from registry (cached)."""
    global _model
    if _model is None:
        model_path = REGISTRY_DIR / "ann_heart_risk.h5"
        _model = keras.saving.load_model(str(model_path))
    return _model


def _load_feature_names() -> list[str]:
    """Load feature names saved during preprocessing."""
    global _feature_names
    if _feature_names is None:
        with open(REGISTRY_DIR / "ann_feature_names.pkl", "rb") as f:
            _feature_names = pickle.load(f)
    return _feature_names


def _get_explainer(background_data: np.ndarray | None = None) -> shap.KernelExplainer:
    """Create or return cached SHAP KernelExplainer.

    Uses a small background sample (50 points from training data or
    a synthetic zero-vector if no training data is available).
    """
    global _explainer
    if _explainer is None:
        model = _load_model()

        if background_data is None:
            # Create a minimal synthetic background (zeros)
            n_features = model.input_shape[-1]
            background_data = np.zeros((10, n_features), dtype=np.float32)

        # Use kmeans summary of background for efficiency
        if len(background_data) > 50:
            background_summary = shap.kmeans(background_data, 50)
        else:
            background_summary = background_data

        _explainer = shap.KernelExplainer(
            lambda x: model.predict(x, verbose=0).ravel(),
            background_summary,
        )
    return _explainer


def explain(
    input_array: np.ndarray,
    background_data: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute SHAP values for a single input.

    Parameters
    ----------
    input_array : np.ndarray
        Shape (1, n_features) — preprocessed input.
    background_data : np.ndarray, optional
        Training data sample for the explainer background.

    Returns
    -------
    dict mapping feature name → SHAP value.
    """
    explainer = _get_explainer(background_data)
    feature_names = _load_feature_names()

    shap_values = explainer.shap_values(input_array, nsamples=100)

    # shap_values may be a list (one per output) or a 2D array
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    if sv.ndim == 2:
        sv = sv[0]

    return {name: float(val) for name, val in zip(feature_names, sv)}


def generate_shap_chart(
    shap_values_dict: dict[str, float],
    save_path: str | None = None,
) -> bytes:
    """Generate a horizontal bar chart of SHAP values.

    Returns PNG image bytes.  Optionally saves to disk at save_path.
    """
    # Sort by absolute value, show top 10
    sorted_items = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ff6b6b" if v > 0 else "#00d4aa" for v in values]
    ax.barh(range(len(names)), values, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
    ax.set_title("Feature Contributions — Heart Disease Risk", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="#555", linewidth=0.8, linestyle="--")

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_bytes)

    return img_bytes
