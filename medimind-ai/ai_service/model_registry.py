import json
from pathlib import Path

import joblib

DISEASES = ("diabetes", "heart", "kidney", "stroke")
BASE_DIR = Path(__file__).resolve().parent


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_joblib(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def load_all_models() -> dict:
    registry = {}
    for disease in DISEASES:
        model_dir = BASE_DIR / "ml_models" / disease
        registry[disease] = {
            "model": _load_joblib(model_dir / "model.joblib"),
            "scaler": _load_joblib(model_dir / "scaler.joblib"),
            "feature_columns": _load_json(model_dir / "feature_columns.json", []),
            "shap_explainer": _load_joblib(model_dir / "shap_explainer.joblib"),
        }
    return registry
