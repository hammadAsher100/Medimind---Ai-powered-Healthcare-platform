import json
import os
import threading
from pathlib import Path

DISEASES = ("diabetes", "heart", "kidney", "stroke")

if os.environ.get("DISABLE_ML", "False").lower() == "true":
    joblib = None
else:
    import joblib

_cache = {}
_lock = threading.Lock()

def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _load_joblib(path: Path):
    if joblib is None:
        return None
    if not path.exists():
        return None
    return joblib.load(path)

def get_disease_model(disease: str) -> dict:
    if disease not in DISEASES:
        return {}
    
    with _lock:
        if disease in _cache:
            return _cache[disease]
            
        model_base_dir = Path(os.environ.get("MODEL_BASE_DIR", "/opt/medimind/models"))
        model_dir = model_base_dir / disease
        
        bundle = {
            "model": _load_joblib(model_dir / "model.joblib"),
            "scaler": _load_joblib(model_dir / "scaler.joblib"),
            "feature_columns": _load_json(model_dir / "feature_columns.json", []),
        }
        
        if os.environ.get("ENABLE_SHAP_EXPLANATIONS", "True").lower() == "true":
            bundle["shap_explainer"] = _load_joblib(model_dir / "shap_explainer.joblib")
        else:
            bundle["shap_explainer"] = None
            
        _cache[disease] = bundle
        return bundle
