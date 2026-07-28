from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CNNModelConfig:
    model_id: str
    display_name: str
    disease: str
    modality: str
    model_path: Path
    labels: dict[int, str]
    input_size: tuple[int, int]
    color_mode: str
    normalization: str
    threshold: float
    dataset_dirs: tuple[Path, ...]
    training_source: Path
    description: str


def _first_existing_model(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _pneumonia_model_path() -> Path:
    env_path = os.environ.get("CNN_PNEUMONIA_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        PROJECT_ROOT / "ml" / "registry" / "cnn_pneumonia_v4.h5",  # Try v4 first
        PROJECT_ROOT / "ml" / "registry" / "cnn_pneumonia_v3.h5",
        PROJECT_ROOT / "ml" / "registry" / "cnn_pneumonia_v2.h5",
        PROJECT_ROOT / "ml" / "registry" / "cnn_pneumonia.h5",
        PROJECT_ROOT / "ml" / "cnn" / "cnn_pneumonia.h5",
        PROJECT_ROOT / "ai_service" / "ml_models" / "pneumonia" / "cnn_pneumonia.h5",
    ]

    discovered = sorted((PROJECT_ROOT / "ml").glob("**/*pneumonia*.h5"))
    candidates.extend(discovered)
    return _first_existing_model(candidates)


def get_cnn_model_configs() -> dict[str, CNNModelConfig]:
    pneumonia = CNNModelConfig(
        model_id="pneumonia_xray",
        display_name="Chest X-ray Pneumonia Detection",
        disease="Pneumonia",
        modality="chest_xray",
        model_path=_pneumonia_model_path(),
        labels={0: "NORMAL", 1: "PNEUMONIA"},
        input_size=(224, 224),
        color_mode="RGB",
        normalization="rescale_1_255",
        threshold=0.5,
        dataset_dirs=(
            PROJECT_ROOT / "data" / "raw" / "xray" / "chest_xray",  # Primary: nested dataset
            PROJECT_ROOT / "data" / "raw" / "xray",                 # Fallback: flat structure
            PROJECT_ROOT / "data" / "processed" / "xray",           # Fallback: processed
        ),
        training_source=PROJECT_ROOT / "ml" / "cnn" / "train.py",
        description=(
            "Binary chest X-ray classifier trained with MobileNetV2 to distinguish "
            "normal scans from pneumonia findings."
        ),
    )
    return {pneumonia.model_id: pneumonia}

