from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cnn.config import CNNModelConfig, get_cnn_model_configs
from cnn.fallback import DatasetCentroidFallback
from cnn.preprocessing import ImageValidationError, OODImageError, preprocess_image, validate_chest_xray


class CNNModelUnavailable(RuntimeError):
    """Raised when a requested CNN model has not been loaded."""


class CNNModelRegistry:
    def __init__(self, configs: dict[str, CNNModelConfig] | None = None):
        self.configs = configs or get_cnn_model_configs()
        self.models: dict[str, Any] = {}
        self.backends: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        self.warnings: dict[str, str] = {}

    def load_all(self) -> None:
        for model_id, config in self.configs.items():
            self.load_model(model_id, config)

    def load_model(self, model_id: str, config: CNNModelConfig) -> None:
        if model_id in self.models:
            return
        if not config.model_path.exists():
            self._load_dataset_fallback(model_id, config, f"Model artifact not found at {config.model_path}")
            return
        try:
            try:
                from tensorflow import keras as tf_keras

                model = tf_keras.models.load_model(str(config.model_path), compile=False)
            except ImportError:
                import keras

                model = keras.models.load_model(str(config.model_path), compile=False)
            self.models[model_id] = model
            self.backends[model_id] = "keras_cnn"
            self.errors.pop(model_id, None)
            self.warnings.pop(model_id, None)
        except Exception as exc:
            self._load_dataset_fallback(model_id, config, f"Model load failed: {exc}")

    def _load_dataset_fallback(self, model_id: str, config: CNNModelConfig, warning: str) -> None:
        try:
            self.models[model_id] = DatasetCentroidFallback.from_dataset(config)
            self.backends[model_id] = "dataset_knn_fallback"
            self.warnings[model_id] = (
                f"{warning}. Using labeled xray dataset nearest-neighbor fallback until the trained CNN artifact is restored."
            )
            self.errors.pop(model_id, None)
        except Exception as exc:
            self.errors[model_id] = f"{warning}. Fallback unavailable: {exc}"

    def list_models(self) -> list[dict]:
        return [self._model_status(model_id, config) for model_id, config in self.configs.items()]

    def status(self) -> dict:
        return {
            "loaded_count": len(self.models),
            "configured_count": len(self.configs),
            "models": self.list_models(),
        }

    def predict(
        self,
        model_id: str,
        image_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict:
        config = self.configs.get(model_id)
        if not config:
            raise KeyError(f"Unknown CNN model: {model_id}")
        if model_id not in self.models:
            if model_id not in self.errors:
                self.load_model(model_id, config)
            raise CNNModelUnavailable(self.errors.get(model_id, "Model is not loaded."))

        import logging
        logger = logging.getLogger(__name__)

        # Out-of-distribution check: reject non-chest-X-ray images
        try:
            ood_result = validate_chest_xray(image_bytes, filename=filename, content_type=content_type)
            if not ood_result.get("ood_skipped", False) and ood_result.get("rejected", False):
                raise OODImageError(
                    "Please upload a chest X-ray image."
                )
        except OODImageError:
            raise

        processed = preprocess_image(image_bytes, config, filename=filename, content_type=content_type)
        raw_prediction = self.models[model_id].predict(processed.batch, verbose=0)
        probabilities = self._probabilities(config, raw_prediction)
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted_class])
        
        # Determine predicted index for binary model
        if predicted_class == config.labels.get(1):
            predicted_index = 1
        else:
            predicted_index = 0

        logger.info(
            f"Prediction Log -> File: {filename} | "
            f"Raw output: {raw_prediction} | "
            f"Probabilities: {probabilities} | "
            f"Predicted index: {predicted_index} | "
            f"Predicted class: {predicted_class} | "
            f"Confidence: {confidence}"
        )

        return {
            "model_id": model_id,
            "model_name": config.display_name,
            "model_version": "pneumonia_xray_v1",
            "disease": config.disease,
            "modality": config.modality,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "confidence_percentage": round(confidence * 100, 2),
            "confidence_interpretation": self._confidence_interpretation(confidence),
            "probabilities": {label: round(float(value), 6) for label, value in probabilities.items()},
            "diagnosis": self._diagnosis_text(config, predicted_class, confidence),
            "assessment_summary": self._assessment_summary(predicted_class, confidence),
            "clinical_recommendations": self._clinical_recommendations(predicted_class),
            "formatted_report": self._formatted_report(predicted_class, confidence, probabilities),
            "metadata": {
                **processed.metadata,
                "inference_backend": self.backends.get(model_id, "unknown"),
                "warning": self.warnings.get(model_id),
                "model_name": config.display_name,
                "model_version": "pneumonia_xray_v1",
                "analysis_time": datetime.now(timezone.utc).isoformat(),
                "processing_status": "completed",
                "model_path": str(config.model_path),
                "training_source": str(config.training_source),
                "dataset_dirs": [str(path) for path in config.dataset_dirs],
                "threshold": config.threshold,
                "labels": config.labels,
            },
        }

    def _probabilities(self, config: CNNModelConfig, raw_prediction: Any) -> dict[str, float]:
        values = np.asarray(raw_prediction, dtype=np.float32).reshape(-1)
        if values.size == 1:
            positive_probability = float(np.clip(values[0], 0.0, 1.0))
            return {
                config.labels[0]: 1.0 - positive_probability,
                config.labels[1]: positive_probability,
            }

        values = values.astype(np.float64)
        if np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-3):
            shifted = values - np.max(values)
            exp_values = np.exp(shifted)
            values = exp_values / exp_values.sum()
        return {
            config.labels.get(index, f"class_{index}"): float(probability)
            for index, probability in enumerate(values)
        }

    def _diagnosis_text(self, config: CNNModelConfig, predicted_class: str, confidence: float) -> str:
        percentage = round(confidence * 100, 1)
        if predicted_class == "PNEUMONIA":
            return (
                f"The chest X-ray model detected imaging patterns consistent with pneumonia "
                f"with {percentage}% confidence. This is decision support only and should be "
                "reviewed by a qualified clinician."
            )
        return (
            f"The chest X-ray model classified this image as normal with {percentage}% confidence. "
            "This does not replace clinical review, especially if symptoms are present."
        )

    def _confidence_interpretation(self, confidence: float) -> dict:
        percentage = confidence * 100
        if percentage >= 95:
            label = "Very High Confidence"
        elif percentage >= 85:
            label = "High Confidence"
        elif percentage >= 70:
            label = "Moderate Confidence"
        else:
            label = "Low Confidence"
        return {
            "label": label,
            "percentage": round(percentage, 1),
            "description": "Confidence reflects the model probability for the primary finding, not diagnostic certainty.",
        }

    def _assessment_summary(self, predicted_class: str, confidence: float) -> str:
        interpretation = self._confidence_interpretation(confidence)["label"].lower()
        if predicted_class == "PNEUMONIA":
            return (
                "The AI model detected imaging features that are consistent with pneumonia. "
                f"This prediction has {interpretation} and should be interpreted as clinical "
                "decision support rather than a definitive diagnosis."
            )
        return (
            "The AI model did not detect imaging features strongly suggestive of pneumonia. "
            f"This prediction has {interpretation}; continue clinical evaluation if symptoms persist."
        )

    def _clinical_recommendations(self, predicted_class: str) -> list[str]:
        if predicted_class == "PNEUMONIA":
            return [
                "Consult a qualified physician or radiology professional.",
                "Correlate the AI finding with symptoms, physical examination, and medical history.",
                "Additional imaging or laboratory testing may be recommended by the care team.",
            ]
        return [
            "No significant pneumonia pattern was detected by the AI model.",
            "Continue clinical evaluation if fever, cough, chest pain, or breathing difficulty persists.",
            "Seek professional medical advice if symptoms worsen or new symptoms appear.",
        ]

    def _formatted_report(self, predicted_class: str, confidence: float, probabilities: dict[str, float]) -> str:
        percentage = confidence * 100
        
        if predicted_class == "PNEUMONIA":
            interpretation = (
                "The uploaded chest X Ray shows radiographic patterns consistent with pneumonia. "
                "This AI generated prediction should assist clinical assessment and is not a substitute "
                "for evaluation by a qualified healthcare professional."
            )
        else:
            interpretation = (
                "The uploaded chest X Ray does not show radiographic patterns strongly consistent with pneumonia. "
                "This AI generated prediction should assist clinical assessment and is not a substitute "
                "for evaluation by a qualified healthcare professional."
            )
        
        alt_probs = "\n".join([f"{label.capitalize()}: {prob * 100:.1f}%" for label, prob in probabilities.items()])
        
        return (
            f"Prediction: {predicted_class.capitalize()}\n"
            f"Confidence: {percentage:.1f}%\n\n"
            f"Interpretation:\n{interpretation}\n\n"
            f"Alternative Probabilities:\n{alt_probs}"
        )

    def _model_status(self, model_id: str, config: CNNModelConfig) -> dict:
        data = asdict(config)
        data["model_path"] = str(config.model_path)
        data["dataset_dirs"] = [str(path) for path in config.dataset_dirs]
        data["training_source"] = str(config.training_source)
        data["loaded"] = model_id in self.models
        data["available"] = config.model_path.exists()
        data["backend"] = self.backends.get(model_id)
        data["fallback_active"] = bool(self.backends.get(model_id, "").endswith("_fallback"))
        data["warning"] = self.warnings.get(model_id)
        data["error"] = self.errors.get(model_id)
        data["dataset_summary"] = self._dataset_summary(config.dataset_dirs)
        return data

    def _dataset_summary(self, dataset_dirs: tuple[Path, ...]) -> dict:
        summary: dict[str, dict[str, int]] = {}
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                continue
            split_summary: dict[str, int] = {}
            for split in ("train", "val", "test"):
                split_dir = dataset_dir / split
                if not split_dir.exists():
                    continue
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        key = f"{split}/{class_dir.name}"
                        split_summary[key] = len([path for path in class_dir.iterdir() if path.is_file()])
            summary[str(dataset_dir)] = split_summary
        return summary
