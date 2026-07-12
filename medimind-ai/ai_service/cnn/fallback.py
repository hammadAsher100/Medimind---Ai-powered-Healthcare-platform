from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cnn.config import CNNModelConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetCentroidFallback:
    """Small local image baseline used only when the trained CNN artifact is absent."""

    def __init__(self, config: CNNModelConfig, centroids: dict[int, np.ndarray], temperature: float):
        self.config = config
        self.centroids = centroids
        self.temperature = max(float(temperature), 1e-6)

    @classmethod
    def from_dataset(cls, config: CNNModelConfig) -> "DatasetCentroidFallback":
        train_dir = next(
            (dataset_dir / "train" for dataset_dir in config.dataset_dirs if (dataset_dir / "train").exists()),
            None,
        )
        if train_dir is None:
            raise FileNotFoundError("No xray training split was found for fallback calibration.")

        feature_groups: dict[int, list[np.ndarray]] = {index: [] for index in config.labels}
        for index, label in config.labels.items():
            class_dir = train_dir / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing fallback class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    feature_groups[index].append(cls._features_from_path(image_path, config))

        if any(not features for features in feature_groups.values()):
            raise ValueError("Fallback calibration requires at least one image for every class.")

        centroids = {
            index: np.mean(np.vstack(features), axis=0)
            for index, features in feature_groups.items()
        }
        centroid_values = list(centroids.values())
        if len(centroid_values) > 1:
            distances = [
                np.linalg.norm(centroid_values[left] - centroid_values[right])
                for left in range(len(centroid_values))
                for right in range(left + 1, len(centroid_values))
            ]
            temperature = float(np.median(distances) / 2.0)
        else:
            temperature = 1.0
        return cls(config=config, centroids=centroids, temperature=temperature)

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        predictions = []
        for image_array in batch:
            features = self._features_from_array(image_array)
            distances = {
                index: float(np.linalg.norm(features - centroid))
                for index, centroid in self.centroids.items()
            }
            weights = {
                index: np.exp(-distance / self.temperature)
                for index, distance in distances.items()
            }
            total = sum(weights.values()) or 1.0
            pneumonia_probability = float(weights.get(1, 0.0) / total)
            predictions.append([pneumonia_probability])
        return np.asarray(predictions, dtype=np.float32)

    @classmethod
    def _features_from_path(cls, image_path: Path, config: CNNModelConfig) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert(config.color_mode).resize(config.input_size)
            array = np.asarray(image, dtype=np.float32) / 255.0
        return cls._features_from_array(array)

    @staticmethod
    def _features_from_array(image_array: np.ndarray) -> np.ndarray:
        grayscale = image_array.mean(axis=2)
        small = np.asarray(
            Image.fromarray(np.uint8(np.clip(grayscale, 0, 1) * 255)).resize((32, 32)),
            dtype=np.float32,
        ) / 255.0

        left_lung = grayscale[44:180, 22:102]
        right_lung = grayscale[44:180, 122:202]
        center = grayscale[30:194, 96:128]
        threshold = float(grayscale.mean() + grayscale.std())
        stats = np.asarray(
            [
                grayscale.mean(),
                grayscale.std(),
                left_lung.mean(),
                right_lung.mean(),
                abs(left_lung.mean() - right_lung.mean()),
                left_lung.std(),
                right_lung.std(),
                center.mean(),
                float((left_lung > threshold).mean()),
                float((right_lung > threshold).mean()),
            ],
            dtype=np.float32,
        )
        return np.concatenate([small.reshape(-1), stats])

