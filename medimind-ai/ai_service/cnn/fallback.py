from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cnn.config import CNNModelConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class DatasetCentroidFallback:
    """Dataset-calibrated local baseline used only when the trained CNN artifact is absent."""

    def __init__(
        self,
        config: CNNModelConfig,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        k_neighbors: int = 9,
    ):
        self.config = config
        self.feature_matrix = feature_matrix
        self.labels = labels
        self.feature_mean = feature_mean
        self.feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
        self.k_neighbors = max(1, min(k_neighbors, len(labels)))

    @classmethod
    def from_dataset(cls, config: CNNModelConfig) -> "DatasetCentroidFallback":
        split_dirs = []
        for dataset_dir in config.dataset_dirs:
            for split in ("train", "val"):
                split_dir = dataset_dir / split
                if split_dir.exists():
                    split_dirs.append(split_dir)
        if not split_dirs:
            raise FileNotFoundError("No xray training split was found for fallback calibration.")

        features: list[np.ndarray] = []
        labels: list[int] = []
        for split_dir in split_dirs:
            for index, label in config.labels.items():
                class_dir = split_dir / label
                if not class_dir.exists():
                    continue
                for image_path in sorted(class_dir.iterdir()):
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                        features.append(cls._features_from_path(image_path, config))
                        labels.append(index)

        if not features or set(labels) != set(config.labels):
            raise ValueError("Fallback calibration requires at least one image for every class.")

        raw_matrix = np.vstack(features).astype(np.float32)
        feature_mean = raw_matrix.mean(axis=0)
        feature_std = raw_matrix.std(axis=0)
        normalized_matrix = (raw_matrix - feature_mean) / np.where(feature_std < 1e-6, 1.0, feature_std)
        return cls(
            config=config,
            feature_matrix=normalized_matrix.astype(np.float32),
            labels=np.asarray(labels, dtype=np.int64),
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
        )

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        predictions = []
        for image_array in batch:
            features = self._features_from_array(image_array)
            normalized = (features - self.feature_mean) / self.feature_std
            distances = np.linalg.norm(self.feature_matrix - normalized, axis=1)
            neighbor_indices = np.argsort(distances)[: self.k_neighbors]
            neighbor_distances = distances[neighbor_indices]
            neighbor_labels = self.labels[neighbor_indices]
            temperature = max(float(np.median(neighbor_distances)), 1e-6)
            weights = np.exp(-neighbor_distances / temperature)
            normal_weight = float(weights[neighbor_labels == 0].sum())
            pneumonia_weight = float(weights[neighbor_labels == 1].sum())
            total = normal_weight + pneumonia_weight
            pneumonia_probability = pneumonia_weight / total if total else 0.5
            pneumonia_probability = float(np.clip(0.02 + (0.96 * pneumonia_probability), 0.02, 0.98))
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
        normalized_gray = (grayscale - grayscale.mean()) / (grayscale.std() + 1e-6)
        small = np.asarray(
            Image.fromarray(np.uint8(np.clip(grayscale, 0, 1) * 255)).resize((40, 40)),
            dtype=np.float32,
        ) / 255.0
        contrast_small = np.asarray(
            Image.fromarray(np.uint8(np.clip((normalized_gray + 3.0) / 6.0, 0, 1) * 255)).resize((24, 24)),
            dtype=np.float32,
        ) / 255.0

        grad_y, grad_x = np.gradient(grayscale)
        gradient_magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
        gradient_small = np.asarray(
            Image.fromarray(np.uint8(np.clip(gradient_magnitude / (gradient_magnitude.max() + 1e-6), 0, 1) * 255)).resize((16, 16)),
            dtype=np.float32,
        ) / 255.0

        left_lung = grayscale[44:180, 22:102]
        right_lung = grayscale[44:180, 122:202]
        center = grayscale[30:194, 96:128]
        upper_lungs = grayscale[32:96, 22:202]
        lower_lungs = grayscale[128:192, 22:202]
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
                upper_lungs.mean(),
                lower_lungs.mean(),
                upper_lungs.std(),
                lower_lungs.std(),
                gradient_magnitude.mean(),
                gradient_magnitude.std(),
                float((left_lung > threshold).mean()),
                float((right_lung > threshold).mean()),
                float((upper_lungs > threshold).mean()),
                float((lower_lungs > threshold).mean()),
            ],
            dtype=np.float32,
        )
        row_profile = grayscale.mean(axis=1)[::4].astype(np.float32)
        column_profile = grayscale.mean(axis=0)[::4].astype(np.float32)
        return np.concatenate([
            small.reshape(-1),
            contrast_small.reshape(-1),
            gradient_small.reshape(-1),
            row_profile,
            column_profile,
            stats,
        ]).astype(np.float32)
