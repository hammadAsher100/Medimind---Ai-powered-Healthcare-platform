"""
Preprocessing pipeline for the Chest X-Ray Pneumonia dataset.

Uses Keras ImageDataGenerator with augmentation for training and
simple rescaling for validation/test.  Computes class weights for
the ~3:1 pneumonia:normal imbalance.

Expected directory layout (from Kaggle download):
  data/raw/xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
RAW_DATA_DIR = os.getenv("CNN_RAW_DATA", "data/raw/xray")


def compute_class_weights(train_dir: str) -> dict[int, float]:
    """Compute class weights from the training directory structure.

    Returns {0: weight_normal, 1: weight_pneumonia} to counteract
    the ~3:1 pneumonia:normal imbalance.
    """
    normal_dir = Path(train_dir) / "NORMAL"
    pneumonia_dir = Path(train_dir) / "PNEUMONIA"

    n_normal = len(list(normal_dir.glob("*"))) if normal_dir.exists() else 1
    n_pneumonia = len(list(pneumonia_dir.glob("*"))) if pneumonia_dir.exists() else 1
    total = n_normal + n_pneumonia

    return {
        0: total / (2 * n_normal),
        1: total / (2 * n_pneumonia),
    }


def get_train_generator(data_dir: str = RAW_DATA_DIR) -> "DirectoryIterator":
    """Training data generator with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=(0.9, 1.1),
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode="nearest",
    )

    train_dir = str(Path(data_dir) / "train")
    return train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
    )


def get_val_generator(data_dir: str = RAW_DATA_DIR) -> "DirectoryIterator":
    """Validation data generator (no augmentation)."""
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_dir = str(Path(data_dir) / "val")
    return val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )


def get_test_generator(data_dir: str = RAW_DATA_DIR) -> "DirectoryIterator":
    """Test data generator (no augmentation)."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_dir = str(Path(data_dir) / "test")
    return test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )


def preprocess_single_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess a single uploaded image for inference.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes (jpg/png).

    Returns
    -------
    np.ndarray of shape (1, 224, 224, 3) normalised to [0, 1].
    """
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
