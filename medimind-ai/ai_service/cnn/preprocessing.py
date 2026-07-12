from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from cnn.config import CNNModelConfig


MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not suitable for CNN inference."""


@dataclass(frozen=True)
class PreprocessedImage:
    batch: np.ndarray
    metadata: dict


def validate_image_upload(image_bytes: bytes, filename: str | None, content_type: str | None) -> None:
    if not image_bytes:
        raise ImageValidationError("Uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image is larger than the 20 MB upload limit.")

    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type and normalized_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError("Only JPEG, PNG, and WEBP images are supported.")

    if filename:
        lower_name = filename.lower()
        if not any(lower_name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            raise ImageValidationError("Image file extension must be jpg, jpeg, png, or webp.")


def preprocess_image(
    image_bytes: bytes,
    config: CNNModelConfig,
    filename: str | None = None,
    content_type: str | None = None,
) -> PreprocessedImage:
    validate_image_upload(image_bytes, filename, content_type)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            original = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }
            converted = image.convert(config.color_mode)
            resized = converted.resize(config.input_size)
            array = np.asarray(resized, dtype=np.float32)
    except UnidentifiedImageError as exc:
        raise ImageValidationError("Uploaded file could not be read as an image.") from exc

    if config.normalization == "rescale_1_255":
        array = array / 255.0
    else:
        raise ImageValidationError(f"Unsupported normalization pipeline: {config.normalization}")

    batch = np.expand_dims(array, axis=0)
    metadata = {
        "filename": filename,
        "content_type": content_type,
        "original_image": original,
        "input_shape": list(batch.shape),
        "preprocessing": {
            "resize": list(config.input_size),
            "color_mode": config.color_mode,
            "normalization": config.normalization,
        },
    }
    return PreprocessedImage(batch=batch, metadata=metadata)

