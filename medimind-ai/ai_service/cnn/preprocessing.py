from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from cnn.config import CNNModelConfig


MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── OOD detection constants ──────────────────────────────────────────────
# These thresholds were derived empirically from chest X-ray datasets.
# They are deliberately conservative to avoid false-positive rejection
# of genuine X-rays while catching obviously wrong uploads.
MIN_GRAYSCALE_RATIO = 0.90       # ≤ 5% of pixels should have strong color
MAX_ASPECT_RATIO = 2.0            # 4:3 = 1.33, 16:9 = 1.78 — X-rays are rarely wider
MIN_ASPECT_RATIO = 0.5            # portrait X-rays are ~0.75–1.0
MAX_EDGE_DENSITY = 0.35           # X-rays have soft gradients; text / photos are sharper


class ImageValidationError(ValueError):
    """Raised when an uploaded image is not suitable for CNN inference."""


class OODImageError(ImageValidationError):
    """Raised when an image does not appear to be a chest X-ray."""


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


def check_grayscale_ratio(image: Image.Image) -> tuple[float, bool]:
    """Check if the image is predominantly grayscale.

    Computes the per-pixel channel standard deviation across R,G,B.
    True X-rays have low channel variance (near-monochrome).
    Color photos have high channel variance.

    Returns (grayscale_ratio, passes).
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return 1.0, True

    channel_1, channel_2 = arr[..., 0:1], arr[..., 1:2]
    # For each pixel, the max deviation from the mean of the 3 channels
    mean_across_channels = arr.mean(axis=2, keepdims=True)
    max_deviation = np.abs(arr - mean_across_channels).max(axis=2)
    channel_std = max_deviation.mean()

    # Heuristic: channel-std < 15 on [0,255] scale suggests nearly gray
    grayscale_ratio = float(np.clip(1.0 - (channel_std / 64.0), 0.0, 1.0))
    return grayscale_ratio, grayscale_ratio >= MIN_GRAYSCALE_RATIO


def check_aspect_ratio(image: Image.Image) -> tuple[float, bool]:
    """Check that the image aspect ratio is plausible for a chest X-ray.

    Chest X-rays are typically portrait orientation with ratio between
    0.75 (3:4) and 1.33 (4:3). Panoramic / squashed images are likely
    not X-rays.

    Returns (aspect_ratio, passes).
    """
    w, h = image.size
    ratio = w / max(h, 1)
    passes = MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO
    return float(ratio), passes


def check_edge_density(image: Image.Image) -> tuple[float, bool]:
    """Estimate edge density via simple gradient magnitude.

    Chest X-rays have smooth transitions. Photos of everyday objects,
    text documents, and medical illustrations have sharper edges.

    Returns (edge_density, passes).
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_x = np.abs(np.diff(gray, axis=1))
    # Pad back to original size
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode="edge")
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode="edge")
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_density = float((magnitude > 30.0).mean())
    return edge_density, edge_density <= MAX_EDGE_DENSITY


def check_brightness_distribution(image: Image.Image) -> tuple[float, bool]:
    """Check that the histogram shape is plausible for an X-ray.

    True X-rays have a broad, often bimodal histogram (dark background +
    brighter anatomical structures). Solid-color or narrow-histogram
    images are suspect.

    Returns (histogram_spread, passes) where spread is the IQR / 255.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    p25, p75 = float(np.percentile(gray, 25)), float(np.percentile(gray, 75))
    spread = (p75 - p25) / 255.0
    # A very narrow histogram (< 10% of dynamic range) is suspicious
    passes = spread > 0.04
    return float(spread), passes


def validate_chest_xray(
    image_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Reject images that are clearly not chest X-rays.

    Uses lightweight heuristics — no model needed:
      1. Grayscale ratio: X-rays are near-monochrome.
      2. Aspect ratio: X-rays are not extreme panoramas.
      3. Edge density: X-rays have soft gradients, not sharp edges.
      4. Brightness spread: X-rays use most of the dynamic range.

    These checks are deliberately conservative (favor false-accept over
    false-reject). A dedicated X-ray vs. non-X-ray classifier could be
    added later as a more precise alternative.

    Returns the OOD diagnostics dict. Raises OODImageError if rejected.
    """
    if os.environ.get("DISABLE_OOD", "False").lower() == "true":
        return {"ood_skipped": True}

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            checks: dict[str, Any] = {"filename": filename, "content_type": content_type}

            gr_ratio, gr_pass = check_grayscale_ratio(image)
            checks["grayscale_ratio"] = round(gr_ratio, 4)
            checks["grayscale_pass"] = gr_pass

            ar_ratio, ar_pass = check_aspect_ratio(image)
            checks["aspect_ratio"] = round(ar_ratio, 4)
            checks["aspect_pass"] = ar_pass

            ed_density, ed_pass = check_edge_density(image)
            checks["edge_density"] = round(ed_density, 4)
            checks["edge_pass"] = ed_pass

            h_spread, h_pass = check_brightness_distribution(image)
            checks["histogram_spread"] = round(h_spread, 4)
            checks["histogram_pass"] = h_pass

            checks["all_passed"] = gr_pass and ar_pass and ed_pass and h_pass
            checks["rejected"] = not checks["all_passed"]

            if not checks["all_passed"]:
                failed = []
                if not gr_pass:
                    failed.append("grayscale ratio")
                if not ar_pass:
                    failed.append(f"aspect ratio ({ar_ratio:.2f})")
                if not ed_pass:
                    failed.append("edge density")
                if not h_pass:
                    failed.append("brightness spread")
                raise OODImageError(
                    "Please upload a chest X-ray image. "
                    f"The uploaded image does not appear to be an X-ray ({', '.join(failed)})."
                )

            return checks
    except OODImageError:
        raise
    except Exception as exc:
        # If we can't even read the image for OOD checks, let the normal
        # validation pipeline handle it (it may be a corrupted file).
        return {"ood_skipped": True, "reason": str(exc)}


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

