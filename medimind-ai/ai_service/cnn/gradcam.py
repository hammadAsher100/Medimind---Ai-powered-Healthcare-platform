"""Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN interpretability.

Generates heatmap overlays that highlight the regions of a chest X-ray that
most influenced the model's prediction.  Designed for the VGG16-based
pneumonia classifier used in MediMind's chest X-ray pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ── Jet colormap lookup table (fallback when matplotlib is unavailable) ─────
_JET_LUT = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 1.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ],
    dtype=np.float32,
)

_LAYER_NAME = "block5_conv3"


# ── Public interface ──────────────────────────────────────────────────────────


def generate_gradcam(
    model: Any,
    input_batch: np.ndarray,
    predicted_class_idx: int,
    original_image_bytes: bytes,
) -> dict[str, Any] | None:
    """Generate a Grad-CAM heatmap for a chest X-ray prediction.

    Args:
        model: Loaded Keras CNN model (must have ``block5_conv3`` or another
            Conv2D layer whose activations can be probed).
        input_batch: Preprocessed input with shape ``(1, H, W, 3)``,
            normalised according to the model's config.
        predicted_class_idx: ``0`` = NORMAL, ``1`` = PNEUMONIA.
        original_image_bytes: Raw bytes of the user-uploaded image so the
            heatmap can be composited at the original resolution.

    Returns:
        A dict with the following keys, or ``None`` when Grad-CAM cannot be
        computed (fallback model, no conv layers, TF unavailable, etc.):

        - ``heatmap_base64`` — standalone jet-colormap heatmap as a data URI.
        - ``overlay_base64``  — heatmap blended onto the original X-ray.
        - ``explanation``     — clinical interpretability text.
        - ``layer_used``      — name of the targeted conv layer.
        - ``status``          — ``"generated"`` on success.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras as tf_keras
    except ImportError:
        logger.warning("TensorFlow not available; skipping Grad-CAM.")
        return None

    try:
        conv_layer = _find_conv_layer(model)
        if conv_layer is None:
            logger.warning("No suitable Conv2D layer found for Grad-CAM.")
            return None

        # Build a multi-output model: [conv activations, final prediction]
        grad_model = _build_grad_model(model, conv_layer, tf_keras)
        if grad_model is None:
            return None

        # Convert the batch to a TF tensor so GradientTape can track it
        input_tensor = tf.constant(input_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            # Binary sigmoid output: shape (1, 1).  Compute class score.
            if predicted_class_idx == 1:
                class_score = predictions[0, 0]
            else:
                class_score = 1.0 - predictions[0, 0]

        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            logger.warning("Grad-CAM: gradient is None — skipping.")
            return None

        # Global-average-pool the gradients over spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (512,)

        # Weight each feature map by its pooled gradient
        conv_outputs_np = conv_outputs[0]  # (14, 14, 512)
        heatmap = tf.reduce_sum(
            conv_outputs_np * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1
        )
        heatmap = tf.maximum(heatmap, 0.0)  # ReLU: keep only positive influence

        # Normalise to [0, 1]
        heatmap_max = tf.reduce_max(heatmap)
        heatmap_np = (
            (heatmap / heatmap_max).numpy()
            if heatmap_max > 1e-8
            else np.zeros((14, 14), dtype=np.float32)
        )

        # ── Load original image for overlay ──────────────────────────────
        with PILImage.open(io.BytesIO(original_image_bytes)) as orig:
            orig_rgb = orig.convert("RGB")
            orig_size = orig_rgb.size  # (width, height)

        # Resize heatmap to original image dimensions
        heatmap_img = PILImage.fromarray(
            (heatmap_np * 255).astype(np.uint8), mode="L"
        ).resize(orig_size, PILImage.BILINEAR)
        heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0

        # Apply jet colormap
        heatmap_colored = _apply_jet_colormap(heatmap_resized)  # (H, W, 3)

        # Composite overlay: heatmap semi-transparent over original
        orig_array = np.asarray(orig_rgb, dtype=np.float32)
        alpha = 0.4
        overlay_array = (orig_array * (1.0 - alpha) + heatmap_colored * alpha).astype(
            np.uint8
        )
        overlay_img = PILImage.fromarray(overlay_array, mode="RGB")

        # Standalone heatmap (resized to original size)
        heatmap_only_img = PILImage.fromarray(heatmap_colored, mode="RGB")

        # ── Encode both as base64 data URIs ──────────────────────────────
        def _encode_pil(img: PILImage.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('ascii')}"
            )

        return {
            "heatmap_base64": _encode_pil(heatmap_only_img),
            "overlay_base64": _encode_pil(overlay_img),
            "explanation": (
                "The highlighted regions (yellow/red) indicate the areas of the "
                "chest X-ray that most influenced the model's prediction.  Red "
                "signifies stronger influence on the decision.  This heatmap is "
                "an interpretability aid and does not represent a clinical diagnosis."
            ),
            "layer_used": _LAYER_NAME,
            "status": "generated",
        }

    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc)
        return None


# ── Internal helpers ─────────────────────────────────────────────────────────


def _find_conv_layer(model: Any) -> Any | None:
    """Locate the target convolutional layer, searching nested models.

    Handles both flat Sequential models and models where the feature
    extractor is a nested Functional model (e.g. VGG16 inside a Sequential).
    """
    # Direct lookup by name (flat model)
    try:
        return model.get_layer(_LAYER_NAME)
    except (ValueError, AttributeError):
        pass

    # Recurse into nested containers (Sequential → Functional VGG16)
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                return layer.get_layer(_LAYER_NAME)
            except (ValueError, AttributeError):
                continue
        if hasattr(layer, "name") and layer.name == _LAYER_NAME:
            return layer

    # Fallback: last Conv2D layer anywhere in the model tree
    return _last_conv2d(model)


def _last_conv2d(container: Any) -> Any | None:
    """Walk the layer tree and return the last Conv2D found."""
    import tensorflow.keras.layers as tf_layers

    last: Any | None = None

    def _walk(parent: Any) -> None:
        nonlocal last
        for layer in getattr(parent, "layers", []):
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                _walk(layer)  # descendant model
            elif isinstance(layer, tf_layers.Conv2D):
                last = layer

    try:
        _walk(container)
    except Exception:
        pass
    return last


def _build_grad_model(
    model: Any, conv_layer: Any, tf_keras: Any
) -> Any | None:
    """Create a temporary model that outputs conv activations + predictions.

    Keras 3.x Sequential models don't expose ``model.output`` as a graph
    tensor, so we manually trace through every layer (including nested
    Functional sub-models like VGG16) to build a new functional graph.
    """
    import tensorflow as tf  # noqa: F811

    try:
        # Infer input shape from the model (e.g. (224, 224, 3))
        input_shape = model.input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if input_shape is None:
            logger.warning("Grad-CAM: model has no input_shape.")
            return None
        # input_shape is (None, 224, 224, 3) — drop the batch dim
        if len(input_shape) == 4:
            input_shape = input_shape[1:]
        inputs = tf_keras.Input(shape=tuple(input_shape))

    except Exception as exc:
        logger.warning("Grad-CAM: could not create Input tensor — %s", exc)
        return None

    try:
        x = inputs
        conv_output: Any | None = None

        for layer in model.layers:
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                # Nested Functional model (e.g. VGG16 inside Sequential)
                for inner in layer.layers:
                    # InputLayer is not callable — it's a tensor spec that
                    # gets resolved when the parent model is built.
                    if isinstance(inner, tf_keras.layers.InputLayer):
                        continue
                    x = inner(x)
                    if inner.name == conv_layer.name:
                        conv_output = x
            else:
                x = layer(x)
                if layer.name == conv_layer.name:
                    conv_output = x

        if conv_output is None:
            logger.warning(
                "Grad-CAM: conv layer %s not found in trace.", conv_layer.name
            )
            return None

        return tf_keras.Model(inputs=inputs, outputs=[conv_output, x])

    except Exception as exc:
        logger.warning("Failed to build Grad-CAM gradient model: %s", exc)
        return None


def _apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Apply the *jet* colormap to a single-channel ``[0, 1]`` heatmap.

    Returns an ``(H, W, 3)`` uint8 array.
    """
    try:
        import matplotlib as mpl

        rgba = mpl.colormaps["jet"](heatmap)  # (H, W, 4)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        # LUT fallback: nearest-neighbor interpolation in jet palette
        idx = (heatmap * (_JET_LUT.shape[0] - 1)).astype(np.int32)
        idx = np.clip(idx, 0, _JET_LUT.shape[0] - 1)
        return (_JET_LUT[idx] * 255).astype(np.uint8)
