"""Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN interpretability.

Generates heatmap overlays that highlight the regions of a chest X-ray that
most influenced the model's prediction.  Supports MobileNetV2-based models
(and VGG16 / ResNet50 as fallbacks) saved as .h5 / SavedModel.
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

# Conv layer name candidates tried in order across architectures:
# MobileNetV2: "Conv_1"  |  VGG16: "block5_conv3"  |  ResNet50: "conv5_block3_out"
_LAYER_CANDIDATES = ["Conv_1", "out_relu", "block5_conv3", "conv5_block3_out"]


# ── Public interface ──────────────────────────────────────────────────────────


def generate_gradcam(
    model: Any,
    input_batch: np.ndarray,
    predicted_class_idx: int,
    original_image_bytes: bytes,
) -> dict[str, Any] | None:
    """Generate a Grad-CAM heatmap for a chest X-ray prediction.

    Args:
        model: Loaded Keras CNN model.
        input_batch: Preprocessed input with shape ``(1, H, W, 3)``.
        predicted_class_idx: ``0`` = NORMAL, ``1`` = PNEUMONIA.
        original_image_bytes: Raw bytes of the user-uploaded image.

    Returns:
        Dict with ``heatmap_base64``, ``overlay_base64``, ``explanation``,
        ``layer_used``, ``status``, or ``None`` on failure.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras as tf_keras
    except ImportError:
        logger.warning("TensorFlow not available; skipping Grad-CAM.")
        return None

    try:
        # ── Find target conv layer and its containing sub-model ───────────
        conv_layer, sub_model = _find_target_conv_layer(model)
        if conv_layer is None:
            logger.warning("Grad-CAM: no suitable Conv2D layer found.")
            return None

        logger.info(
            "Grad-CAM: using layer '%s' (sub-model: %s)",
            conv_layer.name,
            sub_model.name if sub_model else "none",
        )

        # ── Build an inner grad-model from the sub-model's own graph ─────
        # MobileNetV2 has skip/residual connections so we cannot trace it
        # layer-by-layer.  Instead we build a Functional model purely from
        # MobileNetV2's own connected graph, then call the remaining top
        # layers (GAP, Dense, BN, Dropout, Dense) separately inside the tape.
        if sub_model is not None:
            try:
                inner_grad_model = tf_keras.Model(
                    inputs=sub_model.inputs,
                    outputs=[conv_layer.output, sub_model.output],
                )
            except Exception as exc:
                logger.warning("Grad-CAM: inner model build failed: %s", exc)
                return None

            # Collect the top layers that come *after* the sub-model
            top_layers = _get_top_layers(model, sub_model)

            input_tensor = tf.constant(input_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, base_features = inner_grad_model(
                    input_tensor, training=False
                )
                tape.watch(conv_outputs)

                # Run the remaining top layers to get the final prediction
                x = base_features
                for layer in top_layers:
                    try:
                        x = layer(x, training=False)
                    except TypeError:
                        x = layer(x)
                predictions = x

                class_score = (
                    predictions[0, 0]
                    if predicted_class_idx == 1
                    else 1.0 - predictions[0, 0]
                )

            grads = tape.gradient(class_score, conv_outputs)

        else:
            # Flat Functional model — standard approach
            try:
                grad_model = tf_keras.Model(
                    inputs=model.inputs,
                    outputs=[conv_layer.output, model.outputs[0]],
                )
            except Exception as exc:
                logger.warning("Grad-CAM: flat grad model build failed: %s", exc)
                return None

            input_tensor = tf.constant(input_batch, dtype=tf.float32)

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor, training=False)
                tape.watch(conv_outputs)
                class_score = (
                    predictions[0, 0]
                    if predicted_class_idx == 1
                    else 1.0 - predictions[0, 0]
                )

            grads = tape.gradient(class_score, conv_outputs)

        # ── Compute heatmap from gradients ────────────────────────────────
        if grads is None:
            logger.warning("Grad-CAM: gradient is None — skipping.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_np = conv_outputs[0]
        heatmap = tf.reduce_sum(
            conv_outputs_np * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1
        )
        heatmap = tf.maximum(heatmap, 0.0)

        heatmap_max = tf.reduce_max(heatmap)
        heatmap_np = (
            (heatmap / heatmap_max).numpy()
            if heatmap_max > 1e-8
            else np.zeros(heatmap.shape, dtype=np.float32)
        )

        # ── Load original image for overlay ──────────────────────────────
        with PILImage.open(io.BytesIO(original_image_bytes)) as orig:
            orig_rgb = orig.convert("RGB")
            orig_size = orig_rgb.size  # (width, height)

        heatmap_img = PILImage.fromarray(
            (heatmap_np * 255).astype(np.uint8), mode="L"
        ).resize(orig_size, PILImage.BILINEAR)
        heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0

        heatmap_colored = _apply_jet_colormap(heatmap_resized)

        orig_array = np.asarray(orig_rgb, dtype=np.float32)
        alpha = 0.4
        overlay_array = (
            orig_array * (1.0 - alpha) + heatmap_colored * alpha
        ).astype(np.uint8)
        overlay_img = PILImage.fromarray(overlay_array, mode="RGB")
        heatmap_only_img = PILImage.fromarray(heatmap_colored, mode="RGB")

        def _encode_pil(img: PILImage.Image) -> str:
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return (
                "data:image/png;base64,"
                + base64.b64encode(buf.getvalue()).decode("ascii")
            )

        logger.info("Grad-CAM: heatmap generated successfully.")
        return {
            "heatmap_base64": _encode_pil(heatmap_only_img),
            "overlay_base64": _encode_pil(overlay_img),
            "explanation": (
                "The highlighted regions (yellow/red) indicate the areas of the "
                "chest X-ray that most influenced the model's prediction.  Red "
                "signifies stronger influence on the decision.  This heatmap is "
                "an interpretability aid and does not represent a clinical diagnosis."
            ),
            "layer_used": conv_layer.name,
            "status": "generated",
        }

    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc, exc_info=True)
        return None


# ── Internal helpers ─────────────────────────────────────────────────────────


def _find_target_conv_layer(model: Any) -> tuple[Any | None, Any | None]:
    """Return (conv_layer, sub_model) for Grad-CAM.

    Searches nested sub-models (e.g. MobileNetV2 inside Sequential) first,
    then the top-level model.  Returns (None, None) if nothing is found.
    """
    # --- Search inside nested sub-models first ---------------------------
    for layer in model.layers:
        if not (hasattr(layer, "layers") and len(layer.layers) > 5):
            continue
        # Try named candidates inside this sub-model
        for candidate in _LAYER_CANDIDATES:
            try:
                inner = layer.get_layer(candidate)
                return inner, layer
            except (ValueError, AttributeError):
                pass
        # Fall back to the last Conv2D inside the sub-model
        last = _last_conv2d(layer)
        if last is not None:
            return last, layer

    # --- Try named candidates at the top level ---------------------------
    for candidate in _LAYER_CANDIDATES:
        try:
            return model.get_layer(candidate), None
        except (ValueError, AttributeError):
            pass

    # --- Last Conv2D anywhere in the top-level model ---------------------
    last = _last_conv2d(model)
    return last, None


def _get_top_layers(model: Any, sub_model: Any) -> list[Any]:
    """Return layers in *model* that come after *sub_model*."""
    top_layers: list[Any] = []
    found = False
    for layer in model.layers:
        if layer is sub_model:
            found = True
            continue
        if found:
            top_layers.append(layer)
    return top_layers


def _last_conv2d(container: Any) -> Any | None:
    """Walk the layer tree and return the last Conv2D found."""
    try:
        import tensorflow.keras.layers as tf_layers
    except ImportError:
        return None

    last: Any | None = None

    def _walk(parent: Any) -> None:
        nonlocal last
        for layer in getattr(parent, "layers", []):
            if hasattr(layer, "layers") and len(layer.layers) > 0:
                _walk(layer)
            elif isinstance(layer, tf_layers.Conv2D):
                last = layer

    try:
        _walk(container)
    except Exception:
        pass
    return last


def _apply_jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Apply the *jet* colormap to a single-channel ``[0, 1]`` heatmap.

    Returns an ``(H, W, 3)`` uint8 array.
    """
    try:
        import matplotlib as mpl

        rgba = mpl.colormaps["jet"](heatmap)
        return (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        idx = (heatmap * (_JET_LUT.shape[0] - 1)).astype(np.int32)
        idx = np.clip(idx, 0, _JET_LUT.shape[0] - 1)
        return (_JET_LUT[idx] * 255).astype(np.uint8)
