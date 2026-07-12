"""
Grad-CAM implementation for the Pneumonia Detection CNN.

Generates a class-activation heatmap overlay on the original chest X-ray
by computing gradients of the predicted class w.r.t. the last convolutional
layer's feature maps.

This is a manual implementation targeting specific MobileNetV2 layer names,
not relying on third-party Grad-CAM libraries, for maximum Keras 3 compatibility.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import keras
import cv2
from PIL import Image

REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))

_model: Optional[keras.Model] = None


def _load_model() -> keras.Model:
    """Load the trained CNN from registry (cached)."""
    global _model
    if _model is None:
        model_path = REGISTRY_DIR / "cnn_pneumonia.h5"
        _model = keras.saving.load_model(str(model_path))
    return _model


def _find_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last convolutional layer in the model.

    Walks the model's layers (including nested models like MobileNetV2)
    in reverse order and returns the first Conv2D layer name found.
    """
    # Check nested models first (MobileNetV2 is typically model.layers[1])
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            # Nested model (e.g., MobileNetV2 backbone)
            for sub_layer in reversed(layer.layers):
                if "conv" in sub_layer.name.lower() and len(sub_layer.output_shape) == 4:
                    return sub_layer.name
        if "conv" in layer.name.lower() and len(layer.output_shape) == 4:
            return layer.name

    raise ValueError("No convolutional layer found in model")


def generate_gradcam(
    image_array: np.ndarray,
    model: keras.Model | None = None,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for a preprocessed image.

    Parameters
    ----------
    image_array : np.ndarray
        Shape (1, 224, 224, 3), normalised to [0, 1].
    model : keras.Model, optional
        If None, loads from registry.
    last_conv_layer_name : str, optional
        Name of the last conv layer.  Auto-detected if not provided.

    Returns
    -------
    np.ndarray — heatmap of shape (224, 224) with values in [0, 1].
    """
    import tensorflow as tf

    if model is None:
        model = _load_model()

    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    # Build a sub-model that maps input → [last_conv_output, predictions]
    # For nested MobileNetV2, we need to find the actual layer object
    last_conv_layer = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if sub_layer.name == last_conv_layer_name:
                    last_conv_layer = sub_layer
                    break
        if layer.name == last_conv_layer_name:
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model")

    # Create gradient model
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, 0]  # Binary output (pneumonia probability)

    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def create_heatmap_overlay(
    original_image_bytes: bytes,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    save_path: str | None = None,
) -> bytes:
    """Overlay the Grad-CAM heatmap on the original X-ray image.

    Parameters
    ----------
    original_image_bytes : bytes
        Raw image file bytes.
    heatmap : np.ndarray
        Grad-CAM heatmap (H, W) with values in [0, 1].
    alpha : float
        Opacity of the heatmap overlay.
    save_path : str, optional
        If provided, save the overlay image to this path.

    Returns
    -------
    PNG image bytes of the overlay.
    """
    # Load original image
    img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Apply colormap (jet: blue=cold=low, red=hot=high)
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = np.uint8(img_array * (1 - alpha) + heatmap_colored * alpha)

    # Convert to PNG bytes
    overlay_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)
    img_bytes = buf.read()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_bytes)

    return img_bytes
