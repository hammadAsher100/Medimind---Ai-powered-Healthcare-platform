"""
CNN model definition for chest X-ray pneumonia detection.

Uses MobileNetV2 (pretrained on ImageNet) with fine-tuning of the last
30 layers for better feature adaptation to X-ray domain.

Architecture:
  MobileNetV2 (top 30 layers unfrozen) → GlobalAveragePooling2D →
  Dense(128, relu) → BatchNorm → Dropout(0.4) → Dense(1, sigmoid)
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers, Model
from keras.applications import MobileNetV2


def precision(y_true, y_pred):
    """Precision metric."""
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + 1e-7)


def recall(y_true, y_pred):
    """Recall metric."""
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + 1e-7)


def f1_score(y_true, y_pred):
    """F1 score metric."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-7)


def build_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    fine_tune_from: int | None = 70,
) -> Model:
    """Build and compile the pneumonia detection CNN.

    Parameters
    ----------
    input_shape : tuple
        Image dimensions (H, W, C).
    learning_rate : float
        Adam optimizer learning rate.
    fine_tune_from : int, optional
        Unfreeze base layers starting from this layer index
        for fine-tuning.  Default 70 unfreezes top layers.

    Returns
    -------
    Compiled Keras Model.
    """
    # ── Base model ───────────────────────────────────────────────────────
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Fine-tune: freeze early layers, unfreeze top layers
    if fine_tune_from is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False
    else:
        base_model.trainable = False

    # ── Classification head ──────────────────────────────────────────────
    inputs = keras.Input(shape=input_shape)
    # Do not hardcode training=True, otherwise BN will fail on batch_size=1 during inference
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="pneumonia_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", precision, recall, f1_score],
    )

    return model
