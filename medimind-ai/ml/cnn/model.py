"""
CNN model definition for chest X-ray pneumonia detection.

Uses MobileNetV2 (pretrained on ImageNet) with a lightweight classification
head. Input normalization is embedded in the model so training and production
both accept RGB float images in the [0, 1] range.

Architecture:
  [0,1] to [-1,1] rescaling → MobileNetV2 →
  GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)

Training-only augmentation is applied by ``ml/cnn/train.py`` and is not
serialized into the production artifact.
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
    learning_rate: float = 1e-3,
    fine_tune_layers: int = 0,
) -> Model:
    """Build and compile the pneumonia detection CNN.

    Parameters
    ----------
    input_shape : tuple
        Image dimensions (H, W, C).
    learning_rate : float
        Adam optimizer learning rate.
    fine_tune_layers : int
        Number of final MobileNetV2 layers to unfreeze. Batch-normalization
        layers remain frozen for stable small-batch fine-tuning.

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

    base_model.trainable = fine_tune_layers > 0
    if fine_tune_layers > 0:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    # ── Classification head ──────────────────────────────────────────────
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_preprocessing")(inputs)
    # Batch normalization stays in inference mode during transfer learning.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs, outputs, name="pneumonia_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def enable_fine_tuning(model: Model, fine_tune_layers: int = 20, learning_rate: float = 1e-5) -> None:
    """Unfreeze the final non-BN backbone layers and recompile at a low LR."""
    base_model = model.get_layer("mobilenetv2_1.00_224")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
