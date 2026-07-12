"""
ANN model definition for heart disease risk prediction.

Architecture: Input → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) →
Dropout(0.2) → Dense(1, sigmoid)

Binary classification: 0 = no heart disease, 1 = heart disease present.
"""
from __future__ import annotations

import keras
from keras import layers, Model


def build_ann(input_dim: int, learning_rate: float = 1e-3) -> Model:
    """Build and compile the heart disease ANN.

    Parameters
    ----------
    input_dim : int
        Number of input features (after one-hot encoding).
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    Compiled Keras Model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="heart_disease_ann")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
