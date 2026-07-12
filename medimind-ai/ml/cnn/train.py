"""
Training script for the Pneumonia Detection CNN.

Loads chest X-ray images, trains with transfer learning (MobileNetV2),
evaluates, and logs everything to MLflow.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import mlflow
import mlflow.keras
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from dotenv import load_dotenv
import keras

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.cnn.preprocess import (
    get_train_generator, get_val_generator, get_test_generator,
    compute_class_weights,
)
from ml.cnn.model import build_cnn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "clinical-copilot")
REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))
RAW_DATA_DIR = os.getenv("CNN_RAW_DATA", "data/raw/xray")


def train(
    epochs: int = 25,
    learning_rate: float = 1e-4,
    patience: int = 7,
) -> dict:
    """Train the CNN, log to MLflow, save best model.

    Returns
    -------
    dict with evaluation metrics.
    """
    # ── Data generators ──────────────────────────────────────────────────
    train_gen = get_train_generator(RAW_DATA_DIR)
    val_gen = get_val_generator(RAW_DATA_DIR)
    test_gen = get_test_generator(RAW_DATA_DIR)

    class_weights = compute_class_weights(str(Path(RAW_DATA_DIR) / "train"))
    print(f"[CNN] Class weights: {class_weights}")
    print(f"[CNN] Training samples: {train_gen.samples}")
    print(f"[CNN] Validation samples: {val_gen.samples}")
    print(f"[CNN] Test samples: {test_gen.samples}")

    # ── Build model ──────────────────────────────────────────────────────
    model = build_cnn(learning_rate=learning_rate)
    model.summary()

    # ── MLflow ───────────────────────────────────────────────────────────
    _mlruns = (PROJECT_ROOT / "mlruns").as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = _mlruns
    mlflow.set_tracking_uri(_mlruns)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cnn_pneumonia"):
        mlflow.log_params({
            "model_type": "CNN",
            "backbone": "MobileNetV2",
            "input_shape": "224x224x3",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "class_weights": json.dumps(class_weights),
        })

        # ── Train ────────────────────────────────────────────────────────
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
            ),
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Evaluate on test set ─────────────────────────────────────────
        test_gen.reset()
        y_proba = model.predict(test_gen).ravel()
        y_true = test_gen.classes
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

        print(f"\n[CNN] Evaluation metrics: {json.dumps(metrics, indent=2)}")
        print(f"\n[CNN] Classification report:\n{classification_report(y_true, y_pred)}")

        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(y_true, y_pred),
            "classification_report.txt",
        )

        # ── Save model ──────────────────────────────────────────────────
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        model_path = REGISTRY_DIR / "cnn_pneumonia.h5"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))
        mlflow.keras.log_model(model, "cnn_model")
        print(f"[CNN] Model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    train()
