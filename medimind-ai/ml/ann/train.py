"""
Training script for the Heart Disease ANN.

Loads UCI Heart Disease data, preprocesses, trains the ANN with early stopping,
evaluates, and logs everything to MLflow.  Saves the best model to ml/registry/.
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

# Append project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.ann.preprocess import load_raw_data, preprocess_and_split
from ml.ann.model import build_ann

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "clinical-copilot")
REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))
RAW_DATA_PATH = os.getenv("ANN_RAW_DATA", "data/raw/tabular/heart.csv")


def train(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    patience: int = 15,
) -> dict:
    """Train the ANN, log to MLflow, save best model.

    Returns
    -------
    dict with evaluation metrics.
    """
    # ── Load and preprocess ──────────────────────────────────────────────
    df = load_raw_data(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(
        df, save_artifacts=True
    )
    input_dim = X_train.shape[1]

    print(f"[ANN] Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"[ANN] Feature count: {input_dim}, Features: {feature_names[:5]}...")
    print(f"[ANN] Target distribution — train: {np.bincount(y_train.astype(int))}, test: {np.bincount(y_test.astype(int))}")

    # ── Class weights ────────────────────────────────────────────────────
    n_neg, n_pos = np.bincount(y_train.astype(int))
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}
    print(f"[ANN] Class weights: {class_weight}")

    # ── Build model ──────────────────────────────────────────────────────
    model = build_ann(input_dim=input_dim, learning_rate=learning_rate)
    model.summary()

    # ── MLflow logging ───────────────────────────────────────────────────
    _mlruns = (PROJECT_ROOT / "mlruns").as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = _mlruns
    mlflow.set_tracking_uri(_mlruns)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="ann_heart_disease") as run:
        mlflow.log_params({
            "model_type": "ANN",
            "input_dim": input_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "patience": patience,
            "architecture": "64-32-1",
        })

        # ── Train ────────────────────────────────────────────────────────
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
        ]

        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # ── Evaluate ─────────────────────────────────────────────────────
        y_proba = model.predict(X_test).ravel()
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print(f"\n[ANN] Evaluation metrics: {json.dumps(metrics, indent=2)}")
        print(f"\n[ANN] Classification report:\n{classification_report(y_test, y_pred)}")
        print(f"[ANN] Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

        mlflow.log_metrics(metrics)
        mlflow.log_text(
            classification_report(y_test, y_pred),
            "classification_report.txt",
        )

        # ── Save model ───────────────────────────────────────────────────
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        model_path = REGISTRY_DIR / "ann_heart_risk.h5"
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))
        print(f"[ANN] Model saved to {model_path}")

        # Log model to MLflow registry
        mlflow.keras.log_model(model, "ann_model")

    return metrics


if __name__ == "__main__":
    train()
