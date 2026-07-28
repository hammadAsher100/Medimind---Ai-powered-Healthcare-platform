"""
Training script for the Pneumonia Detection CNN.

Loads chest X-ray images, trains with transfer learning (MobileNetV2),
evaluates, and logs everything to MLflow. Rebuilt from scratch.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import keras
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
from dotenv import load_dotenv

load_dotenv()

# Fixed random seed for reproducibility
keras.utils.set_random_seed(42)
tf.random.set_seed(42)
np.random.seed(42)

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

def get_actual_data_dir(base_dir: str) -> str:
    base = Path(base_dir)
    if (base / "chest_xray").exists():
        return str(base / "chest_xray")
    return base_dir

def train(
    epochs: int = 15,
    learning_rate: float = 1e-4,
    patience: int = 5,
) -> dict:
    """Train the CNN, log to MLflow, save best model."""
    actual_data_dir = get_actual_data_dir(RAW_DATA_DIR)
    
    train_gen = get_train_generator(actual_data_dir)
    val_gen = get_val_generator(actual_data_dir)
    test_gen = get_test_generator(actual_data_dir)

    class_weights = compute_class_weights(str(Path(actual_data_dir) / "train"))
    print(f"[CNN] Class weights: {class_weights}")
    print(f"[CNN] Training samples: {train_gen.samples}")
    print(f"[CNN] Validation samples: {val_gen.samples}")
    print(f"[CNN] Test samples: {test_gen.samples}")

    model = build_cnn(learning_rate=learning_rate)
    model.summary()

    _mlruns = (PROJECT_ROOT / "mlruns").as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = _mlruns
    mlflow.set_tracking_uri(_mlruns)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="cnn_pneumonia_rebuild"):
        mlflow.log_params({
            "model_type": "CNN",
            "backbone": "MobileNetV2",
            "input_shape": "224x224x3",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "patience": patience,
            "class_weights": json.dumps(class_weights),
            "seed": 42
        })

        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        model_path = REGISTRY_DIR / "cnn_pneumonia.h5"

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

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
        
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n[CNN] Evaluation metrics: {json.dumps(metrics, indent=2)}")
        print(f"\n[CNN] Confusion Matrix:\n{cm}")
        print(f"\n[CNN] Classification report:\n{classification_report(y_true, y_pred)}")

        mlflow.log_metrics(metrics)
        mlflow.log_text(
            f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{classification_report(y_true, y_pred)}",
            "evaluation_report.txt",
        )

        # Ensure model is saved to the final path if checkpointing didn't trigger for some reason
        if not model_path.exists():
            model.save(str(model_path))

        mlflow.log_artifact(str(model_path))
        mlflow.keras.log_model(model, "cnn_model")
        print(f"[CNN] Best model saved to {model_path}")

    return metrics

if __name__ == "__main__":
    train(epochs=15)
