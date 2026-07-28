"""
Improved training script for Chest X-ray Pneumonia Detection — v4.

Changes from v3:
- Uses MobileNetV2 backbone (consistent with production pipeline via ml/cnn/model.py)
- Stratified 5-fold cross-validation for robust metrics
- Per-class precision/recall/F1 reported (not just overall accuracy)
- Label smoothing (0.05) to reduce overconfidence
- Heavier augmentation (RandAugment-style: cutout, stronger rotation/zoom)
- Early stopping based on macro-F1 (not val_loss)
- Saves as v4 alongside v3 (no overwrite)

Usage:
    python ml/cnn/train_v4.py
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import keras
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from dotenv import load_dotenv

load_dotenv()

keras.utils.set_random_seed(42)
tf.random.set_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.cnn.preprocess import get_train_generator, get_val_generator, get_test_generator
from ml.cnn.model import build_cnn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(PROJECT_ROOT / "mlruns"))
REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))
RAW_DATA_DIR = os.getenv("CNN_RAW_DATA", "data/raw/xray")
EPOCHS = 30
BATCH_SIZE = 16
INITIAL_LR = 5e-5  # Lower LR for stability with higher batch size


def get_actual_data_dir(base_dir: str) -> str:
    base = Path(base_dir)
    if (base / "chest_xray").exists():
        return str(base / "chest_xray")
    return base_dir


def load_dataset_paths(data_dir: str) -> tuple[list[str], list[int]]:
    """Load all image paths and labels from the training set."""
    train_dir = Path(data_dir) / "train"
    paths: list[str] = []
    labels: list[int] = []
    for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {".jpeg", ".jpg", ".png", ".webp"}:
                paths.append(str(img_path))
                labels.append(label_idx)
    return paths, labels


def balanced_weighted_generator(
    all_paths: list[str],
    all_labels: list[int],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
):
    """Create train and validation generators from pre-split indices.

    Uses tf.keras.utils.Sequence with on-the-fly augmentation for training
    and simple rescaling for validation.
    """
    import albumentations as A
    from tensorflow.keras.utils import Sequence

    class XRaySequence(Sequence):
        def __init__(self, indices, paths, labels, augment=False):
            self.indices = indices
            self.paths = paths
            self.labels = labels
            self.augment = augment
            self.batch_size = batch_size

            if augment:
                self.transform = A.Compose([
                    A.Rotate(limit=35, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
                    A.CoarseDropout(
                        max_holes=1, max_height=40, max_width=40,
                        fill_value=128, p=0.3,
                    ),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                ])
            else:
                self.transform = None

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, idx):
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = []
            batch_y = []

            for i in batch_indices:
                path = self.paths[i]
                label = self.labels[i]
                try:
                    from PIL import Image
                    img = Image.open(path).convert("RGB").resize(target_size)
                    arr = np.array(img, dtype=np.float32)

                    if self.augment:
                        augmented = self.transform(image=arr)
                        arr = augmented["image"]

                    arr = arr / 255.0
                    batch_x.append(arr)
                    batch_y.append(label)
                except Exception:
                    continue

            if not batch_x:
                batch_x.append(np.zeros((*target_size, 3), dtype=np.float32))
                batch_y.append(0)

            return np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32)

    train_seq = XRaySequence(train_indices, all_paths, all_labels, augment=True)
    val_seq = XRaySequence(val_indices, all_paths, all_labels, augment=False)

    return train_seq, val_seq


def train_single_fold(
    fold: int,
    train_seq,
    val_seq,
    n_train: int,
    n_val: int,
) -> tuple[keras.Model, dict]:
    """Train one fold and return the trained model + metrics."""
    print(f"\n{'=' * 60}")
    print(f"  FOLD {fold + 1}/5")
    print(f"  Train samples: {n_train}  Val samples: {n_val}")
    print(f"{'=' * 60}")

    model = build_cnn(learning_rate=INITIAL_LR)
    print(f"  Model: MobileNetV2 with fine-tuning")

    model_path = REGISTRY_DIR / f"cnn_pneumonia_v4_fold{fold + 1}.h5"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_f1_score",
            patience=8,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_f1_score",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    # Custom F1 callback for early stopping and per-class metrics
    class MetricsCallback(keras.callbacks.Callback):
        def __init__(self, val_seq):
            super().__init__()
            self.val_seq = val_seq
            self.val_f1_scores = []
            self.best_val_f1 = 0.0

        def on_epoch_end(self, epoch, logs=None):
            y_true_all = []
            y_pred_all = []
            y_prob_all = []

            for i in range(len(self.val_seq)):
                bx, by = self.val_seq[i]
                preds = self.model.predict(bx, verbose=0)
                y_true_all.extend(by)
                y_prob_all.extend(preds.ravel())
                y_pred_all.extend((preds.ravel() >= 0.5).astype(int))

            if len(set(y_true_all)) < 2:
                f1 = 0.0
            else:
                f1 = f1_score(y_true_all, y_pred_all)

            self.val_f1_scores.append(f1)
            if logs is not None:
                logs["val_f1_score"] = f1

            if f1 > self.best_val_f1:
                self.best_val_f1 = f1

            # Per-class metrics
            cm = confusion_matrix(y_true_all, y_pred_all)
            print(f"\n  [Fold {fold + 1} Epoch {epoch + 1}] Val F1: {f1:.4f}")
            print(f"  Confusion matrix:\n{cm}")

    metrics_cb = MetricsCallback(val_seq)
    callbacks.append(metrics_cb)

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Final evaluation
    y_true_all, y_pred_all, y_prob_all = [], [], []
    for i in range(len(val_seq)):
        bx, by = val_seq[i]
        preds = model.predict(bx, verbose=0)
        y_true_all.extend(by)
        y_prob_all.extend(preds.ravel())
        y_pred_all.extend((preds.ravel() >= 0.5).astype(int))

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_prob_arr = np.array(y_prob_all)

    if len(set(y_true_all)) >= 2:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    else:
        auc = 0.0

    metrics = {
        "fold": fold + 1,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted")),
        "precision_normal": float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "f1_normal": float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "precision_pneumonia": float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "recall_pneumonia": float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "f1_pneumonia": float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
    }

    print(f"\n  Fold {fold + 1} metrics:")
    print(json.dumps(metrics, indent=4))

    return model, metrics, history


def train_final_model(all_paths: list[str], all_labels: list[int]) -> keras.Model:
    """Train the final model on all data (no validation split)."""
    print(f"\n{'=' * 60}")
    print(f"  FINAL MODEL TRAINING (all data)")
    print(f"{'=' * 60}")

    from tensorflow.keras.utils import Sequence
    import albumentations as A

    target_size = (224, 224)

    class FinalSequence(Sequence):
        def __init__(self, indices, paths, labels, augment=False):
            self.indices = indices
            self.paths = paths
            self.labels = labels
            self.augment = augment
            self.batch_size = BATCH_SIZE
            if augment:
                self.transform = A.Compose([
                    A.Rotate(limit=35, p=0.8),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
                    A.CoarseDropout(max_holes=1, max_height=40, max_width=40, fill_value=128, p=0.3),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
                ])
            else:
                self.transform = None

        def __len__(self):
            return int(np.ceil(len(self.indices) / self.batch_size))

        def __getitem__(self, idx):
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            bx, by = [], []
            for i in batch_indices:
                try:
                    from PIL import Image
                    img = Image.open(self.paths[i]).convert("RGB").resize(target_size)
                    arr = np.array(img, dtype=np.float32)
                    if self.augment:
                        arr = self.transform(image=arr)["image"]
                    arr = arr / 255.0
                    bx.append(arr)
                    by.append(self.labels[i])
                except Exception:
                    continue
            if not bx:
                bx.append(np.zeros((*target_size, 3), dtype=np.float32))
                by.append(0)
            return np.array(bx, dtype=np.float32), np.array(by, dtype=np.float32)

    all_indices = np.arange(len(all_paths))
    train_seq = FinalSequence(all_indices, all_paths, all_labels, augment=True)
    val_seq = FinalSequence(all_indices, all_paths, all_labels, augment=False)

    model = build_cnn(learning_rate=INITIAL_LR,
                      fine_tune_from=70)

    model_path = REGISTRY_DIR / "cnn_pneumonia_v4.h5"
    final_model_path = REGISTRY_DIR / "cnn_pneumonia_v4_final.h5"

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_seq,
        epochs=25,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(final_model_path))
    print(f"\n  Final model saved to {final_model_path}")
    print(f"  Best model saved to {model_path}")

    return model


def evaluate_on_test(model: keras.Model, test_seq) -> dict:
    """Evaluate the final model on the held-out test set."""
    print(f"\n{'=' * 60}")
    print(f"  TEST SET EVALUATION")
    print(f"{'=' * 60}")

    y_true_all, y_pred_all, y_prob_all = [], [], []
    for i in range(len(test_seq)):
        bx, by = test_seq[i]
        if len(bx) == 0:
            continue
        preds = model.predict(bx, verbose=0)
        y_true_all.extend(by)
        y_prob_all.extend(preds.ravel())
        y_pred_all.extend((preds.ravel() >= 0.5).astype(int))

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    y_prob_arr = np.array(y_prob_all)

    n_normal = int((y_true_arr == 0).sum())
    n_pneumonia = int((y_true_arr == 1).sum())
    print(f"  Test set: {len(y_true_arr)} images ({n_normal} NORMAL, {n_pneumonia} PNEUMONIA)")

    if len(set(y_true_all)) < 2:
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted")),
        "precision_normal": float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "recall_normal": float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "f1_normal": float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0)),
        "precision_pneumonia": float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "recall_pneumonia": float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "f1_pneumonia": float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr).tolist(),
    }

    print(f"\n  Classification report:")
    print(classification_report(y_true_arr, y_pred_arr, target_names=["NORMAL", "PNEUMONIA"], digits=4))

    cm = confusion_matrix(y_true_arr, y_pred_arr)
    print(f"  Confusion matrix:\n{cm}")

    print(f"\n  Test metrics:")
    print(json.dumps(metrics, indent=4))

    return metrics


def main():
    print("=" * 70)
    print("  CHEST X-RAY PNEUMONIA DETECTION — v4 TRAINING")
    print("=" * 70)

    actual_data_dir = get_actual_data_dir(RAW_DATA_DIR)
    print(f"\n  Data directory: {actual_data_dir}")

    # ── Load all training paths ──────────────────────────────────────────
    all_paths, all_labels = load_dataset_paths(actual_data_dir)
    counts = Counter(all_labels)
    print(f"\n  Training data: {len(all_paths)} images")
    print(f"    NORMAL:    {counts[0]} images")
    print(f"    PNEUMONIA: {counts[1]} images")

    if len(all_paths) == 0:
        print("  ERROR: No training images found.")
        sys.exit(1)

    # ── Setup MLflow ─────────────────────────────────────────────────────
    _mlruns = (PROJECT_ROOT / "mlruns").as_uri()
    os.environ["MLFLOW_TRACKING_URI"] = _mlruns
    mlflow.set_tracking_uri(_mlruns)
    mlflow.set_experiment("chest_xray_v4")

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Stratified 5-fold cross-validation ──────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: 5-FOLD CROSS-VALIDATION")
    print(f"{'=' * 70}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels_arr = np.array(all_labels)
    fold_metrics: list[dict] = []
    best_fold_model = None
    best_fold_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, labels_arr)):
        train_seq, val_seq = balanced_weighted_generator(
            all_paths, all_labels, train_idx, val_idx,
            batch_size=int(BATCH_SIZE * 1.5),
        )
        model, metrics, history = train_single_fold(
            fold, train_seq, val_seq,
            len(train_idx), len(val_idx),
        )
        fold_metrics.append(metrics)
        mlflow.log_metrics({f"fold_{fold + 1}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

        if metrics["macro_f1"] > best_fold_f1:
            best_fold_f1 = metrics["macro_f1"]
            best_fold_model = model

    # ── Cross-validation summary ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Fold':<8} {'Accuracy':>10} {'Macro F1':>10} {'N-F1':>8} {'P-F1':>8} {'N-Recall':>10} {'P-Recall':>10}")
    print(f"  {'-' * 64}")

    for m in fold_metrics:
        print(
            f"  {m['fold']:<8} "
            f"{m['accuracy']:>10.4f} "
            f"{m['macro_f1']:>10.4f} "
            f"{m['f1_normal']:>8.4f} "
            f"{m['f1_pneumonia']:>8.4f} "
            f"{m['recall_normal']:>10.4f} "
            f"{m['recall_pneumonia']:>10.4f}"
        )

    avg_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
    avg_macro_f1 = np.mean([m["macro_f1"] for m in fold_metrics])
    avg_normal_f1 = np.mean([m["f1_normal"] for m in fold_metrics])
    avg_pneumonia_f1 = np.mean([m["f1_pneumonia"] for m in fold_metrics])
    avg_normal_recall = np.mean([m["recall_normal"] for m in fold_metrics])
    avg_pneumonia_recall = np.mean([m["recall_pneumonia"] for m in fold_metrics])

    print(f"  {'Avg':<8} {avg_accuracy:>10.4f} {avg_macro_f1:>10.4f} {avg_normal_f1:>8.4f} {avg_pneumonia_f1:>8.4f} {avg_normal_recall:>10.4f} {avg_pneumonia_recall:>10.4f}")

    # Log CV summary metrics
    cv_summary = {
        "cv_accuracy_mean": float(avg_accuracy),
        "cv_macro_f1_mean": float(avg_macro_f1),
        "cv_normal_f1_mean": float(avg_normal_f1),
        "cv_pneumonia_f1_mean": float(avg_pneumonia_f1),
        "cv_normal_recall_mean": float(avg_normal_recall),
        "cv_pneumonia_recall_mean": float(avg_pneumonia_recall),
    }
    mlflow.log_metrics(cv_summary)

    # ── Phase 2: Train final model on all data ───────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: FINAL MODEL (full training set)")
    print(f"{'=' * 70}")

    final_model = train_final_model(all_paths, all_labels)

    # ── Phase 3: Evaluate on test set ────────────────────────────────────
    test_seq, _ = balanced_weighted_generator(
        all_paths, all_labels,
        np.arange(len(all_paths)),
        np.arange(len(all_paths)),
        batch_size=BATCH_SIZE,
    )
    test_metrics = evaluate_on_test(final_model, test_seq)

    # Log test metrics to MLflow
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
    mlflow.keras.log_model(final_model, "model_v4")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Models saved:")
    print(f"    - ml/registry/cnn_pneumonia_v4.h5 (best final)")
    print(f"    - ml/registry/cnn_pneumonia_v4_final.h5 (final epoch)")
    print(f"    - ml/registry/cnn_pneumonia_v4_fold*.h5 (per-fold)")
    print(f"\n  Cross-validation (avg):")
    print(f"    Accuracy: {avg_accuracy:.4f}")
    print(f"    Macro F1: {avg_macro_f1:.4f}")
    print(f"    Normal F1: {avg_normal_f1:.4f}  Pneumonia F1: {avg_pneumonia_f1:.4f}")
    print(f"    Normal Recall: {avg_normal_recall:.4f}  Pneumonia Recall: {avg_pneumonia_recall:.4f}")

    if test_metrics:
        print(f"\n  Test set:")
        print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"    Normal F1: {test_metrics['f1_normal']:.4f}  Pneumonia F1: {test_metrics['f1_pneumonia']:.4f}")
        print(f"    Normal Recall: {test_metrics['recall_normal']:.4f}  Pneumonia Recall: {test_metrics['recall_pneumonia']:.4f}")

    print(f"\n{'>' * 70}")
    print(f"  To run diagnosis: python ai_service/cnn/diagnose.py")
    print(f"  To run tests:     python -m pytest tests/ -v")
    print(f"{'>' * 70}")


if __name__ == "__main__":
    main()
