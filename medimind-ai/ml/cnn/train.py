"""Reproducible MobileNetV2 training for pneumonia chest X-rays.

The production contract is intentionally explicit:

* labels: NORMAL=0, PNEUMONIA=1
* input: 224x224 RGB float32 in [0, 1]
* MobileNetV2 [-1, 1] normalization: embedded inside the saved model
* output: one sigmoid probability for PNEUMONIA
* artifact: cnn_pneumonia.h5 (kept outside Git and synchronized through S3)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import keras
import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
IMAGE_SIZE = (224, 224)
LABELS = {0: "NORMAL", 1: "PNEUMONIA"}
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.cnn.model import build_cnn, enable_fine_tuning


def set_reproducible_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def resolve_dataset_dir(configured_path: str | Path) -> Path:
    base = Path(configured_path).expanduser().resolve()
    nested = base / "chest_xray"
    dataset_dir = nested if nested.is_dir() else base
    for split in ("train", "val", "test"):
        for class_name in LABELS.values():
            required = dataset_dir / split / class_name
            if not required.is_dir():
                raise FileNotFoundError(f"Missing dataset directory: {required}")
    return dataset_dir


def collect_split(dataset_dir: Path, splits: tuple[str, ...]) -> tuple[list[str], np.ndarray, dict]:
    """Collect labeled paths and remove byte-identical duplicates deterministically."""
    paths: list[str] = []
    labels: list[int] = []
    seen_hashes: dict[str, tuple[str, int]] = {}
    duplicate_count = 0

    for split in splits:
        for label, class_name in LABELS.items():
            for image_path in sorted((dataset_dir / split / class_name).iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in seen_hashes:
                    original_path, original_label = seen_hashes[digest]
                    if original_label != label:
                        raise ValueError(
                            f"Identical image has conflicting labels: {original_path} and {image_path}"
                        )
                    duplicate_count += 1
                    continue
                seen_hashes[digest] = (str(image_path), label)
                paths.append(str(image_path))
                labels.append(label)

    counts = {LABELS[label]: labels.count(label) for label in LABELS}
    return paths, np.asarray(labels, dtype=np.int32), {
        "counts": counts,
        "duplicates_removed": duplicate_count,
        "hashes": set(seen_hashes),
    }


def assert_no_test_leakage(training_hashes: set[str], dataset_dir: Path) -> None:
    overlaps = []
    for class_name in LABELS.values():
        for image_path in sorted((dataset_dir / "test" / class_name).iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in training_hashes:
                    overlaps.append(str(image_path))
    if overlaps:
        raise ValueError(f"Held-out test leakage detected in {len(overlaps)} image(s): {overlaps[:3]}")


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    content = tf.io.read_file(path)
    image = tf.io.decode_image(content, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    return image, tf.cast(label, tf.float32)


def make_dataset(
    paths: list[str],
    labels: np.ndarray,
    batch_size: int,
    *,
    training: bool,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if training:
        augmentation = keras.Sequential(
            [
                keras.layers.RandomRotation(0.02, fill_mode="nearest", seed=SEED),
                keras.layers.RandomTranslation(0.03, 0.03, fill_mode="nearest", seed=SEED + 1),
                keras.layers.RandomZoom(0.05, fill_mode="nearest", seed=SEED + 2),
                keras.layers.RandomContrast(0.08, seed=SEED + 3),
            ],
            name="training_only_augmentation",
        )
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    return dataset.with_options(options)


def collect_probabilities(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict(dataset, verbose=1).reshape(-1)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset]).astype(np.int32)
    return labels, probabilities


def choose_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum_sensitivity: float = 0.92,
) -> tuple[float, dict]:
    """Meet a clinical sensitivity floor, then minimize false positives."""
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        candidates.append(
            {
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "balanced_accuracy": float((sensitivity + specificity) / 2),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
            }
        )

    eligible = [item for item in candidates if item["sensitivity"] >= minimum_sensitivity]
    pool = eligible or candidates
    selected = max(
        pool,
        key=lambda item: (item["specificity"], item["balanced_accuracy"], item["f1"]),
    )
    return selected["threshold"], selected


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def callbacks(checkpoint_path: Path, patience: int) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def best_history_metrics(history: keras.callbacks.History) -> dict:
    values = history.history
    best_index = int(np.argmin(values["val_loss"]))
    return {
        "best_epoch": best_index + 1,
        "loss": float(values["loss"][best_index]),
        "accuracy": float(values["accuracy"][best_index]),
        "precision": float(values["precision"][best_index]),
        "recall": float(values["recall"][best_index]),
        "auc": float(values["auc"][best_index]),
        "val_loss": float(values["val_loss"][best_index]),
        "val_accuracy": float(values["val_accuracy"][best_index]),
        "val_precision": float(values["val_precision"][best_index]),
        "val_recall": float(values["val_recall"][best_index]),
        "val_auc": float(values["val_auc"][best_index]),
    }


def known_image_checks(model: keras.Model, threshold: float, dataset_dir: Path) -> list[dict]:
    checks = []
    for label, class_name in LABELS.items():
        candidates = sorted(
            path
            for path in (dataset_dir / "test" / class_name).iterdir()
            if path.suffix.lower() in SUPPORTED_EXTENSIONS
        )[:5]
        dataset = make_dataset([str(path) for path in candidates], np.asarray([label] * len(candidates)), 5, training=False)
        _, probabilities = collect_probabilities(model, dataset)
        for path, probability in zip(candidates, probabilities):
            predicted_label = int(probability >= threshold)
            checks.append(
                {
                    "expected": class_name,
                    "predicted": LABELS[predicted_label],
                    "pneumonia_probability": float(probability),
                    "passed": predicted_label == label,
                    "sample": path.name,
                }
            )
    return checks


def train(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    head_epochs: int = 6,
    fine_tune_epochs: int = 10,
    batch_size: int = 16,
    patience: int = 3,
) -> dict:
    set_reproducible_seed()
    dataset_dir = resolve_dataset_dir(dataset_path)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_paths, all_labels, audit = collect_split(dataset_dir, ("train", "val"))
    assert_no_test_leakage(audit["hashes"], dataset_dir)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=0.20,
        random_state=SEED,
        stratify=all_labels,
    )
    test_paths, test_labels, test_audit = collect_split(dataset_dir, ("test",))

    train_dataset = make_dataset(train_paths, train_labels, batch_size, training=True)
    validation_dataset = make_dataset(validation_paths, validation_labels, batch_size, training=False)
    test_dataset = make_dataset(test_paths, test_labels, batch_size, training=False)

    class_values = np.unique(train_labels)
    weights = compute_class_weight(class_weight="balanced", classes=class_values, y=train_labels)
    class_weights = {int(label): float(weight) for label, weight in zip(class_values, weights)}

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "label_mapping": LABELS,
                "combined_source_counts": audit["counts"],
                "duplicates_removed": audit["duplicates_removed"],
                "train_samples": len(train_paths),
                "validation_samples": len(validation_paths),
                "test_counts": test_audit["counts"],
                "class_weights": class_weights,
                "seed": SEED,
            },
            indent=2,
        )
    )

    head_checkpoint = output_path.with_name("cnn_pneumonia_head.keras")
    fine_tune_checkpoint = output_path.with_name("cnn_pneumonia_finetune.keras")
    model = build_cnn(learning_rate=1e-3, fine_tune_layers=0)

    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=callbacks(head_checkpoint, patience),
        verbose=1,
    )

    head_model = keras.models.load_model(head_checkpoint, compile=False)
    head_labels, head_probabilities = collect_probabilities(head_model, validation_dataset)
    head_threshold, head_selection = choose_threshold(head_labels, head_probabilities)

    model.set_weights(head_model.get_weights())
    enable_fine_tuning(model, fine_tune_layers=20, learning_rate=1e-5)
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks(fine_tune_checkpoint, patience),
        verbose=1,
    )

    fine_tuned_model = keras.models.load_model(fine_tune_checkpoint, compile=False)
    fine_labels, fine_probabilities = collect_probabilities(fine_tuned_model, validation_dataset)
    fine_threshold, fine_selection = choose_threshold(fine_labels, fine_probabilities)

    candidates = [
        ("head", head_model, head_threshold, head_selection),
        ("fine_tuned", fine_tuned_model, fine_threshold, fine_selection),
    ]
    selected_stage, selected_model, selected_threshold, validation_selection = max(
        candidates,
        key=lambda item: (
            item[3]["balanced_accuracy"],
            item[3]["f1"],
            item[3]["specificity"],
        ),
    )

    test_true, test_probabilities = collect_probabilities(selected_model, test_dataset)
    test_metrics = classification_metrics(test_true, test_probabilities, selected_threshold)
    validation_true, validation_probabilities = collect_probabilities(selected_model, validation_dataset)
    validation_metrics = classification_metrics(
        validation_true,
        validation_probabilities,
        selected_threshold,
    )
    sample_checks = known_image_checks(selected_model, selected_threshold, dataset_dir)

    acceptance = {
        "roc_auc_at_least_0_90": test_metrics["roc_auc"] >= 0.90,
        "pneumonia_sensitivity_at_least_0_90": test_metrics["recall_sensitivity"] >= 0.90,
        "normal_specificity_at_least_0_70": test_metrics["specificity"] >= 0.70,
        "known_images_at_least_80_percent": sum(item["passed"] for item in sample_checks)
        >= 0.8 * len(sample_checks),
    }
    if not all(acceptance.values()):
        raise RuntimeError(
            "Retrained model failed acceptance criteria; existing production artifact was not replaced. "
            + json.dumps({"acceptance": acceptance, "test_metrics": test_metrics})
        )

    candidate_path = output_path.with_name(f"{output_path.stem}.candidate{output_path.suffix}")
    selected_model.save(candidate_path, include_optimizer=False)
    with h5py.File(candidate_path, "a") as model_file:
        model_file.attrs["medimind_classification_threshold"] = float(selected_threshold)
        model_file.attrs["medimind_negative_label"] = LABELS[0]
        model_file.attrs["medimind_positive_label"] = LABELS[1]
        model_file.attrs["medimind_input_contract"] = "224x224 RGB float32 [0,1]"
    # Refuse to replace a working artifact unless the candidate reloads.
    keras.models.load_model(candidate_path, compile=False)
    candidate_path.replace(output_path)
    result = {
        "architecture": "MobileNetV2 transfer learning with embedded normalization",
        "selected_stage": selected_stage,
        "selected_threshold": float(selected_threshold),
        "validation_threshold_selection": validation_selection,
        "head_training": best_history_metrics(head_history),
        "fine_tune_training": best_history_metrics(fine_tune_history),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "acceptance": acceptance,
        "known_image_checks": sample_checks,
        "model_path": str(output_path),
        "model_filename": output_path.name,
    }
    print(json.dumps(result, indent=2))

    history_path = PROJECT_ROOT / "mlruns" / "cnn_pneumonia_training_result.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("CNN_RAW_DATA", str(PROJECT_ROOT / "data" / "raw" / "xray")),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "ml" / "registry" / "cnn_pneumonia.h5"),
    )
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.data_dir,
        arguments.output,
        head_epochs=arguments.head_epochs,
        fine_tune_epochs=arguments.fine_tune_epochs,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
    )
