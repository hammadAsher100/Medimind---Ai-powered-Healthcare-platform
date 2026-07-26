"""
Diagnostic script for the Chest X-ray Pneumonia CNN.

Loads each available model version, runs inference on the FULL test set,
and reports per-class precision/recall/F1, confusion matrix, calibration,
and overfitting indicators.

Usage:
    python ai_service/cnn/diagnose.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    brier_score_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "ai_service"))

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "xray" / "chest_xray"
TEST_DIR = DATA_DIR / "test"
REGISTRY_DIR = PROJECT_ROOT / "ml" / "registry"

IMAGE_SIZE = (224, 224)


def load_test_images(split_dir: Path) -> tuple[list[Path], list[int], list[str]]:
    """Load all test images, return (paths, labels, filenames).

    Labels: 0 = NORMAL, 1 = PNEUMONIA
    """
    paths: list[Path] = []
    labels: list[int] = []
    filenames: list[str] = []

    for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} does not exist, skipping")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                paths.append(img_path)
                labels.append(label_idx)
                filenames.append(img_path.name)

    return paths, labels, filenames


def preprocess(image_path: Path) -> np.ndarray:
    """Load and preprocess a single image to (1, 224, 224, 3) float32 [0,1]."""
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def diagnose_model(model_path: Path, model_label: str) -> dict:
    """Run full diagnostics on one model against the test set."""
    print(f"\n{'=' * 70}")
    print(f"  MODEL: {model_label}  ({model_path.name})")
    print(f"{'=' * 70}")

    if not model_path.exists():
        print(f"  SKIPPED — file not found: {model_path}")
        return {}

    # Load model
    try:
        import keras
        model = keras.models.load_model(str(model_path), compile=False)
        print(f"  Loaded successfully")
    except Exception as exc:
        print(f"  FAILED to load: {exc}")
        return {}

    # Load test data
    print(f"\n  Loading test set from {TEST_DIR} ...")
    paths, true_labels, filenames = load_test_images(TEST_DIR)
    n_normal = sum(1 for l in true_labels if l == 0)
    n_pneumonia = sum(1 for l in true_labels if l == 1)
    print(f"  Test set: {len(paths)} images ({n_normal} NORMAL, {n_pneumonia} PNEUMONIA)")
    print(f"  Test imbalance ratio: {n_pneumonia / max(n_normal, 1):.2f}:1 (PNEUMONIA:NORMAL)")

    # Run inference
    predictions: list[int] = []
    probabilities: list[float] = []

    for img_path in paths:
        batch = preprocess(img_path)
        raw = model.predict(batch, verbose=0)
        prob = float(np.asarray(raw).reshape(-1)[0])
        probabilities.append(prob)
        predictions.append(1 if prob >= 0.5 else 0)

    # Per-class metrics
    print(f"\n  --- Classification Report ---")
    report = classification_report(
        true_labels, predictions,
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4,
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"  {line}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n  --- Confusion Matrix ---")
    print(f"                   Predicted")
    print(f"                  NORMAL  PNEUMONIA")
    print(f"  Actual NORMAL     {cm[0][0]:4d}      {cm[0][1]:4d}")
    print(f"  Actual PNEUMONIA  {cm[1][0]:4d}      {cm[1][1]:4d}")

    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=[0, 1], zero_division=0
    )

    # Overfitting indicator: if NORMAL recall << PNEUMONIA recall, model is biased
    normal_recall = recall[0]
    pneumonia_recall = recall[1]
    recall_gap = abs(normal_recall - pneumonia_recall)

    print(f"\n  --- Per-Class Summary ---")
    print(f"  NORMAL:    precision={precision[0]:.4f}  recall={normal_recall:.4f}  f1={f1[0]:.4f}  support={support[0]}")
    print(f"  PNEUMONIA: precision={precision[1]:.4f}  recall={pneumonia_recall:.4f}  f1={f1[1]:.4f}  support={support[1]}")
    print(f"  Macro F1:  {np.mean(f1):.4f}")
    print(f"  Recall gap (|NORMAL - PNEUMONIA|): {recall_gap:.4f}", end="")
    if recall_gap > 0.15:
        print(f"  ⚠ HIGH — model is biased toward one class")
    else:
        print(f"  ✓ acceptable")

    # Calibration analysis
    print(f"\n  --- Calibration Analysis ---")
    probs_arr = np.array(probabilities)
    preds_arr = np.array(predictions)

    # Bin predictions by confidence and check accuracy in each bin
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for lo, hi in bins:
        mask = (probs_arr >= lo) & (probs_arr < hi)
        if mask.sum() == 0:
            continue
        bin_preds = preds_arr[mask]
        bin_true = np.array(true_labels)[mask]
        # For PNEUMONIA predictions (prob >= 0.5), check if they're correct
        # For NORMAL predictions (prob < 0.5), check if they're correct
        correct = ((bin_preds == 1) & (bin_true == 1)) | ((bin_preds == 0) & (bin_true == 0))
        acc = correct.mean()
        print(f"  Confidence [{lo:.1f}-{hi:.1f}): {mask.sum():4d} images, accuracy={acc:.4f}")

    # Brier score (lower is better calibrated)
    brier = brier_score_loss(true_labels, probabilities)
    print(f"  Brier score: {brier:.4f} (lower = better calibrated, 0 = perfect)")

    # Confidence distribution
    print(f"\n  --- Confidence Distribution ---")
    print(f"  Mean confidence: {probs_arr.mean():.4f}")
    print(f"  Std confidence:  {probs_arr.std():.4f}")
    print(f"  Min confidence:  {probs_arr.min():.4f}")
    print(f"  Max confidence:  {probs_arr.max():.4f}")

    # Flag images with low confidence (potential OOD or hard examples)
    low_conf_mask = probs_arr < 0.55
    if low_conf_mask.sum() > 0:
        print(f"\n  ⚠ {low_conf_mask.sum()} images with confidence < 0.55 (potential issues):")
        for i in np.where(low_conf_mask)[0][:10]:
            label_name = "NORMAL" if true_labels[i] == 0 else "PNEUMONIA"
            pred_name = "NORMAL" if predictions[i] == 0 else "PNEUMONIA"
            print(f"    {filenames[i]:40s}  true={label_name:10s}  pred={pred_name:10s}  conf={probabilities[i]:.4f}")

    # Overall accuracy (note the imbalance!)
    overall_acc = (preds_arr == np.array(true_labels)).mean()
    print(f"\n  --- Overall ---")
    print(f"  Overall accuracy: {overall_acc:.4f} ({(preds_arr == np.array(true_labels)).sum()}/{len(paths)})")
    print(f"  ⚠ NOTE: Test set is imbalanced ({n_pneumonia}/{n_normal} = {n_pneumonia/max(n_normal,1):.2f}:1)")
    print(f"  Macro F1 ({np.mean(f1):.4f}) is more reliable than accuracy for this distribution")

    return {
        "model_label": model_label,
        "model_path": str(model_path),
        "test_size": len(paths),
        "normal_count": n_normal,
        "pneumonia_count": n_pneumonia,
        "accuracy": float(overall_acc),
        "macro_f1": float(np.mean(f1)),
        "normal_precision": float(precision[0]),
        "normal_recall": float(normal_recall),
        "normal_f1": float(f1[0]),
        "pneumonia_precision": float(precision[1]),
        "pneumonia_recall": float(pneumonia_recall),
        "pneumonia_f1": float(f1[1]),
        "recall_gap": float(recall_gap),
        "brier_score": float(brier),
        "confusion_matrix": cm.tolist(),
    }


def test_ood_images(registry):
    """Test inference on non-X-ray images to check behavior."""
    print(f"\n{'=' * 70}")
    print(f"  OOD / NON-X-RAY TESTS")
    print(f"{'=' * 70}")

    # Test 1: solid color image (should NOT look like an X-ray)
    print(f"\n  Test 1: Solid red image (224x224)")
    try:
        from io import BytesIO
        red_img = Image.new("RGB", (224, 224), (200, 50, 50))
        buf = BytesIO()
        red_img.save(buf, format="JPEG")
        result = registry.predict("pneumonia_xray", buf.getvalue(), filename="red.jpg", content_type="image/jpeg")
        print(f"    Result: {result['predicted_class']} ({result['confidence_percentage']:.1f}%)")
        print(f"    ⚠ No OOD rejection — non-X-ray was classified!")
    except Exception as exc:
        print(f"    Error: {exc}")

    # Test 2: colorful photo-like image
    print(f"\n  Test 2: Random color noise image (224x224)")
    try:
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise)
        buf = BytesIO()
        noise_img.save(buf, format="PNG")
        result = registry.predict("pneumonia_xray", buf.getvalue(), filename="noise.png", content_type="image/png")
        print(f"    Result: {result['predicted_class']} ({result['confidence_percentage']:.1f}%)")
        print(f"    ⚠ No OOD rejection — noise image was classified!")
    except Exception as exc:
        print(f"    Error: {exc}")

    # Test 3: testing images
    testing_dir = PROJECT_ROOT / "testing"
    for img_name in ["Chest-Xray-Normal.jpeg", "Chest-Xray-Pneuomia.jpeg"]:
        img_path = testing_dir / img_name
        if img_path.exists():
            print(f"\n  Test 3: {img_name}")
            try:
                result = registry.predict("pneumonia_xray", img_path.read_bytes(), filename=img_name)
                print(f"    Result: {result['predicted_class']} ({result['confidence_percentage']:.1f}%)")
            except Exception as exc:
                print(f"    Error: {exc}")


def main():
    print("=" * 70)
    print("  CHEST X-RAY CNN DIAGNOSTIC REPORT")
    print("=" * 70)

    # Check test data
    print(f"\n  Test directory: {TEST_DIR}")
    if not TEST_DIR.exists():
        print(f"  ERROR: Test directory not found!")
        sys.exit(1)

    # Diagnose all model versions
    model_versions = {
        "v1": REGISTRY_DIR / "cnn_pneumonia.h5",
        "v2": REGISTRY_DIR / "cnn_pneumonia_v2.h5",
        "v3": REGISTRY_DIR / "cnn_pneumonia_v3.h5",
        "v3_final": REGISTRY_DIR / "cnn_pneumonia_v3_final.h5",
    }

    results = {}
    for label, path in model_versions.items():
        results[label] = diagnose_model(path, label)

    # Summary comparison
    print(f"\n{'=' * 70}")
    print(f"  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<12} {'Accuracy':>10} {'Macro F1':>10} {'N-Recall':>10} {'P-Recall':>10} {'Recall Gap':>12} {'Brier':>10}")
    print(f"  {'-' * 74}")
    for label, r in results.items():
        if not r:
            continue
        print(
            f"  {label:<12} "
            f"{r['accuracy']:>10.4f} "
            f"{r['macro_f1']:>10.4f} "
            f"{r['normal_recall']:>10.4f} "
            f"{r['pneumonia_recall']:>10.4f} "
            f"{r['recall_gap']:>12.4f} "
            f"{r['brier_score']:>10.4f}"
        )

    # Find best model
    valid_results = {k: v for k, v in results.items() if v}
    if valid_results:
        best_f1 = max(valid_results.items(), key=lambda x: x[1]["macro_f1"])
        best_balanced = max(valid_results.items(), key=lambda x: x[1]["macro_f1"] - x[1]["recall_gap"])
        print(f"\n  Best by Macro F1:      {best_f1[0]} ({best_f1[1]['macro_f1']:.4f})")
        print(f"  Best by balanced F1:   {best_balanced[0]} (F1={best_balanced[1]['macro_f1']:.4f}, gap={best_balanced[1]['recall_gap']:.4f})")

    # Test OOD behavior with the best available model
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "ai_service"))
        from cnn.config import get_cnn_model_configs
        from cnn.registry import CNNModelRegistry

        os.environ["DISABLE_OOD"] = "True"  # Skip OOD for diagnosis (testing current behavior)
        registry = CNNModelRegistry()
        registry.load_all()
        test_ood_images(registry)
    except Exception as exc:
        print(f"\n  OOD tests skipped: {exc}")

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
