"""
Download public training datasets and train all four disease models.

Usage:
    python train_all_models.py              # Download datasets + train all
    python train_all_models.py --download   # Only download datasets
    python train_all_models.py --train      # Only train (datasets must exist)

Datasets are saved to datasets/ and trained artifacts go to
ai_service/ml_models/<disease>/.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets"
AI_SERVICE_DIR = ROOT / "ai_service"

DATASET_URLS = {
    "diabetes": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "kidney": "https://raw.githubusercontent.com/manishkr21/Chronic-Kidney-Disease-prediction/master/kidney_disease.csv",
    "stroke": "https://raw.githubusercontent.com/fedesoriano/stroke-prediction-dataset/main/healthcare-dataset-stroke-data.csv",
}

DIABETES_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age", "Outcome",
]

HEART_COLUMNS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "st_depression", "st_slope",
    "num_major_vessels", "thal", "target",
]


def download_datasets() -> None:
    """Download public CSV datasets into datasets/ directory."""
    import urllib.request

    DATASET_DIR.mkdir(exist_ok=True)
    for disease, url in DATASET_URLS.items():
        dest = DATASET_DIR / f"{disease}.csv"
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            continue
        print(f"  [download] {disease} → {dest.name}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"  [error] Failed to download {disease}: {exc}")
            continue

    # The Pima diabetes dataset has no header, add column names
    diabetes_path = DATASET_DIR / "diabetes.csv"
    if diabetes_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(diabetes_path)
            if "Outcome" not in df.columns and df.shape[1] == 9:
                df = pd.read_csv(diabetes_path, header=None, names=DIABETES_COLUMNS)
                df.to_csv(diabetes_path, index=False)
                print("  [fix] Added column headers to diabetes.csv")
        except Exception:
            pass

    # The Cleveland heart dataset has no header, add column names
    heart_path = DATASET_DIR / "heart.csv"
    if heart_path.exists():
        import pandas as pd
        try:
            df = pd.read_csv(heart_path)
            if "target" not in df.columns and df.shape[1] == 14:
                df = pd.read_csv(heart_path, header=None, names=HEART_COLUMNS)
                df = df.replace("?", float("nan"))
                df.to_csv(heart_path, index=False)
                print("  [fix] Added column headers to heart.csv")
        except Exception:
            pass

    print("  Done.\n")


def train_disease(disease: str) -> dict:
    """Train a single disease model using the common training pipeline."""
    sys.path.insert(0, str(AI_SERVICE_DIR))
    os.environ.setdefault("MLFLOW_TRACKING_URI", "")
    os.environ.setdefault("DISABLE_MLFLOW", "true")

    from ml_models.train_common import run_training

    data_path = DATASET_DIR / f"{disease}.csv"
    if not data_path.exists():
        print(f"  [skip] {data_path} not found — run with --download first")
        return {}
    print(f"  Training {disease}...")
    try:
        result = run_training(str(data_path), disease)
        print(f"  ✓ {disease}: best_model={result['best_model']}, "
              f"f1={result['metrics']['f1']:.3f}, "
              f"roc_auc={result['metrics']['roc_auc']:.3f}")
        return result
    except Exception as exc:
        print(f"  ✗ {disease}: {exc}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets and train MediMind ML models")
    parser.add_argument("--download", action="store_true", help="Only download datasets")
    parser.add_argument("--train", action="store_true", help="Only train models (datasets must exist)")
    args = parser.parse_args()

    do_both = not args.download and not args.train

    if args.download or do_both:
        print("Downloading datasets...")
        download_datasets()

    if args.train or do_both:
        print("Training models...")
        diseases = ["diabetes", "heart", "kidney", "stroke"]
        results = {}
        for disease in diseases:
            results[disease] = train_disease(disease)

        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        for disease, result in results.items():
            if result:
                print(f"  {disease:<12} ✓ {result['best_model']}")
            else:
                print(f"  {disease:<12} ✗ Failed or skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
