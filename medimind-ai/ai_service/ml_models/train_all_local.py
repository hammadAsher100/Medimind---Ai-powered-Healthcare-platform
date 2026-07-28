"""Train all local ML models: heart, diabetes, kidney, stroke.

Generates synthetic training data for missing datasets, then trains models
using the same pipeline as train_common.py. Run from the project root:

    .venv/Scripts/python.exe ai_service/ml_models/train_all_local.py

Requires the .venv (scikit-learn, xgboost, shap, pandas, imbalanced-learn).
"""

import json
import sys
from pathlib import Path

# Add ai_service to path so imports work
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "ai_service"))

import numpy as np
import pandas as pd

from train_common import run_training  # noqa: E402


# ── Synthetic data generators ──────────────────────────────────────────────

N_SAMPLES = 1000
RNG = np.random.default_rng(42)


def _gen_diabetes(n: int = N_SAMPLES) -> pd.DataFrame:
    """Generate synthetic Pima Indians-style diabetes data."""
    rows = []
    for _ in range(n):
        age = round(float(RNG.normal(40, 15)), 1)
        pregnancies = int(max(0, round(RNG.normal(3, 3))))
        glucose = round(float(RNG.normal(120, 35)), 1)
        blood_pressure = round(float(RNG.normal(70, 12)), 1)
        skin_thickness = round(float(max(5, RNG.normal(28, 10))), 1)
        insulin = round(float(max(10, RNG.normal(120, 80))), 1)
        bmi = round(float(max(12, RNG.normal(30, 7))), 1)
        dpf = round(float(max(0.05, RNG.normal(0.5, 0.3))), 3)
        # Outcome roughly correlated with glucose, bmi, age
        risk = (glucose > 140) * 0.4 + (bmi > 30) * 0.3 + (age > 50) * 0.2 + (dpf > 0.6) * 0.1
        outcome = int(RNG.uniform() < 0.15 + 0.35 * min(risk, 1.0))
        rows.append([pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, dpf, age, outcome])
    df = pd.DataFrame(rows, columns=[
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree_function", "age", "Outcome",
    ])
    # Ensure both classes present
    if df["Outcome"].nunique() < 2:
        df.iloc[-1, -1] = 1 - df.iloc[0, -1]
    return df


def _gen_kidney(n: int = N_SAMPLES) -> pd.DataFrame:
    """Generate synthetic chronic kidney disease data."""
    rows = []
    for _ in range(n):
        age = round(float(RNG.normal(50, 15)), 1)
        bp = round(float(RNG.normal(75, 12)), 1)
        sg = round(float(RNG.choice([1.005, 1.010, 1.015, 1.020, 1.025])), 3)
        albumin = float(RNG.choice([0, 0, 1, 2, 3]))
        sugar = float(RNG.choice([0, 0, 0, 1, 2]))
        rbc = float(RNG.choice([0, 1]))
        pc = float(RNG.choice([0, 1]))
        pcc = float(RNG.choice([0, 1]))
        ba = float(RNG.choice([0, 1]))
        bgr = round(float(RNG.normal(130, 50)), 1)
        bu = round(float(RNG.normal(40, 20)), 1)
        sc = round(float(max(0.5, RNG.normal(1.5, 1.0))), 2)
        sodium = round(float(RNG.normal(137, 6)), 1)
        potassium = round(float(RNG.normal(4.5, 0.8)), 2)
        hemoglobin = round(float(RNG.normal(12, 2.5)), 1)
        pcv = round(float(RNG.normal(38, 8)), 1)
        wc = round(float(RNG.normal(8000, 3000)))
        rc = round(float(RNG.normal(5, 1)), 1)
        htn = float(RNG.choice([0, 1]))
        dm = float(RNG.choice([0, 1]))
        cad = float(RNG.choice([0, 1]))
        appet = float(RNG.choice([0, 1]))
        pe = float(RNG.choice([0, 1]))
        ane = float(RNG.choice([0, 1]))
        # CKD risk
        ckd_risk = (sc > 1.8) * 0.4 + (hemoglobin < 11) * 0.3 + (albumin > 0) * 0.3 + (htn > 0) * 0.1
        ckd_prob = 0.1 + 0.4 * min(ckd_risk, 1.0)
        classification = int(RNG.uniform() < ckd_prob)
        rows.append([age, bp, sg, albumin, sugar, rbc, pc, pcc, ba, bgr, bu, sc,
                     sodium, potassium, hemoglobin, pcv, wc, rc, htn, dm, cad,
                     appet, pe, ane, classification])
    df = pd.DataFrame(rows, columns=[
        "age", "blood_pressure", "specific_gravity", "albumin", "sugar",
        "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria",
        "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium",
        "potassium", "hemoglobin", "packed_cell_volume",
        "white_blood_cell_count", "red_blood_cell_count", "hypertension",
        "diabetes_mellitus", "coronary_artery_disease", "appetite",
        "pedal_edema", "anemia", "classification",
    ])
    if df["classification"].nunique() < 2:
        df.iloc[-1, -1] = 1 - df.iloc[0, -1]
    return df


def _gen_stroke(n: int = N_SAMPLES) -> pd.DataFrame:
    """Generate synthetic stroke prediction data."""
    rows = []
    for _ in range(n):
        age = round(float(RNG.normal(55, 18)), 1)
        hypertension = int(RNG.uniform() < (0.1 + 0.01 * max(0, age - 40)))
        heart_disease = int(RNG.uniform() < 0.08 + 0.005 * max(0, age - 50))
        ever_married = RNG.choice(["Yes", "No"])
        work_type = RNG.choice(["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        residence_type = RNG.choice(["Urban", "Rural"])
        avg_glucose_level = round(float(RNG.normal(110, 40)), 1)
        bmi = round(float(max(12, RNG.normal(28, 7))), 1)
        smoking_status = RNG.choice(["never smoked", "formerly smoked", "smokes", "Unknown"])
        gender = RNG.choice(["Male", "Female"])
        # Stroke risk
        s_risk = (age > 65) * 0.4 + (hypertension > 0) * 0.3 + (avg_glucose_level > 140) * 0.2 + (heart_disease > 0) * 0.2
        s_prob = 0.02 + 0.3 * min(s_risk, 1.0)
        stroke = int(RNG.uniform() < s_prob)
        rows.append([gender, age, hypertension, heart_disease, ever_married,
                     work_type, residence_type, avg_glucose_level, bmi,
                     smoking_status, stroke])
    df = pd.DataFrame(rows, columns=[
        "gender", "age", "hypertension", "heart_disease", "ever_married",
        "work_type", "residence_type", "avg_glucose_level", "bmi",
        "smoking_status", "stroke",
    ])
    if df["stroke"].nunique() < 2:
        df.iloc[-1, -1] = 1 - df.iloc[0, -1]
    return df


def _gen_heart_with_headers(input_csv: Path) -> pd.DataFrame:
    """Load raw heart CSV (no headers) and attach column names."""
    columns = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "st_depression", "st_slope",
        "num_major_vessels", "thal", "num",
    ]
    df = pd.read_csv(input_csv, header=None, names=columns)
    return df


def main() -> int:
    data_dir = _PROJECT_ROOT / "data" / "raw" / "tabular"
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Heart ──────────────────────────────────────────────────────
    print("=" * 60)
    print("1/4  Training Heart model...")
    heart_csv = data_dir / "heart_with_headers.csv"
    if not heart_csv.exists():
        df = _gen_heart_with_headers(data_dir / "heart.csv")
        df.to_csv(heart_csv, index=False)
        print(f"    Generated {heart_csv} ({len(df)} rows)")
    result = run_training(str(heart_csv), "heart")
    print(f"    Heart model: {result['best_model']}, RoC-AUC: {result['metrics']['roc_auc']:.3f}")

    # ── 2. Diabetes ───────────────────────────────────────────────────
    print("=" * 60)
    print("2/4  Training Diabetes model...")
    diabetes_csv = data_dir / "diabetes.csv"
    if not diabetes_csv.exists():
        df = _gen_diabetes()
        df.to_csv(diabetes_csv, index=False)
        print(f"    Generated {diabetes_csv} ({len(df)} rows)")
    result = run_training(str(diabetes_csv), "diabetes")
    print(f"    Diabetes model: {result['best_model']}, RoC-AUC: {result['metrics']['roc_auc']:.3f}")

    # ── 3. Kidney ─────────────────────────────────────────────────────
    print("=" * 60)
    print("3/4  Training Kidney model...")
    kidney_csv = data_dir / "kidney.csv"
    if not kidney_csv.exists():
        df = _gen_kidney()
        df.to_csv(kidney_csv, index=False)
        print(f"    Generated {kidney_csv} ({len(df)} rows)")
    result = run_training(str(kidney_csv), "kidney")
    print(f"    Kidney model: {result['best_model']}, RoC-AUC: {result['metrics']['roc_auc']:.3f}")

    # ── 4. Stroke ─────────────────────────────────────────────────────
    print("=" * 60)
    print("4/4  Training Stroke model...")
    stroke_csv = data_dir / "stroke.csv"
    if not stroke_csv.exists():
        df = _gen_stroke()
        df.to_csv(stroke_csv, index=False)
        print(f"    Generated {stroke_csv} ({len(df)} rows)")
    result = run_training(str(stroke_csv), "stroke")
    print(f"    Stroke model: {result['best_model']}, RoC-AUC: {result['metrics']['roc_auc']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("All models trained successfully!")
    for disease in ("diabetes", "heart", "kidney", "stroke"):
        model_dir = _HERE / disease
        artifacts = list(model_dir.glob("*"))
        print(f"  {disease}: {len(artifacts)} artifacts in {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
