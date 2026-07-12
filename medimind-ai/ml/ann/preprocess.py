"""
Preprocessing pipeline for the UCI Heart Disease dataset.

Handles missing value imputation, feature scaling (StandardScaler),
one-hot encoding of categorical features, and train/test splitting.
Saves fitted scaler and encoder artifacts to ml/registry/ for
inference-time reuse.

Dataset columns (Cleveland subset):
  age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
  oldpeak, slope, ca, thal, target (0=no disease, 1=disease)
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Column names for the UCI Heart Disease Cleveland dataset
COLUMN_NAMES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

NUMERIC_COLS: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS: list[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

RAW_DATA_PATH = os.getenv("ANN_RAW_DATA", "data/raw/tabular/heart.csv")
REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_PATH", "ml/registry"))


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the UCI Heart Disease CSV.

    Handles both header-present and headerless CSVs.  Replaces '?' with NaN
    and coerces columns to numeric where possible.
    """
    try:
        df = pd.read_csv(path, na_values=["?"])
    except Exception:
        # Headerless Cleveland .data file
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values=["?"])

    # Ensure correct column names
    if list(df.columns[:3]) != COLUMN_NAMES[:3]:
        df.columns = COLUMN_NAMES

    # Coerce numeric
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and binarise the multi-class target.

    The original target has values 0-4; we collapse to 0 (no disease)
    vs 1 (disease present) for binary classification.
    """
    # Impute numeric with median
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0)

    # Binarise target: 0 stays 0, 1-4 become 1
    df["target"] = (df["target"] > 0).astype(int)

    return df


def preprocess_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Scale numerics, one-hot encode categoricals, split train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    df = clean_data(df.copy())

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Separate features / target
    y = df_encoded["target"].values
    X = df_encoded.drop(columns=["target"])
    feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32), y.astype(np.float32), test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric columns (by index — they are the first len(NUMERIC_COLS) cols)
    scaler = StandardScaler()
    num_idx = [feature_names.index(c) for c in NUMERIC_COLS if c in feature_names]
    X_train[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_test[:, num_idx] = scaler.transform(X_test[:, num_idx])

    if save_artifacts:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_DIR / "ann_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(REGISTRY_DIR / "ann_feature_names.pkl", "wb") as f:
            pickle.dump(feature_names, f)
        with open(REGISTRY_DIR / "ann_num_indices.pkl", "wb") as f:
            pickle.dump(num_idx, f)

    return X_train, X_test, y_train, y_test, feature_names


def preprocess_single_input(raw_features: dict) -> np.ndarray:
    """Transform a single patient's raw feature dict into a model-ready array.

    Used at inference time.  Loads saved scaler + feature names from registry.
    """
    with open(REGISTRY_DIR / "ann_feature_names.pkl", "rb") as f:
        feature_names: list[str] = pickle.load(f)
    with open(REGISTRY_DIR / "ann_scaler.pkl", "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    with open(REGISTRY_DIR / "ann_num_indices.pkl", "rb") as f:
        num_idx: list[int] = pickle.load(f)

    # Build a single-row DataFrame matching training schema
    row = {}
    for col in COLUMN_NAMES[:-1]:  # exclude 'target'
        if col in CATEGORICAL_COLS:
            val = raw_features.get(col, 0)
            # Create one-hot columns matching training
            for fn in feature_names:
                if fn.startswith(f"{col}_"):
                    suffix = fn.split(f"{col}_", 1)[1]
                    row[fn] = 1.0 if str(val) == suffix or float(val) == float(suffix) else 0.0
        else:
            row[col] = float(raw_features.get(col, 0))

    # Ensure all feature columns present
    arr = np.array([[row.get(fn, 0.0) for fn in feature_names]], dtype=np.float32)

    # Scale numeric columns
    arr[:, num_idx] = scaler.transform(arr[:, num_idx])

    return arr
