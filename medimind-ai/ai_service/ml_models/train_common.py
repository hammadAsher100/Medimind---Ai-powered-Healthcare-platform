import argparse
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from preprocessing.features import FEATURE_ENGINEERING

TARGET_CANDIDATES = {
    "diabetes": ["Outcome", "outcome", "diabetes"],
    "heart": ["target", "condition", "num"],
    "kidney": ["classification", "class", "ckd"],
    "stroke": ["stroke"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the disease CSV dataset.")
    parser.add_argument("--disease", required=True, choices=["diabetes", "heart", "kidney", "stroke"])
    return parser.parse_args()


def find_target(df: pd.DataFrame, disease: str) -> str:
    for candidate in TARGET_CANDIDATES[disease]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Could not infer target column for {disease}. Expected one of {TARGET_CANDIDATES[disease]}.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [column.strip().lower().replace(" ", "_").replace("-", "_") for column in df.columns]
    return df


def save_eda(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"rows": [df.shape[0]], "columns": [df.shape[1]]}).to_csv(output_dir / "shape.csv", index=False)
    df.isna().sum().to_csv(output_dir / "nulls.csv")
    df[target].value_counts(dropna=False).to_csv(output_dir / "class_distribution.csv")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png")
        plt.close()


def preprocess(df: pd.DataFrame, disease: str, target: str):
    df = normalize_columns(df)
    target = target.strip().lower().replace(" ", "_").replace("-", "_")
    y = df[target]
    x = df.drop(columns=[target])
    x = FEATURE_ENGINEERING[disease](x)

    numeric_columns = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    if numeric_columns:
        x[numeric_columns] = SimpleImputer(strategy="median").fit_transform(x[numeric_columns])
    for column in categorical_columns:
        x[[column]] = SimpleImputer(strategy="most_frequent").fit_transform(x[[column]])

    for column in categorical_columns:
        x[column] = x[column].astype(str).str.strip().str.lower()

    x = pd.get_dummies(x, columns=categorical_columns, drop_first=False)
    feature_columns = x.columns.tolist()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    if y.dtype == object:
        y = LabelEncoder().fit_transform(y.astype(str).str.strip().str.lower())
    y = np.asarray(y).astype(int)
    if disease == "heart":
        y = (y > 0).astype(int)
    return x, x_scaled, y, scaler, feature_columns


def train_models(x_scaled, y):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=300, max_depth=4, learning_rate=0.05),
    }
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}
    for name, model in models.items():
        scores = cross_validate(model, x_scaled, y, cv=splitter, scoring=scoring, n_jobs=-1)
        results[name] = {metric.replace("test_", ""): float(np.mean(values)) for metric, values in scores.items() if metric.startswith("test_")}
    best_name = max(results, key=lambda key: results[key]["roc_auc"])
    return best_name, models[best_name], results


def make_shap_explainer(model, sample):
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, sample)
    return shap.Explainer(model, sample)


def run_training(data_path: str, disease: str) -> dict:
    output_dir = Path(__file__).resolve().parent / disease
    df = pd.read_csv(data_path)
    target = find_target(df, disease)
    save_eda(df, target, output_dir / "eda")
    x_frame, x_scaled, y, scaler, feature_columns = preprocess(df, disease, target)

    if len(np.unique(y)) < 2:
        raise ValueError("Target must contain at least two classes.")
    min_class_count = min(np.bincount(y))
    if min_class_count > 1:
        k_neighbors = max(1, min(5, min_class_count - 1))
        x_scaled, y = SMOTE(random_state=42, k_neighbors=k_neighbors).fit_resample(x_scaled, y)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, stratify=y, random_state=42)
    best_name, best_model, cv_results = train_models(x_train, y_train)
    best_model.fit(x_train, y_train)
    probabilities = best_model.predict_proba(x_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.predict(x_test)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    sample = x_train[: min(100, len(x_train))]
    explainer = make_shap_explainer(best_model, sample)
    shap_values = explainer(sample)
    shap_abs = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(shap_abs)[-20:]
    plt.figure(figsize=(9, 7))
    plt.barh([feature_columns[i] for i in top_indices], shap_abs[top_indices])
    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png")
    plt.close()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("medimind_training")
    with mlflow.start_run(run_name=f"{disease}_training"):
        mlflow.log_params({"disease": disease, "best_model": best_name, "features": len(feature_columns)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path=f"{disease}_model")

    joblib.dump(best_model, output_dir / "model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(explainer, output_dir / "shap_explainer.joblib")
    with (output_dir / "feature_columns.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_model": best_name, "cv_results": cv_results, "test_metrics": metrics}, handle, indent=2)
    return {"best_model": best_name, "metrics": metrics, "feature_count": len(feature_columns)}


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run_training(args.data, args.disease), indent=2))
