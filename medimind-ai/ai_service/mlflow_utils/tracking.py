import os

import mlflow


def configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def log_prediction(disease: str, payload: dict, result: dict) -> None:
    try:
        configure_mlflow()
        mlflow.set_experiment("production_predictions")
        with mlflow.start_run(run_name=f"{disease}_prediction"):
            mlflow.log_params({f"input_{key}": value for key, value in payload.items() if isinstance(value, (int, float, str, bool))})
            mlflow.log_metric("risk_percentage", float(result.get("risk_percentage", 0)))
            mlflow.log_metric("prediction", int(result.get("prediction", 0)))
    except Exception:
        return
