import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from llm.provider import LLMProvider
from mlflow_utils.tracking import log_prediction
from preprocessing.features import FEATURE_ENGINEERING

router = APIRouter(tags=["prediction"])

DISEASE_FIELDS = {
    "diabetes": ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree_function", "age", "pregnancies"],
    "heart": ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol", "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina", "st_depression", "st_slope", "num_major_vessels", "thal"],
    "kidney": ["age", "blood_pressure", "specific_gravity", "albumin", "sugar", "red_blood_cells", "pus_cell", "pus_cell_clumps", "bacteria", "blood_glucose_random", "blood_urea", "serum_creatinine", "sodium", "potassium", "hemoglobin", "packed_cell_volume", "white_blood_cell_count", "red_blood_cell_count", "hypertension", "diabetes_mellitus", "coronary_artery_disease", "appetite", "pedal_edema", "anemia"],
    "stroke": ["age", "hypertension", "heart_disease", "ever_married", "work_type", "residence_type", "avg_glucose_level", "bmi", "smoking_status", "gender"],
}


def _risk_level(percent: float) -> str:
    if percent >= 70:
        return "High"
    if percent >= 40:
        return "Medium"
    return "Low"


def _heuristic_risk(disease: str, payload: dict[str, Any]) -> float:
    score = 0.15
    if disease == "diabetes":
        score += max(0, float(payload.get("glucose", 90)) - 100) / 220
        score += max(0, float(payload.get("bmi", 24)) - 25) / 80
        score += max(0, float(payload.get("age", 35)) - 45) / 140
    elif disease == "heart":
        score += max(0, float(payload.get("cholesterol", 180)) - 180) / 300
        score += max(0, float(payload.get("resting_bp", 120)) - 120) / 220
        score += max(0, float(payload.get("age", 40)) - 50) / 120
    elif disease == "kidney":
        score += max(0, float(payload.get("serum_creatinine", 1)) - 1.2) / 8
        score += max(0, float(payload.get("blood_urea", 30)) - 40) / 200
        score += 0.12 if str(payload.get("hypertension", "")).lower() in {"yes", "1", "true"} else 0
    elif disease == "stroke":
        score += max(0, float(payload.get("avg_glucose_level", 90)) - 110) / 260
        score += max(0, float(payload.get("age", 45)) - 55) / 100
        score += 0.12 if payload.get("hypertension") else 0
        score += 0.12 if payload.get("heart_disease") else 0
    return max(0.02, min(0.98, score)) * 100


def _prepare_frame(disease: str, payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    missing = [field for field in DISEASE_FIELDS[disease] if field not in payload]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_fields": missing})
    df = pd.DataFrame([payload])
    df = FEATURE_ENGINEERING[disease](df)
    df = pd.get_dummies(df)
    for column in feature_columns:
        if column not in df:
            df[column] = 0
    if feature_columns:
        df = df[feature_columns]
    return df.fillna(0)


def _predict_with_model(bundle: dict, frame: pd.DataFrame) -> tuple[int, float]:
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    if model is None:
        raise RuntimeError("Model artifact is missing.")
    data = scaler.transform(frame) if scaler is not None else frame
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(data)[0][1])
    else:
        probability = float(model.predict(data)[0])
    return int(probability >= 0.5), probability * 100


def _shap_factors(bundle: dict, frame: pd.DataFrame, risk_percentage: float) -> list[dict]:
    explainer = bundle.get("shap_explainer")
    values = None
    try:
        if explainer is not None:
            raw = explainer.shap_values(frame)
            if isinstance(raw, list):
                raw = raw[-1]
            values = np.asarray(raw)[0]
    except Exception:
        values = None
    if values is None:
        numeric = frame.iloc[0].astype(float)
        denom = float(np.abs(numeric).sum()) or 1.0
        values = (numeric / denom * risk_percentage).to_numpy()
    factors = []
    for feature, contribution in sorted(zip(frame.columns, values), key=lambda item: abs(item[1]), reverse=True)[:6]:
        factors.append(
            {
                "feature": feature.replace("_", " ").title(),
                "contribution": round(abs(float(contribution)), 3),
                "direction": "increases_risk" if float(contribution) >= 0 else "decreases_risk",
            }
        )
    return factors


@router.post("/predict/{disease}")
async def predict(disease: str, payload: dict, request: Request):
    try:
        disease = disease.lower()
        if disease not in DISEASE_FIELDS:
            raise HTTPException(status_code=404, detail="Unsupported disease model.")
        bundle = request.app.state.models.get(disease, {})
        feature_columns = bundle.get("feature_columns") or []
        frame = _prepare_frame(disease, payload, feature_columns)
        try:
            prediction, risk_percentage = _predict_with_model(bundle, frame)
        except RuntimeError:
            risk_percentage = _heuristic_risk(disease, payload)
            prediction = int(risk_percentage >= 50)

        top_factors = _shap_factors(bundle, frame, risk_percentage)
        provider = LLMProvider()
        prompt_payload = json.dumps({"disease": disease, "risk_percentage": risk_percentage, "top_factors": top_factors}, indent=2)
        try:
            explanation_text = provider.chat(
                [
                    {"role": "system", "content": "Explain ML disease-risk drivers in simple, cautious patient language. Do not diagnose."},
                    {"role": "user", "content": prompt_payload},
                ]
            )
            ai_recommendation = provider.chat(
                [
                    {"role": "system", "content": "Give educational, non-diagnostic next-step recommendations for a patient to discuss with a clinician."},
                    {"role": "user", "content": prompt_payload},
                ]
            )
        except Exception:
            explanation_text = "The listed factors contributed most to this model estimate. This is educational and not a diagnosis."
            ai_recommendation = "Discuss these results with a qualified healthcare provider, especially if symptoms or abnormal lab values are present."

        result = {
            "disease": disease,
            "risk_percentage": round(risk_percentage, 1),
            "risk_level": _risk_level(risk_percentage),
            "prediction": prediction,
            "shap_explanation": {"top_factors": top_factors, "explanation_text": explanation_text},
            "ai_recommendation": ai_recommendation,
        }
        log_prediction(disease, payload, result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
