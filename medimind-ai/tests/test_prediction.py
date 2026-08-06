import pytest
from fastapi.testclient import TestClient
from ai_service.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def disable_ml(monkeypatch):
    # Force DISABLE_ML=True to avoid needing actual models during basic API tests
    monkeypatch.setenv("DISABLE_ML", "True")

def test_predict_diabetes_valid():
    payload = {
        "glucose": 120,
        "blood_pressure": 80,
        "skin_thickness": 20,
        "insulin": 85,
        "bmi": 25.5,
        "diabetes_pedigree_function": 0.5,
        "age": 45,
        "pregnancies": 2
    }
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "disease" in data
    assert data["disease"] == "diabetes"
    assert "prediction" in data
    assert "risk_percentage" in data
    assert "risk_level" in data
    assert "explanation_available" in data

def test_predict_heart_valid():
    payload = {
        "age": 55,
        "sex": 1,
        "chest_pain_type": 2,
        "resting_bp": 130,
        "cholesterol": 240,
        "fasting_blood_sugar": 0,
        "resting_ecg": 1,
        "max_heart_rate": 150,
        "exercise_angina": 0,
        "st_depression": 1.5,
        "st_slope": 1,
        "num_major_vessels": 0,
        "thal": 2
    }
    response = client.post("/predict/heart", json=payload)
    assert response.status_code == 200
    assert response.json()["disease"] == "heart"

def test_predict_kidney_valid():
    payload = {
        "age": 40, "blood_pressure": 80, "specific_gravity": 1.02, "albumin": 0, "sugar": 0,
        "red_blood_cells": "normal", "pus_cell": "normal", "pus_cell_clumps": "notpresent",
        "bacteria": "notpresent", "blood_glucose_random": 121, "blood_urea": 36, "serum_creatinine": 1.2,
        "sodium": 138, "potassium": 4.4, "hemoglobin": 15.4, "packed_cell_volume": 44,
        "white_blood_cell_count": 7800, "red_blood_cell_count": 5.2, "hypertension": "no",
        "diabetes_mellitus": "no", "coronary_artery_disease": "no", "appetite": "good",
        "pedal_edema": "no", "anemia": "no"
    }
    response = client.post("/predict/kidney", json=payload)
    assert response.status_code == 200

def test_predict_stroke_valid():
    payload = {
        "age": 60, "hypertension": 1, "heart_disease": 0, "ever_married": "Yes",
        "work_type": "Private", "residence_type": "Urban", "avg_glucose_level": 105.2,
        "bmi": 28.5, "smoking_status": "formerly smoked", "gender": "Male"
    }
    response = client.post("/predict/stroke", json=payload)
    assert response.status_code == 200

def test_predict_missing_fields():
    payload = {"age": 45}  # Missing most fields
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "missing_fields" in data["detail"]

def test_predict_unsupported_disease():
    response = client.post("/predict/unknown_disease", json={})
    assert response.status_code == 404

def test_predict_shap_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_SHAP_EXPLANATIONS", "False")
    payload = {
        "glucose": 120, "blood_pressure": 80, "skin_thickness": 20, "insulin": 85,
        "bmi": 25.5, "diabetes_pedigree_function": 0.5, "age": 45, "pregnancies": 2
    }
    response = client.post("/predict/diabetes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["explanation_available"] is False
    assert "shap_explanation" not in data
