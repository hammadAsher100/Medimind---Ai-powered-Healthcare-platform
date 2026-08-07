from unittest.mock import Mock, patch

import requests
from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .models import Prediction


PREDICTION_PAYLOADS = {
    "diabetes": {
        "glucose": 120,
        "blood_pressure": 80,
        "skin_thickness": 20,
        "insulin": 85,
        "bmi": 25.5,
        "diabetes_pedigree_function": 0.5,
        "age": 45,
        "pregnancies": 2,
    },
    "heart": {
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
        "thal": 2,
    },
    "kidney": {
        "age": 40,
        "blood_pressure": 80,
        "specific_gravity": 1.02,
        "albumin": 0,
        "sugar": 0,
        "red_blood_cells": "normal",
        "pus_cell": "normal",
        "pus_cell_clumps": "notpresent",
        "bacteria": "notpresent",
        "blood_glucose_random": 121,
        "blood_urea": 36,
        "serum_creatinine": 1.2,
        "sodium": 138,
        "potassium": 4.4,
        "hemoglobin": 15.4,
        "packed_cell_volume": 44,
        "white_blood_cell_count": 7800,
        "red_blood_cell_count": 5.2,
        "hypertension": "no",
        "diabetes_mellitus": "no",
        "coronary_artery_disease": "no",
        "appetite": "good",
        "pedal_edema": "no",
        "anemia": "no",
    },
    "stroke": {
        "age": 60,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "residence_type": "Urban",
        "avg_glucose_level": 105.2,
        "bmi": 28.5,
        "smoking_status": "formerly smoked",
        "gender": "Male",
    },
}


def fastapi_response(disease="diabetes", **overrides):
    result = {
        "disease": disease,
        "risk_percentage": 28.5,
        "risk_level": "Low",
        "prediction": 0,
        "ai_recommendation": "Discuss these results with a qualified healthcare provider.",
        "explanation_available": False,
    }
    result.update(overrides)
    response = Mock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = result
    return response


class PredictionApiTests(APITestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="prediction-user",
            password="test-password-123",
        )
        self.client.force_authenticate(self.user)

    def predict(self, disease, payload=None):
        return self.client.post(
            reverse("prediction-create", args=[disease]),
            payload if payload is not None else PREDICTION_PAYLOADS[disease],
            format="json",
        )

    @patch("recommendations.views.requests.post")
    def test_all_tabular_prediction_endpoints_succeed_without_shap(self, post):
        for disease in PREDICTION_PAYLOADS:
            with self.subTest(disease=disease):
                post.return_value = fastapi_response(disease)
                response = self.predict(disease)

                self.assertEqual(response.status_code, status.HTTP_200_OK)
                self.assertEqual(response["Content-Type"], "application/json")
                self.assertEqual(response.data["disease"], disease)
                self.assertEqual(response.data["risk_percentage"], 28.5)
                self.assertEqual(response.data["risk_level"], "Low")
                self.assertEqual(response.data["prediction"], 0)
                self.assertEqual(
                    response.data["ai_recommendation"],
                    "Discuss these results with a qualified healthcare provider.",
                )
                self.assertFalse(response.data["explanation_available"])
                self.assertEqual(response.data["shap_explanation"], {})

        self.assertEqual(Prediction.objects.count(), 4)

    @patch("recommendations.views.requests.post")
    def test_missing_shap_explanation_does_not_raise_key_error(self, post):
        post.return_value = fastapi_response()

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data["explanation_available"])
        self.assertEqual(response.data["shap_explanation"], {})

    @patch("recommendations.views.requests.post")
    def test_response_with_shap_explanation_is_preserved(self, post):
        shap_explanation = {
            "top_factors": [
                {
                    "feature": "Glucose",
                    "contribution": 12.5,
                    "direction": "increases_risk",
                }
            ],
            "explanation_text": "Glucose was the strongest factor in this estimate.",
        }
        post.return_value = fastapi_response(
            explanation_available=True,
            shap_explanation=shap_explanation,
        )

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data["explanation_available"])
        self.assertEqual(response.data["shap_explanation"], shap_explanation)

    @patch("recommendations.views.requests.post", side_effect=requests.Timeout("timed out"))
    def test_fastapi_timeout_returns_json(self, post):
        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_504_GATEWAY_TIMEOUT)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(response.data["error"], "Risk analysis timed out. Please try again.")

    @patch("recommendations.views.requests.post")
    def test_fastapi_4xx_returns_json(self, post):
        post.return_value = fastapi_response()
        post.return_value.status_code = 422

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])

    @patch("recommendations.views.requests.post")
    def test_fastapi_5xx_returns_json(self, post):
        post.return_value = fastapi_response()
        post.return_value.status_code = 500

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])

    @patch("recommendations.views.requests.post")
    def test_invalid_form_data_is_rejected_before_fastapi(self, post):
        response = self.predict("diabetes", {"age": 45})

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertIn("missing_fields", response.data["details"])
        post.assert_not_called()

    @patch("recommendations.views.requests.post")
    def test_non_json_upstream_response_returns_json_error(self, post):
        post.return_value = fastapi_response()
        post.return_value.headers = {"Content-Type": "text/html; charset=utf-8"}
        post.return_value.text = "<html>Django traceback</html>"

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertNotContains(response, "traceback", status_code=502)
        post.return_value.json.assert_not_called()

    @patch("recommendations.views.requests.post")
    def test_missing_required_fastapi_field_returns_json_error(self, post):
        upstream = fastapi_response()
        del upstream.json.return_value["risk_level"]
        post.return_value = upstream

        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_502_BAD_GATEWAY)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(Prediction.objects.count(), 0)

    @patch("recommendations.views.requests.post", side_effect=RuntimeError("unexpected failure"))
    def test_unexpected_exception_never_returns_debug_html(self, post):
        response = self.predict("diabetes")

        self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(
            response.data,
            {"success": False, "error": "Unable to complete risk analysis."},
        )
        self.assertNotContains(response, "unexpected failure", status_code=500)
