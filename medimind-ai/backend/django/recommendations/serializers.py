from rest_framework import serializers

from .models import KnowledgeDocument, Prediction, Recommendation


PREDICTION_FIELDS = {
    "diabetes": (
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "pregnancies",
    ),
    "heart": (
        "age",
        "sex",
        "chest_pain_type",
        "resting_bp",
        "cholesterol",
        "fasting_blood_sugar",
        "resting_ecg",
        "max_heart_rate",
        "exercise_angina",
        "st_depression",
        "st_slope",
        "num_major_vessels",
        "thal",
    ),
    "kidney": (
        "age",
        "blood_pressure",
        "specific_gravity",
        "albumin",
        "sugar",
        "red_blood_cells",
        "pus_cell",
        "pus_cell_clumps",
        "bacteria",
        "blood_glucose_random",
        "blood_urea",
        "serum_creatinine",
        "sodium",
        "potassium",
        "hemoglobin",
        "packed_cell_volume",
        "white_blood_cell_count",
        "red_blood_cell_count",
        "hypertension",
        "diabetes_mellitus",
        "coronary_artery_disease",
        "appetite",
        "pedal_edema",
        "anemia",
    ),
    "stroke": (
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "gender",
    ),
}


class PredictionRequestSerializer(serializers.Serializer):
    """Validate the disease form contract before forwarding it to FastAPI."""

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError("Prediction data must be a JSON object.")
        return dict(data)

    def validate(self, attrs):
        disease = self.context["disease"]
        required_fields = PREDICTION_FIELDS.get(disease)
        if required_fields is None:
            raise serializers.ValidationError({"disease": "Unsupported disease model."})

        missing_fields = [
            field for field in required_fields if field not in attrs or attrs[field] in (None, "")
        ]
        if missing_fields:
            raise serializers.ValidationError({"missing_fields": missing_fields})
        return attrs


class PredictionServiceResponseSerializer(serializers.Serializer):
    """Validate FastAPI's required response fields while accepting optional SHAP data."""

    disease = serializers.CharField()
    risk_percentage = serializers.FloatField()
    risk_level = serializers.CharField()
    prediction = serializers.IntegerField()
    ai_recommendation = serializers.CharField(required=False, allow_blank=True, default="")
    explanation_available = serializers.BooleanField(required=False, default=False)
    shap_explanation = serializers.JSONField(required=False, allow_null=True, default=dict)

    def validate_shap_explanation(self, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise serializers.ValidationError("Expected an object when SHAP data is provided.")
        return value


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = (
            "id",
            "disease",
            "input_data",
            "risk_percentage",
            "risk_level",
            "prediction",
            "shap_explanation",
            "ai_recommendation",
            "created_at",
        )
        read_only_fields = ("id", "created_at")


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = ("id", "category", "content", "source", "metadata", "created_at")
        read_only_fields = ("id", "created_at")


class KnowledgeDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeDocument
        fields = ("id", "file", "title", "source", "indexed_chunks", "metadata", "created_at")
        read_only_fields = ("id", "indexed_chunks", "metadata", "created_at")
