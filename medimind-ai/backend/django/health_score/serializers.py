from rest_framework import serializers

from .models import HealthScore


class HealthScoreSerializer(serializers.ModelSerializer):
    class Meta:
        model = HealthScore
        fields = (
            "id",
            "score",
            "bmi",
            "sugar_level",
            "cholesterol",
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "lifestyle_score",
            "strengths",
            "improvements",
            "created_at",
        )
        read_only_fields = ("id", "created_at")
