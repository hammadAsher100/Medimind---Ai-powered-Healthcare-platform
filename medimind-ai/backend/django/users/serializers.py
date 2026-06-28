from rest_framework import serializers

from .models import Allergy, FamilyHistory, MedicalProfile


class MedicalProfileSerializer(serializers.ModelSerializer):
    bmi = serializers.FloatField(read_only=True)

    class Meta:
        model = MedicalProfile
        fields = (
            "id",
            "blood_type",
            "height_cm",
            "weight_kg",
            "smoking_status",
            "alcohol_use",
            "exercise_frequency",
            "bmi",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at", "bmi")


class AllergySerializer(serializers.ModelSerializer):
    class Meta:
        model = Allergy
        fields = ("id", "allergen", "severity", "notes")
        read_only_fields = ("id",)


class FamilyHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = FamilyHistory
        fields = ("id", "condition", "relation", "notes")
        read_only_fields = ("id",)
