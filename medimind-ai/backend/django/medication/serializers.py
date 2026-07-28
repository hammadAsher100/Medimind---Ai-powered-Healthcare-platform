"""DRF serializers for Medication models."""

from rest_framework import serializers

from .models import (
    Medication,
    MedicationAllergy,
    MedicationEvent,
    MedicationExtractionRecord,
    MedicationSafetyAlert,
    PatientMedication,
)


class MedicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medication
        fields = "__all__"
        read_only_fields = ["id", "created_at"]


class PatientMedicationSerializer(serializers.ModelSerializer):
    medication_name = serializers.CharField(
        source="medication.name", read_only=True
    )

    class Meta:
        model = PatientMedication
        fields = "__all__"
        read_only_fields = ["id", "created_at", "updated_at"]


class MedicationEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationEvent
        fields = "__all__"
        read_only_fields = ["id", "event_date"]


class MedicationSafetyAlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationSafetyAlert
        fields = "__all__"
        read_only_fields = ["id", "created_at"]


class MedicationAllergySerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationAllergy
        fields = "__all__"
        read_only_fields = ["id", "created_at"]


class MedicationExtractionRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationExtractionRecord
        fields = "__all__"
        read_only_fields = ["id", "created_at"]
