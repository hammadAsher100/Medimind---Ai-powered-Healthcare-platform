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
        fields = ("id", "name", "rxnorm_code", "generic_name", "drug_class", "common_side_effects", "serious_side_effects", "contraindications", "created_at")
        read_only_fields = fields


class PatientMedicationSerializer(serializers.ModelSerializer):
    medication_name = serializers.CharField(
        source="medication.name", read_only=True
    )

    class Meta:
        model = PatientMedication
        fields = ("id", "medication", "medication_name", "dosage", "frequency", "route", "prescribed_for", "prescriber", "start_date", "end_date", "status", "notes", "created_at", "updated_at")
        read_only_fields = ("id", "created_at", "updated_at")


class MedicationEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationEvent
        fields = ("id", "patient_medication", "event_type", "event_date", "notes", "severity")
        read_only_fields = ("id", "event_date")

    def validate_patient_medication(self, value):
        if value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Invalid medication reference.")
        return value


class MedicationSafetyAlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationSafetyAlert
        fields = ("id", "alert_type", "severity", "title", "description", "medications_involved", "resolution_status", "resolution_notes", "is_synthetic_data", "created_at")
        read_only_fields = ("id", "created_at", "is_synthetic_data")

    def validate_medications_involved(self, values):
        if any(value.user_id != self.context["request"].user.id for value in values):
            raise serializers.ValidationError("Invalid medication reference.")
        return values


class MedicationAllergySerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationAllergy
        fields = ("id", "allergen", "rxnorm_code", "reaction_type", "severity", "notes", "created_at")
        read_only_fields = ("id", "created_at")


class MedicationExtractionRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicationExtractionRecord
        fields = ("id", "source_report", "extraction_method", "medications_found", "extraction_confidence", "is_synthetic_data", "created_at")
        read_only_fields = ("id", "created_at", "is_synthetic_data")

    def validate_source_report(self, value):
        if value and value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Invalid report reference.")
        return value
