"""DRF serializers for Clinical intelligence models."""

from rest_framework import serializers

from .models import (
    ClinicalObservation,
    DataConflict,
    DiagnosticReportRecord,
    ObservationReferenceRange,
    ObservationTrend,
    PatientStateSnapshot,
)


class ObservationReferenceRangeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObservationReferenceRange
        fields = ("id", "low", "high", "unit", "age_min", "age_max", "sex", "source")
        read_only_fields = ("id",)


class ClinicalObservationSerializer(serializers.ModelSerializer):
    reference_ranges = ObservationReferenceRangeSerializer(many=True, read_only=True)

    class Meta:
        model = ClinicalObservation
        fields = (
            "id", "test_name", "standardised_name", "original_value", "numeric_value",
            "original_unit", "normalised_unit", "reference_range_low", "reference_range_high",
            "reference_range_text", "abnormality_status", "collection_date", "report_source",
            "extraction_confidence", "verification_status", "is_manual_entry", "notes",
            "reference_ranges", "created_at", "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at")

    def validate_report_source(self, value):
        if value and value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Invalid report reference.")
        return value


class ObservationTrendSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObservationTrend
        fields = (
            "id", "test_name", "earlier_observation", "later_observation", "absolute_change",
            "percentage_change", "trend_direction", "is_persistent_abnormality",
            "is_sudden_change", "is_missing_follow_up", "is_conflicting", "calculated_at",
        )
        read_only_fields = ("id", "calculated_at")


class DiagnosticReportRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticReportRecord
        fields = (
            "id", "report_title", "report_type", "report_date", "performing_lab",
            "ordering_clinician", "clinical_notes", "conclusion", "status", "source_report",
            "created_at", "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at")

    def validate_source_report(self, value):
        if value and value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("Invalid report reference.")
        return value


class DataConflictSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataConflict
        fields = (
            "id", "conflict_type", "severity", "first_source_label", "first_source_detail",
            "first_source_record_id", "second_source_label", "second_source_detail",
            "second_source_record_id", "explanation", "detection_method", "resolution_status",
            "resolution_notes", "created_at", "updated_at",
        )
        read_only_fields = ("id", "created_at", "updated_at")


class PatientStateSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientStateSnapshot
        fields = (
            "id", "priority_level", "snapshot_data", "critical_findings", "recent_changes",
            "active_risks", "supporting_evidence", "contradictory_evidence",
            "medication_concerns", "missing_information", "uncertainty_notes",
            "suggested_next_steps", "data_sources", "is_current", "created_at",
        )
        read_only_fields = ("id", "created_at")
