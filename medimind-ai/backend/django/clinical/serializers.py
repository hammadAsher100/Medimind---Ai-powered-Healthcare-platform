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
        fields = "__all__"
        read_only_fields = ["id"]


class ClinicalObservationSerializer(serializers.ModelSerializer):
    reference_ranges = ObservationReferenceRangeSerializer(many=True, read_only=True)

    class Meta:
        model = ClinicalObservation
        fields = "__all__"
        read_only_fields = ["id", "created_at", "updated_at"]


class ObservationTrendSerializer(serializers.ModelSerializer):
    class Meta:
        model = ObservationTrend
        fields = "__all__"
        read_only_fields = ["id", "calculated_at"]


class DiagnosticReportRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticReportRecord
        fields = "__all__"
        read_only_fields = ["id", "created_at", "updated_at"]


class DataConflictSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataConflict
        fields = "__all__"
        read_only_fields = ["id", "created_at", "updated_at"]


class PatientStateSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientStateSnapshot
        fields = "__all__"
        read_only_fields = ["id", "created_at"]
