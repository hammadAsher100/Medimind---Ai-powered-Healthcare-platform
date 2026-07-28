"""Admin registration for clinical models."""

from django.contrib import admin

from .models import (
    ClinicalObservation,
    DataConflict,
    DiagnosticReportRecord,
    ObservationReferenceRange,
    ObservationTrend,
    PatientStateSnapshot,
)


@admin.register(ClinicalObservation)
class ClinicalObservationAdmin(admin.ModelAdmin):
    list_display = [
        "test_name", "numeric_value", "abnormality_status",
        "collection_date", "user",
    ]
    list_filter = ["abnormality_status", "verification_status", "is_manual_entry"]
    search_fields = ["test_name", "standardised_name", "user__username"]


@admin.register(ObservationTrend)
class ObservationTrendAdmin(admin.ModelAdmin):
    list_display = [
        "test_name", "trend_direction", "absolute_change",
        "calculated_at", "user",
    ]
    list_filter = [
        "trend_direction", "is_sudden_change",
        "is_persistent_abnormality",
    ]


@admin.register(DiagnosticReportRecord)
class DiagnosticReportRecordAdmin(admin.ModelAdmin):
    list_display = ["report_title", "report_type", "report_date", "status", "user"]
    list_filter = ["report_type", "status"]


@admin.register(DataConflict)
class DataConflictAdmin(admin.ModelAdmin):
    list_display = ["conflict_type", "severity", "resolution_status", "user"]
    list_filter = ["conflict_type", "severity", "resolution_status"]


@admin.register(PatientStateSnapshot)
class PatientStateSnapshotAdmin(admin.ModelAdmin):
    list_display = ["user", "priority_level", "is_current", "created_at"]
    list_filter = ["priority_level", "is_current"]


@admin.register(ObservationReferenceRange)
class ObservationReferenceRangeAdmin(admin.ModelAdmin):
    list_display = ["observation", "low", "high", "unit", "sex"]
