"""Admin registration for medication models."""

from django.contrib import admin

from .models import (
    Medication,
    MedicationAllergy,
    MedicationEvent,
    MedicationExtractionRecord,
    MedicationSafetyAlert,
    PatientMedication,
)


@admin.register(Medication)
class MedicationAdmin(admin.ModelAdmin):
    list_display = ["name", "generic_name", "rxnorm_code", "drug_class"]
    search_fields = ["name", "generic_name", "rxnorm_code"]
    list_filter = ["drug_class"]


@admin.register(PatientMedication)
class PatientMedicationAdmin(admin.ModelAdmin):
    list_display = ["user", "medication", "dosage", "frequency", "status", "start_date"]
    list_filter = ["status"]
    search_fields = ["user__username", "medication__name"]


@admin.register(MedicationEvent)
class MedicationEventAdmin(admin.ModelAdmin):
    list_display = ["user", "patient_medication", "event_type", "event_date", "severity"]
    list_filter = ["event_type", "severity"]


@admin.register(MedicationSafetyAlert)
class MedicationSafetyAlertAdmin(admin.ModelAdmin):
    list_display = ["user", "alert_type", "severity", "title", "resolution_status"]
    list_filter = ["alert_type", "severity", "resolution_status"]


@admin.register(MedicationAllergy)
class MedicationAllergyAdmin(admin.ModelAdmin):
    list_display = ["user", "allergen", "severity", "reaction_type"]
    list_filter = ["severity"]
    search_fields = ["allergen", "user__username"]


@admin.register(MedicationExtractionRecord)
class MedicationExtractionRecordAdmin(admin.ModelAdmin):
    list_display = ["user", "extraction_method", "extraction_confidence", "created_at"]
    list_filter = ["extraction_method", "is_synthetic_data"]
