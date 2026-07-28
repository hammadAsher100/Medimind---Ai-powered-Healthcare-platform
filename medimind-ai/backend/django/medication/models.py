from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Medication(models.Model):
    """Standard medication record (from RxNorm or manual entry)."""

    name = models.CharField(max_length=255, db_index=True)
    rxnorm_code = models.CharField(max_length=64, blank=True, db_index=True)
    generic_name = models.CharField(max_length=255, blank=True)
    drug_class = models.CharField(max_length=255, blank=True)
    common_side_effects = models.JSONField(default=list, blank=True)
    serious_side_effects = models.JSONField(default=list, blank=True)
    contraindications = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class PatientMedication(models.Model):
    """A medication currently or previously taken by a patient."""

    STATUS_CHOICES = [
        ("active", "Active"),
        ("discontinued", "Discontinued"),
        ("on_hold", "On Hold"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="patient_medications",
    )
    medication = models.ForeignKey(
        Medication, on_delete=models.CASCADE, related_name="patient_medications"
    )
    dosage = models.CharField(max_length=128, blank=True)
    frequency = models.CharField(max_length=128, blank=True)
    route = models.CharField(max_length=64, blank=True)
    prescribed_for = models.CharField(max_length=255, blank=True)
    prescriber = models.CharField(max_length=255, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    status = models.CharField(
        max_length=16, choices=STATUS_CHOICES, default="active"
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-start_date"]

    def __str__(self):
        return f"{self.medication.name} ({self.status})"


class MedicationEvent(models.Model):
    """A single medication event (dose taken, skipped, side effect, etc.)."""

    EVENT_TYPES = [
        ("dose_taken", "Dose Taken"),
        ("dose_skipped", "Dose Skipped"),
        ("dose_late", "Dose Late"),
        ("side_effect", "Side Effect"),
        ("dose_changed", "Dose Changed"),
        ("medication_stopped", "Medication Stopped"),
        ("medication_started", "Medication Started"),
        ("adverse_reaction", "Adverse Reaction"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="medication_events",
    )
    patient_medication = models.ForeignKey(
        PatientMedication, on_delete=models.CASCADE, related_name="events"
    )
    event_type = models.CharField(max_length=32, choices=EVENT_TYPES)
    event_date = models.DateTimeField(auto_now_add=True, db_index=True)
    notes = models.TextField(blank=True)
    severity = models.CharField(max_length=16, blank=True)

    class Meta:
        ordering = ["-event_date"]

    def __str__(self):
        return f"{self.get_event_type_display()} — {self.patient_medication}"


class MedicationSafetyAlert(models.Model):
    """Alert generated for a patient based on medication analysis."""

    ALERT_TYPES = [
        ("drug_interaction", "Drug Interaction"),
        ("allergy_conflict", "Allergy Conflict"),
        ("duplicate_therapy", "Duplicate Therapy"),
        ("contraindication", "Contraindication"),
        ("overdosage_risk", "Overdosage Risk"),
        ("missing_info", "Missing Information"),
    ]

    SEVERITY_CHOICES = [
        ("info", "Info"),
        ("warning", "Warning"),
        ("critical", "Critical"),
    ]

    RESOLUTION_CHOICES = [
        ("unresolved", "Unresolved"),
        ("acknowledged", "Acknowledged"),
        ("resolved", "Resolved"),
        ("dismissed", "Dismissed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="medication_safety_alerts",
    )
    alert_type = models.CharField(max_length=64, choices=ALERT_TYPES)
    severity = models.CharField(
        max_length=16, choices=SEVERITY_CHOICES, default="warning"
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    medications_involved = models.ManyToManyField(
        PatientMedication, related_name="safety_alerts"
    )
    resolution_status = models.CharField(
        max_length=16,
        choices=RESOLUTION_CHOICES,
        default="unresolved",
        db_index=True,
    )
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="resolved_medication_alerts",
    )
    resolution_notes = models.TextField(blank=True)
    is_synthetic_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.title} ({self.get_severity_display()})"


class MedicationAllergy(models.Model):
    """Recorded allergy linked to a medication or allergen."""

    SEVERITY_CHOICES = [
        ("mild", "Mild"),
        ("moderate", "Moderate"),
        ("severe", "Severe"),
        ("life_threatening", "Life Threatening"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="medication_allergies",
    )
    allergen = models.CharField(max_length=255, db_index=True)
    rxnorm_code = models.CharField(max_length=64, blank=True)
    reaction_type = models.CharField(max_length=255, blank=True)
    severity = models.CharField(
        max_length=24, choices=SEVERITY_CHOICES, default="unknown"
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["allergen"]

    def __str__(self):
        return f"{self.allergen} ({self.severity})"


class MedicationExtractionRecord(models.Model):
    """Tracks how medication data was extracted from an uploaded report."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="medication_extractions",
    )
    source_report = models.ForeignKey(
        "reports.MedicalReport",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="medication_extractions",
    )
    extraction_method = models.CharField(max_length=64, default="llm")
    medications_found = models.JSONField(default=list, blank=True)
    extraction_confidence = models.FloatField(null=True, blank=True)
    raw_text = models.TextField(blank=True)
    is_synthetic_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Extraction {self.pk} ({self.extraction_method}) "
            f"@ {self.created_at:%Y-%m-%d}"
        )
