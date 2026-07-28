from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class ClinicalObservation(models.Model):
    """A single extracted or entered laboratory observation / vital sign."""

    VERIFICATION_STATUS = [
        ("unverified", "Unverified"),
        ("verified", "Verified"),
        ("corrected", "Corrected"),
        ("entered_in_error", "Entered in Error"),
    ]

    ABNORMALITY_STATUS = [
        ("normal", "Normal"),
        ("high", "High"),
        ("low", "Low"),
        ("critically_high", "Critically High"),
        ("critically_low", "Critically Low"),
        ("unable_to_assess", "Unable to Assess"),
        ("not_tested", "Not Tested"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="clinical_observations",
    )
    test_name = models.CharField(max_length=255, db_index=True)
    standardised_name = models.CharField(max_length=255, blank=True, db_index=True)
    original_value = models.CharField(max_length=255)
    numeric_value = models.FloatField(null=True, blank=True)
    original_unit = models.CharField(max_length=64, blank=True)
    normalised_unit = models.CharField(max_length=64, blank=True)
    reference_range_low = models.FloatField(null=True, blank=True)
    reference_range_high = models.FloatField(null=True, blank=True)
    reference_range_text = models.CharField(max_length=255, blank=True)
    abnormality_status = models.CharField(
        max_length=32, choices=ABNORMALITY_STATUS, default="unable_to_assess"
    )
    collection_date = models.DateTimeField(null=True, blank=True, db_index=True)
    report_source = models.ForeignKey(
        "reports.MedicalReport",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="observations",
    )
    extraction_confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
    )
    verification_status = models.CharField(
        max_length=32, choices=VERIFICATION_STATUS, default="unverified"
    )
    is_manual_entry = models.BooleanField(default=False)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-collection_date", "-created_at"]
        indexes = [
            models.Index(fields=["user", "standardised_name"]),
            models.Index(fields=["user", "collection_date"]),
        ]

    def __str__(self):
        return f"{self.test_name}: {self.original_value} {self.original_unit}"


class ObservationReferenceRange(models.Model):
    """Reference range for a test, potentially varying by age/sex."""

    observation = models.ForeignKey(
        ClinicalObservation, on_delete=models.CASCADE, related_name="reference_ranges"
    )
    low = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=64, blank=True)
    age_min = models.FloatField(null=True, blank=True)
    age_max = models.FloatField(null=True, blank=True)
    sex = models.CharField(max_length=16, blank=True)
    source = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"{self.low}–{self.high} {self.unit}"


class ObservationTrend(models.Model):
    """Pre-computed trend between two clinical observations."""

    TREND_DIRECTIONS = [
        ("improving", "Improving"),
        ("worsening", "Worsening"),
        ("stable", "Stable"),
        ("fluctuating", "Fluctuating"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="observation_trends",
    )
    test_name = models.CharField(max_length=255, db_index=True)
    earlier_observation = models.ForeignKey(
        ClinicalObservation, on_delete=models.CASCADE, related_name="earlier_trends"
    )
    later_observation = models.ForeignKey(
        ClinicalObservation, on_delete=models.CASCADE, related_name="later_trends"
    )
    absolute_change = models.FloatField(null=True, blank=True)
    percentage_change = models.FloatField(null=True, blank=True)
    trend_direction = models.CharField(
        max_length=16, choices=TREND_DIRECTIONS, default="unknown"
    )
    is_persistent_abnormality = models.BooleanField(default=False)
    is_sudden_change = models.BooleanField(default=False)
    is_missing_follow_up = models.BooleanField(default=False)
    is_conflicting = models.BooleanField(default=False)
    calculated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-calculated_at"]
        indexes = [models.Index(fields=["user", "test_name"])]

    def __str__(self):
        return f"{self.test_name}: {self.trend_direction} ({self.absolute_change})"


class DiagnosticReportRecord(models.Model):
    """Structured record of a diagnostic report (lab report, imaging, etc.)."""

    REPORT_STATUS = [
        ("preliminary", "Preliminary"),
        ("final", "Final"),
        ("amended", "Amended"),
        ("corrected", "Corrected"),
        ("cancelled", "Cancelled"),
        ("unknown", "Unknown"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="diagnostic_reports",
    )
    report_title = models.CharField(max_length=255, blank=True)
    report_type = models.CharField(max_length=64, blank=True)
    report_date = models.DateTimeField(null=True, blank=True, db_index=True)
    performing_lab = models.CharField(max_length=255, blank=True)
    ordering_clinician = models.CharField(max_length=255, blank=True)
    clinical_notes = models.TextField(blank=True)
    conclusion = models.TextField(blank=True)
    status = models.CharField(max_length=32, choices=REPORT_STATUS, default="unknown")
    source_report = models.ForeignKey(
        "reports.MedicalReport",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="diagnostic_records",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-report_date", "-created_at"]

    def __str__(self):
        return (
            f"{self.report_title or 'Report'} "
            f"({self.report_date or 'no date'})"
        )


class DataConflict(models.Model):
    """Recorded contradiction between two data sources in the patient record."""

    CONFLICT_TYPES = [
        ("medication_mismatch", "Medication Mismatch"),
        ("value_discrepancy", "Value Discrepancy"),
        ("report_summary_vs_data", "Report Summary vs Data"),
        ("condition_misreported", "Condition Misreported"),
        ("demographic_mismatch", "Demographic Mismatch"),
        ("temporal_conflict", "Temporal Conflict"),
        ("prediction_vs_input", "Prediction vs Input"),
        ("ai_vs_structured", "AI vs Structured Data"),
        ("timeline_vs_record", "Timeline vs Record"),
        ("duplicate_observation", "Duplicate Observation"),
        ("other", "Other"),
    ]

    SEVERITY_LEVELS = [
        ("info", "Info"),
        ("warning", "Warning"),
        ("critical", "Critical"),
    ]

    RESOLUTION_STATUS = [
        ("unresolved", "Unresolved"),
        ("first_source_confirmed", "First Source Confirmed"),
        ("second_source_confirmed", "Second Source Confirmed"),
        ("both_uncertain", "Both Uncertain"),
        ("correction_added", "Correction Added"),
        ("dismissed", "Dismissed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="data_conflicts",
    )
    conflict_type = models.CharField(max_length=64, choices=CONFLICT_TYPES)
    severity = models.CharField(
        max_length=16, choices=SEVERITY_LEVELS, default="warning"
    )
    first_source_label = models.CharField(max_length=255)
    first_source_detail = models.TextField(blank=True)
    first_source_record_id = models.CharField(max_length=64, blank=True)
    second_source_label = models.CharField(max_length=255)
    second_source_detail = models.TextField(blank=True)
    second_source_record_id = models.CharField(max_length=64, blank=True)
    explanation = models.TextField(blank=True)
    detection_method = models.CharField(max_length=64, default="rule_based")
    resolution_status = models.CharField(
        max_length=32,
        choices=RESOLUTION_STATUS,
        default="unresolved",
        db_index=True,
    )
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="resolved_conflicts",
    )
    resolution_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "resolution_status"]),
            models.Index(fields=["conflict_type"]),
        ]

    def __str__(self):
        return (
            f"{self.get_conflict_type_display()} "
            f"({self.get_severity_display()})"
        )


class PatientStateSnapshot(models.Model):
    """Immutable snapshot of a patient's aggregated state at a point in time."""

    PRIORITY_LEVELS = [
        ("routine", "Routine"),
        ("review_soon", "Review Soon"),
        ("review_today", "Review Today"),
        ("urgent", "Urgent"),
        ("emergency", "Emergency"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="patient_state_snapshots",
    )
    priority_level = models.CharField(
        max_length=32, choices=PRIORITY_LEVELS, default="routine"
    )
    snapshot_data = models.JSONField(default=dict)
    critical_findings = models.JSONField(default=list, blank=True)
    recent_changes = models.JSONField(default=list, blank=True)
    active_risks = models.JSONField(default=list, blank=True)
    supporting_evidence = models.JSONField(default=list, blank=True)
    contradictory_evidence = models.JSONField(default=list, blank=True)
    medication_concerns = models.JSONField(default=list, blank=True)
    missing_information = models.JSONField(default=list, blank=True)
    uncertainty_notes = models.JSONField(default=list, blank=True)
    suggested_next_steps = models.JSONField(default=list, blank=True)
    data_sources = models.JSONField(default=list, blank=True)
    is_current = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["user", "is_current"]),
        ]

    def __str__(self):
        return (
            f"PatientState({self.user_id}) "
            f"@ {self.created_at:%Y-%m-%d %H:%M}"
        )
