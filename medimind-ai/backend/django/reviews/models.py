from django.conf import settings
from django.db import models


class ReviewDecision(models.Model):
    """Clinician review decision on an AI recommendation or finding."""

    DECISION_CHOICES = [
        ("accepted", "Accepted"),
        ("modified", "Modified"),
        ("rejected", "Rejected"),
        ("deferred", "Deferred"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="review_decisions",
    )
    recommendation_type = models.CharField(max_length=64, db_index=True)
    recommendation_id = models.CharField(max_length=64, blank=True, db_index=True)
    ai_summary = models.TextField(blank=True)
    clinician_decision = models.CharField(
        max_length=16, choices=DECISION_CHOICES
    )
    clinician_notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Review({self.recommendation_type}) "
            f"— {self.get_clinician_decision_display()}"
        )


class ModelFeedback(models.Model):
    """Structured feedback on an AI model's output for fine-tuning / evaluation."""

    FEEDBACK_TYPES = [
        ("correct", "Correct"),
        ("partially_correct", "Partially Correct"),
        ("incorrect", "Incorrect"),
        ("missing_information", "Missing Information"),
        ("hallucination", "Hallucination"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="model_feedback",
    )
    model_name = models.CharField(max_length=128, db_index=True)
    prediction_id = models.CharField(max_length=64, blank=True, db_index=True)
    feedback_type = models.CharField(max_length=32, choices=FEEDBACK_TYPES)
    original_output = models.TextField(blank=True)
    corrected_output = models.TextField(blank=True)
    notes = models.TextField(blank=True)
    is_synthetic_data = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Feedback({self.model_name}) — "
            f"{self.get_feedback_type_display()}"
        )


class AuditEvent(models.Model):
    """Immutable audit log of significant system actions."""

    EVENT_TYPES = [
        ("prediction_made", "Prediction Made"),
        ("report_uploaded", "Report Uploaded"),
        ("report_analyzed", "Report Analyzed"),
        ("recommendation_generated", "Recommendation Generated"),
        ("review_decision", "Review Decision"),
        ("conflict_detected", "Conflict Detected"),
        ("conflict_resolved", "Conflict Resolved"),
        ("export_generated", "Export Generated"),
        ("feedback_received", "Feedback Received"),
        ("emergency_detected", "Emergency Detected"),
        ("safety_alert", "Safety Alert"),
        ("medication_change", "Medication Change"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="audit_events",
    )
    event_type = models.CharField(max_length=64, choices=EVENT_TYPES, db_index=True)
    event_detail = models.JSONField(default=dict, blank=True)
    source = models.CharField(max_length=64, default="system")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "audit events"

    def __str__(self):
        return (
            f"Audit({self.get_event_type_display()}) "
            f"@ {self.created_at:%Y-%m-%d %H:%M}"
        )
