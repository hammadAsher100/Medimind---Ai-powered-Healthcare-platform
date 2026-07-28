"""Admin registration for reviews models."""

from django.contrib import admin

from .models import AuditEvent, ModelFeedback, ReviewDecision


@admin.register(ReviewDecision)
class ReviewDecisionAdmin(admin.ModelAdmin):
    list_display = ["user", "recommendation_type", "clinician_decision", "created_at"]
    list_filter = ["clinician_decision", "recommendation_type"]


@admin.register(ModelFeedback)
class ModelFeedbackAdmin(admin.ModelAdmin):
    list_display = ["user", "model_name", "feedback_type", "is_synthetic_data", "created_at"]
    list_filter = ["feedback_type", "model_name", "is_synthetic_data"]


@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin):
    list_display = ["user", "event_type", "source", "created_at"]
    list_filter = ["event_type", "source"]
    readonly_fields = ["created_at"]
