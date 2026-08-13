"""DRF serializers for Reviews models."""

from rest_framework import serializers

from .models import AuditEvent, ModelFeedback, ReviewDecision


class ReviewDecisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewDecision
        fields = ("id", "recommendation_type", "recommendation_id", "ai_summary", "clinician_decision", "clinician_notes", "created_at")
        read_only_fields = ("id", "created_at")


class ModelFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelFeedback
        fields = ("id", "model_name", "prediction_id", "feedback_type", "original_output", "corrected_output", "notes", "created_at")
        read_only_fields = ("id", "created_at")


class AuditEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditEvent
        fields = ("id", "event_type", "event_detail", "source", "created_at")
        read_only_fields = ("id", "created_at", "source")
