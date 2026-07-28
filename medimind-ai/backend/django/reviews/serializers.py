"""DRF serializers for Reviews models."""

from rest_framework import serializers

from .models import AuditEvent, ModelFeedback, ReviewDecision


class ReviewDecisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReviewDecision
        fields = "__all__"
        read_only_fields = ["id", "created_at"]


class ModelFeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelFeedback
        fields = "__all__"
        read_only_fields = ["id", "created_at"]


class AuditEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = AuditEvent
        fields = "__all__"
        read_only_fields = ["id", "created_at"]
