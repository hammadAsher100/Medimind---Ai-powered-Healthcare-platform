from rest_framework import serializers

from .models import TimelineEvent


class TimelineEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = TimelineEvent
        fields = ("id", "event_type", "title", "description", "metadata", "created_at")
        read_only_fields = fields
