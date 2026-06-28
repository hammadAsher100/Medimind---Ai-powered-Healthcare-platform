from django.conf import settings
from django.db import models


class TimelineEvent(models.Model):
    EVENT_TYPES = [
        ("report_uploaded", "Report uploaded"),
        ("prediction_made", "Prediction made"),
        ("recommendation_generated", "Recommendation generated"),
        ("medicine_explained", "Medicine explained"),
        ("score_calculated", "Score calculated"),
        ("emergency_detected", "Emergency detected"),
        ("knowledge_indexed", "Knowledge indexed"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="timeline_events")
    event_type = models.CharField(max_length=64, choices=EVENT_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.event_type}: {self.title}"
