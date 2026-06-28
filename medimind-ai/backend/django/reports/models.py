from django.conf import settings
from django.db import models


class MedicalReport(models.Model):
    REPORT_TYPES = [
        ("blood", "Blood"),
        ("lab", "Lab"),
        ("imaging", "Imaging"),
        ("other", "Other"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="medical_reports")
    file = models.FileField(upload_to="reports/%Y/%m/%d/")
    report_type = models.CharField(max_length=16, choices=REPORT_TYPES, default="other")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    extracted_text = models.TextField(blank=True)
    analysis_result = models.JSONField(default=dict, blank=True)
    summary = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.user} - {self.report_type} - {self.uploaded_at:%Y-%m-%d}"
