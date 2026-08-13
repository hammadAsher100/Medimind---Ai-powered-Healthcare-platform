from django.conf import settings
from django.db import models
from medimind.fields import EncryptedTextField


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
    report_date = models.DateField(null=True, blank=True)
    notes = EncryptedTextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    extracted_text = EncryptedTextField(blank=True)
    analysis_result = models.JSONField(default=dict, blank=True)
    summary = EncryptedTextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.user} - {self.report_type} - {self.uploaded_at:%Y-%m-%d}"
