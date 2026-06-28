from django.conf import settings
from django.db import models


class Prediction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="predictions")
    disease = models.CharField(max_length=64)
    input_data = models.JSONField(default=dict)
    risk_percentage = models.FloatField()
    risk_level = models.CharField(max_length=32)
    prediction = models.IntegerField()
    shap_explanation = models.JSONField(default=dict)
    ai_recommendation = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user} {self.disease} {self.risk_percentage:.1f}%"


class Recommendation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="recommendations")
    category = models.CharField(max_length=64)
    content = models.TextField()
    source = models.CharField(max_length=64, default="agent")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.category}: {self.user}"


class KnowledgeDocument(models.Model):
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="knowledge_documents")
    file = models.FileField(upload_to="knowledge/%Y/%m/%d/")
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=255, blank=True)
    indexed_chunks = models.IntegerField(default=0)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title
