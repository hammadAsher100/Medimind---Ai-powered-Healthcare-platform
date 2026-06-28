from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class HealthScore(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="health_scores")
    score = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    bmi = models.FloatField(null=True, blank=True)
    sugar_level = models.FloatField(null=True, blank=True)
    cholesterol = models.FloatField(null=True, blank=True)
    blood_pressure_systolic = models.IntegerField(null=True, blank=True)
    blood_pressure_diastolic = models.IntegerField(null=True, blank=True)
    lifestyle_score = models.IntegerField(default=0)
    strengths = models.JSONField(default=list, blank=True)
    improvements = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.user}: {self.score}"
