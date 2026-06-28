from django.conf import settings
from django.db import models


class MedicalProfile(models.Model):
    BLOOD_TYPES = [
        ("A+", "A+"),
        ("A-", "A-"),
        ("B+", "B+"),
        ("B-", "B-"),
        ("AB+", "AB+"),
        ("AB-", "AB-"),
        ("O+", "O+"),
        ("O-", "O-"),
    ]
    SMOKING_CHOICES = [("never", "Never"), ("former", "Former"), ("current", "Current")]
    ALCOHOL_CHOICES = [("none", "None"), ("light", "Light"), ("moderate", "Moderate"), ("heavy", "Heavy")]
    EXERCISE_CHOICES = [("none", "None"), ("weekly", "Weekly"), ("several_weekly", "Several times weekly"), ("daily", "Daily")]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="medical_profile")
    blood_type = models.CharField(max_length=3, choices=BLOOD_TYPES, blank=True)
    height_cm = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    weight_kg = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    smoking_status = models.CharField(max_length=16, choices=SMOKING_CHOICES, blank=True)
    alcohol_use = models.CharField(max_length=16, choices=ALCOHOL_CHOICES, blank=True)
    exercise_frequency = models.CharField(max_length=24, choices=EXERCISE_CHOICES, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def bmi(self):
        if self.height_cm and self.weight_kg and self.height_cm > 0:
            meters = float(self.height_cm) / 100
            return round(float(self.weight_kg) / (meters * meters), 2)
        return None

    def __str__(self) -> str:
        return f"Medical profile for {self.user}"


class Allergy(models.Model):
    SEVERITY_CHOICES = [("mild", "Mild"), ("moderate", "Moderate"), ("severe", "Severe")]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="allergies")
    allergen = models.CharField(max_length=255)
    severity = models.CharField(max_length=16, choices=SEVERITY_CHOICES)
    notes = models.TextField(blank=True)

    def __str__(self) -> str:
        return f"{self.allergen} ({self.severity})"


class FamilyHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="family_history")
    condition = models.CharField(max_length=255)
    relation = models.CharField(max_length=128)
    notes = models.TextField(blank=True)

    def __str__(self) -> str:
        return f"{self.condition} - {self.relation}"
