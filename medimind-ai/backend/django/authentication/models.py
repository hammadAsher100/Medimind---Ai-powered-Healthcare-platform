from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    GENDER_CHOICES = [
        ("female", "Female"),
        ("male", "Male"),
        ("other", "Other"),
        ("prefer_not_to_say", "Prefer not to say"),
    ]

    date_of_birth = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=32, choices=GENDER_CHOICES, blank=True)
    phone_number = models.CharField(max_length=32, blank=True)

    def __str__(self) -> str:
        return self.get_full_name() or self.username
