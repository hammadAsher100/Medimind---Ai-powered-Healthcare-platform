from django.contrib import admin

from .models import HealthScore


@admin.register(HealthScore)
class HealthScoreAdmin(admin.ModelAdmin):
    list_display = ("user", "score", "bmi", "created_at")
    list_filter = ("created_at",)
