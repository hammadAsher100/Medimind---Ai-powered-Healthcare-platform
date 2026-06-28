from django.contrib import admin

from .models import MedicalReport


@admin.register(MedicalReport)
class MedicalReportAdmin(admin.ModelAdmin):
    list_display = ("user", "report_type", "uploaded_at", "summary")
    list_filter = ("report_type", "uploaded_at")
