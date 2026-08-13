from rest_framework import serializers

from .models import MedicalReport


class MedicalReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalReport
        fields = (
            "id",
            "file",
            "report_type",
            "report_date",
            "notes",
            "uploaded_at",
            "extracted_text",
            "analysis_result",
            "summary",
        )
        read_only_fields = ("id", "uploaded_at", "extracted_text", "analysis_result", "summary")


class MedicalReportSummarySerializer(serializers.ModelSerializer):
    """Bounded report representation for list views; omits extracted medical text."""

    class Meta:
        model = MedicalReport
        fields = ("id", "file", "report_type", "report_date", "notes", "uploaded_at", "summary")
        read_only_fields = ("id", "uploaded_at", "summary")
