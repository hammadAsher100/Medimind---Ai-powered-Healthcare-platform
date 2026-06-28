from rest_framework import serializers

from .models import MedicalReport


class MedicalReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalReport
        fields = (
            "id",
            "file",
            "report_type",
            "uploaded_at",
            "extracted_text",
            "analysis_result",
            "summary",
        )
        read_only_fields = ("id", "uploaded_at", "extracted_text", "analysis_result", "summary")
