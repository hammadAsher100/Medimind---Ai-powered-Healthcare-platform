import requests
from django.conf import settings
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from health_score.models import HealthScore
from timeline.services import create_timeline_event

from .models import MedicalReport
from .serializers import MedicalReportSerializer


def _safe_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract_score_inputs(analysis_result):
    labs = analysis_result.get("lab_values", [])
    values = {str(item.get("name", "")).lower(): _safe_float(item.get("value")) for item in labs}
    return {
        "glucose": values.get("blood sugar") or values.get("glucose") or values.get("hba1c"),
        "ldl": values.get("ldl"),
        "cholesterol": values.get("cholesterol"),
    }


class ReportUploadView(generics.CreateAPIView):
    serializer_class = MedicalReportSerializer
    parser_classes = [MultiPartParser, FormParser]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        report = serializer.save(user=request.user)

        try:
            report.file.open("rb")
            files = {"file": (report.file.name, report.file.file, "application/pdf")}
            response = requests.post(
                f"{settings.FASTAPI_URL}/analyze-report",
                data={"user_id": request.user.id},
                files=files,
                timeout=90,
            )
            response.raise_for_status()
            analysis = response.json()
        except requests.RequestException as exc:
            report.analysis_result = {"error": str(exc)}
            report.summary = "Report uploaded, but AI analysis failed."
            report.save(update_fields=["analysis_result", "summary"])
            create_timeline_event(request.user, "report_uploaded", "Report uploaded", report.summary, {"report_id": report.id})
            return Response(MedicalReportSerializer(report).data, status=status.HTTP_202_ACCEPTED)
        finally:
            report.file.close()

        report.extracted_text = analysis.get("extracted_text", "")
        report.analysis_result = analysis.get("analysis", analysis)
        report.summary = report.analysis_result.get("summary", "")
        report.save(update_fields=["extracted_text", "analysis_result", "summary"])
        create_timeline_event(request.user, "report_uploaded", "Report analyzed", report.summary, {"report_id": report.id})

        score_inputs = _extract_score_inputs(report.analysis_result)
        if any(value is not None for value in score_inputs.values()):
            try:
                score_response = requests.post(f"{settings.FASTAPI_URL}/calculate-health-score", json=score_inputs, timeout=30)
                score_response.raise_for_status()
                score_result = score_response.json()
                HealthScore.objects.create(
                    user=request.user,
                    score=score_result["score"],
                    sugar_level=score_inputs.get("glucose"),
                    cholesterol=score_inputs.get("ldl") or score_inputs.get("cholesterol"),
                    strengths=score_result.get("strengths", []),
                    improvements=score_result.get("needs_improvement", []),
                )
                create_timeline_event(request.user, "score_calculated", "Health score recalculated from report", "", score_result)
            except requests.RequestException:
                pass

        return Response(MedicalReportSerializer(report).data, status=status.HTTP_201_CREATED)


class ReportListView(generics.ListAPIView):
    serializer_class = MedicalReportSerializer

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user)


class ReportDetailView(generics.RetrieveAPIView):
    serializer_class = MedicalReportSerializer

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user)
