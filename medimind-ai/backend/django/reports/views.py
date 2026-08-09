import logging
import os

import pdfplumber
import requests
from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from PIL import Image, UnidentifiedImageError
from rest_framework import generics, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from health_score.models import HealthScore
from timeline.services import create_timeline_event

from .models import MedicalReport
from .serializers import MedicalReportSerializer

logger = logging.getLogger(__name__)

ALLOWED_FILE_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
ALLOWED_MIME_TYPES = set(ALLOWED_FILE_TYPES.values())
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_PDF_PAGES = 100
MAX_IMAGE_PIXELS = 40_000_000


def _validate_uploaded_file(file_obj):
    """Validate size, extension, declared type, and file signature."""
    if not file_obj or file_obj.size <= 0:
        raise ValueError("The uploaded report is empty.")
    if file_obj.size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)} MB.")

    ext = os.path.splitext(file_obj.name)[1].lower()
    if ext not in ALLOWED_FILE_TYPES:
        raise ValueError("Only PDF, PNG, JPG, and JPEG reports are supported.")

    declared_type = (getattr(file_obj, "content_type", "") or "").split(";", 1)[0].strip().lower()
    if declared_type and declared_type not in ALLOWED_MIME_TYPES | {"application/octet-stream"}:
        raise ValueError("The uploaded report has an unsupported content type.")

    file_obj.seek(0)
    try:
        if ext == ".pdf":
            header = file_obj.read(5)
            if header != b"%PDF-":
                raise ValueError("File does not appear to be a valid PDF document.")
            file_obj.seek(0)
            try:
                with pdfplumber.open(file_obj) as pdf:
                    page_count = len(pdf.pages)
            except Exception as exc:
                raise ValueError("The uploaded PDF could not be decoded safely.") from exc
            if page_count > MAX_PDF_PAGES:
                raise ValueError("The PDF contains too many pages.")
            detected_type = "application/pdf"
        else:
            with Image.open(file_obj) as image:
                image.verify()
                image_format = (image.format or "").upper()
            detected_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(image_format)
            if detected_type is None:
                raise ValueError("Image report must be a valid PNG or JPEG file.")
            file_obj.seek(0)
            with Image.open(file_obj) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("The report image dimensions are too large.")
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image report could not be decoded safely.") from exc
    finally:
        file_obj.seek(0)

    expected_type = ALLOWED_FILE_TYPES[ext]
    if detected_type != expected_type:
        raise ValueError("The report file extension does not match its contents.")
    if declared_type in ALLOWED_MIME_TYPES and declared_type != detected_type:
        raise ValueError("The report content type does not match its contents.")
    return detected_type


def _error_response(message, http_status, *, details=None):
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    return Response(payload, status=http_status)


def _serialize_report(report, analysis_status):
    payload = dict(MedicalReportSerializer(report).data)
    payload["success"] = True
    payload["analysis_status"] = analysis_status
    return payload


def _record_timeline_event(user, event_type, title, description, metadata):
    try:
        create_timeline_event(user, event_type, title, description, metadata)
    except Exception:
        logger.exception("Unable to create timeline event for report_id=%s", metadata.get("report_id"))


class ReportJSONErrorMixin:
    """Ensure report API failures never fall through to an HTML traceback."""

    def handle_exception(self, exc):
        if isinstance(exc, RequestDataTooBig):
            logger.warning("Report upload exceeded Django's request-size limit")
            return _error_response(
                "The uploaded report exceeds the 20 MB limit.",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            return super().handle_exception(exc)
        except Exception:
            logger.exception("Unexpected report API failure")
            return _error_response(
                "Unable to process the medical report.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


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


class ReportUploadView(ReportJSONErrorMixin, generics.CreateAPIView):
    serializer_class = MedicalReportSerializer
    parser_classes = [MultiPartParser, FormParser]

    def create(self, request, *args, **kwargs):
        file_obj = request.FILES.get("file")
        if file_obj is None:
            return _error_response("Choose a report file to upload.", status.HTTP_400_BAD_REQUEST)

        try:
            detected_content_type = _validate_uploaded_file(file_obj)
        except ValueError as exc:
            return _error_response(str(exc), status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return _error_response(
                "Please correct the report details and try again.",
                status.HTTP_400_BAD_REQUEST,
                details=serializer.errors,
            )
        try:
            report = serializer.save(user=request.user)
        except Exception:
            logger.exception("Unable to save uploaded report for user_id=%s", request.user.pk)
            return _error_response(
                "Unable to save the uploaded report.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        try:
            report.file.open("rb")
            files = {"file": (os.path.basename(report.file.name), report.file.file, detected_content_type)}
            response = requests.post(
                f"{settings.FASTAPI_URL}/analyze-report",
                data={"user_id": request.user.id},
                files=files,
                timeout=90,
            )
            if response.status_code >= 400:
                logger.warning(
                    "Report analysis service rejected report_id=%s with status=%s",
                    report.id,
                    response.status_code,
                )
                raise requests.HTTPError(response=response)
            content_type = response.headers.get("Content-Type", "").lower()
            if "application/json" not in content_type and "+json" not in content_type:
                raise ValueError("Report analysis service returned non-JSON data.")
            analysis = response.json()
            if not isinstance(analysis, dict) or not isinstance(analysis.get("analysis"), dict):
                raise ValueError("Report analysis response did not match the expected contract.")
        except requests.Timeout:
            logger.exception("Report analysis timed out for report_id=%s", report.id)
            report.analysis_result = {"error": {"code": "analysis_timeout"}}
            report.summary = "Report uploaded, but AI analysis failed."
            report.save(update_fields=["analysis_result", "summary"])
            _record_timeline_event(
                request.user,
                "report_uploaded",
                "Report uploaded",
                report.summary,
                {"report_id": report.id},
            )
            return Response(_serialize_report(report, "pending"), status=status.HTTP_202_ACCEPTED)
        except (requests.RequestException, ValueError):
            logger.exception("Report analysis failed for report_id=%s", report.id)
            report.analysis_result = {"error": {"code": "analysis_unavailable"}}
            report.summary = "Report uploaded, but AI analysis is currently unavailable."
            report.save(update_fields=["analysis_result", "summary"])
            _record_timeline_event(
                request.user,
                "report_uploaded",
                "Report uploaded",
                report.summary,
                {"report_id": report.id},
            )
            return Response(_serialize_report(report, "pending"), status=status.HTTP_202_ACCEPTED)
        finally:
            try:
                report.file.close()
            except Exception:
                pass

        report.extracted_text = analysis.get("extracted_text", "")
        report.analysis_result = analysis["analysis"]
        report.summary = report.analysis_result.get("summary", "")
        report.save(update_fields=["extracted_text", "analysis_result", "summary"])
        _record_timeline_event(
            request.user,
            "report_uploaded",
            "Report analyzed",
            report.summary,
            {"report_id": report.id},
        )

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
            except Exception:
                logger.exception("Unable to update health score from report_id=%s", report.id)

        return Response(_serialize_report(report, "completed"), status=status.HTTP_201_CREATED)


class ReportListView(ReportJSONErrorMixin, generics.ListAPIView):
    serializer_class = MedicalReportSerializer

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user)


class ReportDetailView(ReportJSONErrorMixin, generics.RetrieveAPIView):
    serializer_class = MedicalReportSerializer

    def get_queryset(self):
        return MedicalReport.objects.filter(user=self.request.user)
