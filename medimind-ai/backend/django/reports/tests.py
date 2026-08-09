import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from django.urls import reverse
from PIL import Image
from rest_framework import status
from rest_framework.test import APITestCase

from .models import MedicalReport
from .views import MAX_FILE_SIZE


VALID_PDF = (Path(__file__).resolve().parents[3] / "testing" / "blood_report_jan2026.pdf").read_bytes()


def analysis_response():
    response = Mock()
    response.status_code = 200
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = {
        "extracted_text": "Hemoglobin 14 g/dL",
        "analysis": {
            "lab_values": [],
            "summary": "No urgent abnormality was identified.",
            "plain_english_explanations": {},
            "recommendations": [],
        },
        "extraction_method": "test",
    }
    return response


def image_upload(name, image_format, content_type):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format=image_format)
    return SimpleUploadedFile(name, buffer.getvalue(), content_type=content_type)


class ReportUploadTests(APITestCase):
    def setUp(self):
        self.media_directory = tempfile.TemporaryDirectory()
        self.settings_override = override_settings(MEDIA_ROOT=self.media_directory.name)
        self.settings_override.enable()
        self.user = get_user_model().objects.create_user(
            username="report-user",
            password="test-password-123",
        )
        self.client.force_authenticate(self.user)
        self.url = reverse("report-upload")

    def tearDown(self):
        self.settings_override.disable()
        self.media_directory.cleanup()

    @patch("reports.views.requests.post")
    def test_valid_pdf_is_saved_and_analyzed(self, post):
        post.return_value = analysis_response()
        upload = SimpleUploadedFile(
            "blood-report.pdf",
            VALID_PDF,
            content_type="application/pdf",
        )

        response = self.client.post(
            self.url,
            {
                "file": upload,
                "report_type": "blood",
                "report_date": "2026-08-09",
                "notes": "Annual screening",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(response.data["analysis_status"], "completed")
        report = MedicalReport.objects.get()
        self.assertEqual(str(report.report_date), "2026-08-09")
        self.assertEqual(report.notes, "Annual screening")
        self.assertTrue(Path(report.file.path).is_file())
        self.assertTrue(Path(report.file.path).is_relative_to(Path(self.media_directory.name)))

    @patch("reports.views.requests.post")
    def test_valid_png_and_jpeg_are_saved(self, post):
        for name, image_format, content_type in (
            ("report.png", "PNG", "image/png"),
            ("report.jpg", "JPEG", "image/jpeg"),
        ):
            with self.subTest(name=name):
                post.return_value = analysis_response()
                response = self.client.post(
                    self.url,
                    {"file": image_upload(name, image_format, content_type), "report_type": "imaging"},
                    format="multipart",
                )

                self.assertEqual(response.status_code, status.HTTP_201_CREATED)
                self.assertEqual(response.data["analysis_status"], "completed")
                report = MedicalReport.objects.get(file__endswith=name)
                self.assertTrue(Path(report.file.path).is_file())

    @patch("reports.views.requests.post")
    def test_unsupported_file_is_rejected_without_database_record(self, post):
        upload = SimpleUploadedFile("report.txt", b"not a medical report", content_type="text/plain")

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(MedicalReport.objects.count(), 0)
        post.assert_not_called()

    @patch("reports.views.requests.post")
    def test_empty_upload_is_rejected(self, post):
        upload = SimpleUploadedFile("empty.pdf", b"", content_type="application/pdf")

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(MedicalReport.objects.count(), 0)
        post.assert_not_called()

    @patch("reports.views.requests.post")
    def test_oversized_upload_is_rejected(self, post):
        upload = SimpleUploadedFile(
            "large.pdf",
            b"%PDF-" + (b"0" * MAX_FILE_SIZE),
            content_type="application/pdf",
        )

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(MedicalReport.objects.count(), 0)
        post.assert_not_called()

    @patch("reports.views.requests.post")
    def test_extension_and_contents_must_match(self, post):
        upload = image_upload("disguised.pdf", "PNG", "application/pdf")

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertFalse(response.data["success"])
        self.assertEqual(MedicalReport.objects.count(), 0)
        post.assert_not_called()

    @patch("reports.views.requests.post")
    def test_malformed_pdf_returns_clean_json_error(self, post):
        upload = SimpleUploadedFile(
            "broken.pdf",
            b"%PDF-this-is-not-a-valid-document",
            content_type="application/pdf",
        )

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertFalse(response.data["success"])
        self.assertEqual(response.data["error"], "The uploaded PDF could not be decoded safely.")
        self.assertEqual(MedicalReport.objects.count(), 0)
        post.assert_not_called()

    @patch("reports.views.requests.post", side_effect=requests.Timeout("upstream timeout"))
    def test_analysis_timeout_keeps_uploaded_file_and_returns_json(self, post):
        upload = SimpleUploadedFile(
            "report.pdf",
            VALID_PDF,
            content_type="application/pdf",
        )

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(response.data["analysis_status"], "pending")
        report = MedicalReport.objects.get()
        self.assertTrue(Path(report.file.path).is_file())
        self.assertNotIn("upstream timeout", str(response.data))

    @patch("reports.views.requests.post")
    def test_non_json_analysis_response_keeps_upload(self, post):
        post.return_value = analysis_response()
        post.return_value.headers = {"Content-Type": "text/html"}
        upload = SimpleUploadedFile(
            "report.pdf",
            VALID_PDF,
            content_type="application/pdf",
        )

        response = self.client.post(self.url, {"file": upload}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(response.data["analysis_status"], "pending")
        self.assertEqual(MedicalReport.objects.count(), 1)
