import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from ai_service.app import app
from routers import report_analyzer


client = TestClient(app)


def llm_response():
    return json.dumps(
        {
            "lab_values": [],
            "summary": "Report analyzed for testing.",
            "plain_english_explanations": {},
            "recommendations": [],
        }
    )


def test_pdf_report_analysis(monkeypatch):
    monkeypatch.setattr(report_analyzer.LLMProvider, "chat", lambda self, messages: llm_response())
    pdf_path = Path(__file__).resolve().parents[1] / "testing" / "blood_report_jan2026.pdf"

    response = client.post(
        "/analyze-report",
        data={"user_id": "1"},
        files={"file": ("blood-report.pdf", pdf_path.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["extraction_method"] == "pdf_text"
    assert payload["extracted_text"]
    assert payload["analysis"]["summary"] == "Report analyzed for testing."


def test_image_report_analysis_uses_ocr(monkeypatch):
    monkeypatch.setattr(report_analyzer.LLMProvider, "chat", lambda self, messages: llm_response())
    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(image_to_string=lambda image: "Hemoglobin 14 g/dL"),
    )
    buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(buffer, format="PNG")

    response = client.post(
        "/analyze-report",
        data={"user_id": "1"},
        files={"file": ("lab-report.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["extraction_method"] == "image_ocr"
    assert payload["extracted_text"] == "Hemoglobin 14 g/dL"


def test_report_analyzer_rejects_unsupported_file():
    response = client.post(
        "/analyze-report",
        data={"user_id": "1"},
        files={"file": ("report.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


def test_report_analyzer_rejects_empty_file():
    response = client.post(
        "/analyze-report",
        data={"user_id": "1"},
        files={"file": ("report.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


def test_report_analyzer_rejects_malformed_pdf_as_json():
    response = client.post(
        "/analyze-report",
        data={"user_id": "1"},
        files={"file": ("broken.pdf", b"%PDF-not-valid", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["detail"] == "The uploaded PDF is invalid."
