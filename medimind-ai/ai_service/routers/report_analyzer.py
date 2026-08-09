import json
import logging
import re
from io import BytesIO

import pdfplumber
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from llm.provider import LLMProvider

router = APIRouter(tags=["reports"])
logger = logging.getLogger(__name__)

MAX_REPORT_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_REPORT_TYPES = {"application/pdf", "image/png", "image/jpeg"}

SYSTEM_PROMPT = """You are a medical report analyzer. Extract and explain the following in simple patient language:
1. All lab values (name, value, unit, normal range, status: normal/high/low)
2. Key findings summary
3. What each abnormal value means in plain English
4. Recommended follow-up actions

Focus on: Cholesterol, HDL, LDL, Triglycerides, Blood Sugar, HbA1c, Hemoglobin, Vitamin D, Vitamin B12, Creatinine, Urea, TSH, WBC, RBC, Platelets.

Return ONLY valid JSON with keys: lab_values (array), summary, plain_english_explanations (object), recommendations (array)."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _extract_report_text(content: bytes, content_type: str) -> tuple[str, str]:
    if content_type == "application/pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="The uploaded PDF is invalid.")
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                if len(pdf.pages) > MAX_PDF_PAGES:
                    raise HTTPException(status_code=400, detail="The PDF contains too many pages.")
                extracted = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail="The uploaded PDF is invalid.") from exc
        return extracted, "pdf_text"

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            detected_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get((image.format or "").upper())
            if detected_type != content_type:
                raise HTTPException(status_code=400, detail="The report content type does not match its contents.")
            image = image.convert("RGB")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="The report image is too large.")
            import pytesseract

            extracted = pytesseract.image_to_string(image)
        return extracted, "image_ocr"
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded report image is invalid.") from exc


@router.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...), user_id: int = Form(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded report is empty.")
        if len(content) > MAX_REPORT_BYTES:
            raise HTTPException(status_code=413, detail="The uploaded report exceeds the 20 MB limit.")

        content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
        if content_type not in ALLOWED_REPORT_TYPES:
            raise HTTPException(status_code=400, detail="Only PDF, PNG, and JPEG reports are supported.")

        extracted, extraction_method = _extract_report_text(content, content_type)
        cleaned = re.sub(r"\s+", " ", extracted).strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="No readable text was found in the report.")

        provider = LLMProvider()
        response = provider.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Report text:\n{cleaned[:12000]}"},
            ]
        )
        analysis = _extract_json(response)
        logger.info(
            "Report analysis completed user_id=%s content_type=%s extraction_method=%s text_length=%s",
            user_id,
            content_type,
            extraction_method,
            len(cleaned),
        )
        return {
            "user_id": user_id,
            "extracted_text": cleaned,
            "analysis": analysis,
            "extraction_method": extraction_method,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected report analysis failure for user_id=%s", user_id)
        raise HTTPException(status_code=500, detail="Unable to analyze the medical report.") from exc
