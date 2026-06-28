import json
import re
from io import BytesIO

import pdfplumber
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from llm.provider import LLMProvider

router = APIRouter(tags=["reports"])

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


@router.post("/analyze-report")
async def analyze_report(file: UploadFile = File(...), user_id: int = Form(...)):
    try:
        content = await file.read()
        with pdfplumber.open(BytesIO(content)) as pdf:
            extracted = "\n".join(page.extract_text() or "" for page in pdf.pages)
        cleaned = re.sub(r"\s+", " ", extracted).strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="No extractable text was found in the PDF.")

        provider = LLMProvider()
        response = provider.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"User ID: {user_id}\nReport text:\n{cleaned[:12000]}"},
            ]
        )
        analysis = _extract_json(response)
        return {"user_id": user_id, "extracted_text": cleaned, "analysis": analysis}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
