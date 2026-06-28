import os

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm.provider import LLMProvider

router = APIRouter(tags=["comparison"])


class CompareReportsRequest(BaseModel):
    first_report_id: int
    second_report_id: int
    auth_token: str


def _lab_map(report: dict) -> dict:
    labs = report.get("analysis_result", {}).get("lab_values", [])
    return {str(item.get("name", "")).lower(): item for item in labs}


@router.post("/compare-reports")
async def compare_reports(payload: CompareReportsRequest):
    try:
        django_url = os.environ.get("DJANGO_URL", "http://django:8000")
        headers = {"Authorization": f"Bearer {payload.auth_token}"}
        first = requests.get(f"{django_url}/api/reports/{payload.first_report_id}/", headers=headers, timeout=30)
        second = requests.get(f"{django_url}/api/reports/{payload.second_report_id}/", headers=headers, timeout=30)
        first.raise_for_status()
        second.raise_for_status()
        first_labs = _lab_map(first.json())
        second_labs = _lab_map(second.json())
        rows = []
        for name in sorted(set(first_labs) & set(second_labs)):
            try:
                old = float(first_labs[name].get("value"))
                new = float(second_labs[name].get("value"))
            except (TypeError, ValueError):
                continue
            absolute = new - old
            percent = (absolute / old * 100) if old else 0
            trend = "stable" if abs(percent) < 5 else "improved" if absolute < 0 else "worsened"
            rows.append({"name": name, "old": old, "new": new, "absolute_difference": absolute, "percentage_difference": percent, "trend": trend})
        narrative = LLMProvider().chat(
            [
                {"role": "system", "content": "Explain lab trends cautiously in patient-friendly language. Do not diagnose."},
                {"role": "user", "content": str(rows)},
            ]
        )
        overall = "stable"
        if any(row["trend"] == "worsened" for row in rows):
            overall = "worsened"
        elif any(row["trend"] == "improved" for row in rows):
            overall = "improved"
        return {"comparison_table": rows, "narrative": narrative, "trend": overall}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
