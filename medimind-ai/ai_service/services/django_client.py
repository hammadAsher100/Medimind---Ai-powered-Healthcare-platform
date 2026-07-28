"""Client for communicating with Django backend for persistence."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DJANGO_BASE = os.environ.get(
    "DJANGO_BASE_URL",
    "http://localhost:8000",
)


def _post_django(path: str, data: dict[str, Any]) -> dict[str, Any]:
    """Post data to Django backend with error handling."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{DJANGO_BASE}{path}",
                json=data,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Django API call failed: %s %s — %s", path, e, data)
        return {"status": "error", "message": str(e)}


def record_model_feedback(data: dict[str, Any]) -> dict[str, Any]:
    """Record model feedback via Django backend."""
    return _post_django("/api/reviews/feedback/", data)


def record_review_decision(data: dict[str, Any]) -> dict[str, Any]:
    """Record clinician review decision via Django backend."""
    return _post_django("/api/reviews/decision/", data)


def record_audit_event(data: dict[str, Any]) -> dict[str, Any]:
    """Record an audit event via Django backend."""
    return _post_django("/api/reviews/audit/", data)


def save_observation(data: dict[str, Any]) -> dict[str, Any]:
    """Save a clinical observation via Django backend."""
    return _post_django("/api/clinical/observations/", data)


def save_conflict(data: dict[str, Any]) -> dict[str, Any]:
    """Save a detected conflict via Django backend."""
    return _post_django("/api/clinical/conflicts/", data)


def save_patient_state(data: dict[str, Any]) -> dict[str, Any]:
    """Save a patient state snapshot via Django backend."""
    return _post_django("/api/clinical/state/", data)


def save_medication_alert(data: dict[str, Any]) -> dict[str, Any]:
    """Save a medication safety alert via Django backend."""
    return _post_django("/api/medication/alerts/", data)
