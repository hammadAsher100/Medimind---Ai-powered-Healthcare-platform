"""Production readiness endpoint used by the on-demand wake controller."""

from __future__ import annotations

import logging

import requests
from django.conf import settings
from django.db import connection
from django.http import JsonResponse
from django.views.decorators.http import require_GET

logger = logging.getLogger(__name__)


@require_GET
def readiness(request):
    checks = {"database": False, "ai_service": False}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            checks["database"] = cursor.fetchone() == (1,)
    except Exception:
        logger.exception("Readiness database check failed")

    try:
        response = requests.get(f"{settings.FASTAPI_URL}/readyz", timeout=15)
        if response.ok and "application/json" in response.headers.get("Content-Type", "").lower():
            payload = response.json()
            checks["ai_service"] = payload.get("ready") is True
    except Exception:
        logger.exception("Readiness AI service check failed")

    ready = all(checks.values())
    return JsonResponse(
        {"ready": ready, "service": "medimind-web", "checks": checks},
        status=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )
