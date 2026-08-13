"""Proxy view to forward /ai/ requests to the FastAPI AI service.

In Docker, nginx handles this routing. For local development (and as a
fallback), Django proxies requests to the FastAPI service itself.
"""

import json
import logging

import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from rest_framework_simplejwt.tokens import AccessToken

logger = logging.getLogger(__name__)


@login_required
@require_POST
def ai_proxy(request, path=""):
    """Forward POST requests to the FastAPI AI service.

    The browser-facing fallback requires an authenticated Django session and
    normal CSRF validation. Production Nginx applies the same auth decision.
    """
    fastapi_url = getattr(settings, "FASTAPI_URL", "http://localhost:8001")
    target_url = f"{fastapi_url}/{path}"

    try:
        body = json.loads(request.body) if request.body else {}
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"detail": "Invalid JSON body."}, status=400)

    try:
        resp = requests.post(
            target_url,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {AccessToken.for_user(request.user)}",
            },
            timeout=120,
        )
        try:
            data = resp.json()
        except ValueError:
            data = {"detail": resp.text[:500]}
        return JsonResponse(data, status=resp.status_code, safe=False)
    except requests.ConnectionError:
        logger.error("AI service connection failed: %s", target_url)
        return JsonResponse(
            {"detail": "AI service is not available. Please ensure the FastAPI service is running."},
            status=503,
        )
    except requests.Timeout:
        return JsonResponse(
            {"detail": "AI service timed out. Please try again."},
            status=504,
        )
    except Exception:
        logger.exception("Unexpected error proxying to AI service")
        return JsonResponse({"detail": "The AI request could not be completed."}, status=500)
