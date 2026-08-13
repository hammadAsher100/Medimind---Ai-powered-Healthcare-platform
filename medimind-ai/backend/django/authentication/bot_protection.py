import logging

import requests
from django.conf import settings
from rest_framework.exceptions import ValidationError

logger = logging.getLogger(__name__)

TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def verify_human(request):
    """Reject honeypot submissions and verify Turnstile when it is enabled."""
    if str(request.data.get("website", "")).strip():
        raise ValidationError({"detail": "Unable to verify this request."})

    if not settings.TURNSTILE_ENABLED:
        return

    token = str(request.data.get("turnstile_token", "")).strip()
    if not token:
        raise ValidationError({"detail": "Complete the security check and try again."})

    try:
        response = requests.post(
            TURNSTILE_VERIFY_URL,
            data={
                "secret": settings.TURNSTILE_SECRET_KEY,
                "response": token,
                "remoteip": request.META.get("REMOTE_ADDR", ""),
            },
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError):
        logger.exception("Turnstile verification service failed")
        raise ValidationError(
            {"detail": "Security verification is temporarily unavailable. Please try again."}
        )

    if result.get("success") is not True:
        logger.warning("Turnstile rejected an authentication request")
        raise ValidationError({"detail": "Security verification failed. Please try again."})
