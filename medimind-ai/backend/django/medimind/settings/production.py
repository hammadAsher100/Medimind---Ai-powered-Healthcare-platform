import os

from django.core.exceptions import ImproperlyConfigured

from .base import *  # noqa: F401,F403


def _env_bool(name, default=False):
    """Parse common true/false environment values without bool("False") bugs."""
    value = os.environ.get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ImproperlyConfigured(f"{name} must be a boolean value")


def _env_csv(name, default=""):
    """Return a clean list from a comma-separated environment setting."""
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


DEBUG = False
_PRODUCTION_HOSTS = ["medimind-ai.online", "www.medimind-ai.online"]
_PRODUCTION_ORIGINS = [
    "https://medimind-ai.online",
    "https://www.medimind-ai.online",
]
ALLOWED_HOSTS = list(dict.fromkeys(_env_csv("DJANGO_ALLOWED_HOSTS") + _PRODUCTION_HOSTS))

SECURE_SSL_REDIRECT = _env_bool("SECURE_SSL_REDIRECT", True)
SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", True)
CSRF_COOKIE_SECURE = _env_bool("CSRF_COOKIE_SECURE", True)

# Nginx terminates TLS and forwards the original request scheme.
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Start HSTS conservatively. Increase this only after HTTPS renewal has been
# proven reliable; preload requires a much longer max age.
SECURE_HSTS_SECONDS = int(os.environ.get("SECURE_HSTS_SECONDS", "300").strip())
SECURE_HSTS_INCLUDE_SUBDOMAINS = _env_bool("SECURE_HSTS_INCLUDE_SUBDOMAINS", False)
SECURE_HSTS_PRELOAD = _env_bool("SECURE_HSTS_PRELOAD", False)

# Explicitly restrict browser origins in production.
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = list(
    dict.fromkeys(_env_csv("CORS_ALLOWED_ORIGINS") + _PRODUCTION_ORIGINS)
)

CSRF_TRUSTED_ORIGINS = list(
    dict.fromkeys(_env_csv("CSRF_TRUSTED_ORIGINS") + _PRODUCTION_ORIGINS)
)

# Session — expire early in production
SESSION_COOKIE_AGE = 86400  # 24 hours
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
