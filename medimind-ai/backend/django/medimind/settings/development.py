import os

from .base import *  # noqa: F401,F403

DEBUG = True

if os.environ.get("USE_POSTGRES", "False").lower() != "true":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# In development, allow broader CORS for the local frontend
CORS_ALLOW_ALL_ORIGINS = os.environ.get("CORS_ALLOW_ALL_ORIGINS", "True").lower() == "true"
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()]

# Allow HTTP in development
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8001")

# Allow Browsable API in development
REST_FRAMEWORK["DEFAULT_RENDERER_CLASSES"] = (
    "rest_framework.renderers.JSONRenderer",
    "rest_framework.renderers.BrowsableAPIRenderer",
)
