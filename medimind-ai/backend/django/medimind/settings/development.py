import os

from .base import *  # noqa: F401,F403

DEBUG = True
CORS_ALLOW_ALL_ORIGINS = True

if os.environ.get("USE_POSTGRES", "False").lower() != "true":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8001")
