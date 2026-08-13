import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "unsafe-development-secret")
FIELD_ENCRYPTION_KEYS = [
    key.strip() for key in os.environ.get("FIELD_ENCRYPTION_KEYS", "").split(",") if key.strip()
]
DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"
ALLOWED_HOSTS = [host.strip() for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,django").split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "allauth.socialaccount.providers.apple",
    "rest_framework",
    "rest_framework_simplejwt",
    "rest_framework_simplejwt.token_blacklist",
    "corsheaders",
    "authentication",
    "users",
    "dashboard",
    "reports",
    "health_score",
    "timeline",
    "recommendations",
    # Clinical intelligence
    "clinical",
    "medication",
    "reviews",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "medimind.activity.ActivityTrackingMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "medimind.security_middleware.ContentSecurityPolicyMiddleware",
]

ROOT_URLCONF = "medimind.urls"
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/login/"

AUTHENTICATION_BACKENDS = [
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
]

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": [
            "django.template.context_processors.debug",
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
        ]},
    }
]

WSGI_APPLICATION = "medimind.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("POSTGRES_DB", "medimind"),
        "USER": os.environ.get("POSTGRES_USER", "medimind"),
        "PASSWORD": os.environ.get("POSTGRES_PASSWORD", "medimind_password"),
        "HOST": os.environ.get("POSTGRES_HOST", os.environ.get("DB_HOST", "postgres")),
        "PORT": os.environ.get("POSTGRES_PORT", os.environ.get("DB_PORT", "5432")),
    }
}

# ── Password Validation ──────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 10}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── Password Reset ───────────────────────────────────────
PASSWORD_RESET_TIMEOUT = 1800  # 30 minutes

# ── Session Security ─────────────────────────────────────
SESSION_COOKIE_AGE = 604800  # 7 days
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_EXPIRE_AT_BROWSER_CLOSE = False

# The CSRF token is not an authentication secret. Keeping it readable allows
# same-origin JavaScript to protect session-authenticated API requests.
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = "Lax"
CSRF_USE_SESSIONS = os.environ.get("CSRF_USE_SESSIONS", "False").lower() == "true"

# ── Security Headers ─────────────────────────────────────
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "same-origin"

# ── Upload Limits ────────────────────────────────────────
DATA_UPLOAD_MAX_MEMORY_SIZE = 25 * 1024 * 1024  # Allows a 20 MB file plus multipart overhead
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_621_440  # Stream files larger than 2.5 MB to a temporary file
DATA_UPLOAD_MAX_NUMBER_FIELDS = 100

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
_STATIC_DIR = BASE_DIR / "static"
STATICFILES_DIRS = [_STATIC_DIR] if _STATIC_DIR.is_dir() else []
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

AUTH_USER_MODEL = "authentication.User"
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://fastapi:8001")

# Social login credentials are read only from the process environment. Provider
# buttons remain hidden until a complete credential set is configured.
GOOGLE_OAUTH_CLIENT_ID = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
GOOGLE_OAUTH_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
APPLE_OAUTH_CLIENT_ID = os.environ.get("APPLE_OAUTH_CLIENT_ID", "").strip()
APPLE_OAUTH_KEY_ID = os.environ.get("APPLE_OAUTH_KEY_ID", "").strip()
APPLE_OAUTH_TEAM_ID = os.environ.get("APPLE_OAUTH_TEAM_ID", "").strip()
APPLE_OAUTH_PRIVATE_KEY = os.environ.get("APPLE_OAUTH_PRIVATE_KEY", "").replace("\\n", "\n").strip()

SOCIAL_LOGIN_GOOGLE_ENABLED = bool(GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET)
SOCIAL_LOGIN_APPLE_ENABLED = bool(
    APPLE_OAUTH_CLIENT_ID
    and APPLE_OAUTH_KEY_ID
    and APPLE_OAUTH_TEAM_ID
    and APPLE_OAUTH_PRIVATE_KEY
)
SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "SCOPE": ["profile", "email"],
        "AUTH_PARAMS": {"access_type": "online"},
        "OAUTH_PKCE_ENABLED": True,
        "EMAIL_AUTHENTICATION": True,
        "EMAIL_AUTHENTICATION_AUTO_CONNECT": True,
        **(
            {
                "APPS": [{
                    "client_id": GOOGLE_OAUTH_CLIENT_ID,
                    "secret": GOOGLE_OAUTH_CLIENT_SECRET,
                    "key": "",
                }]
            }
            if SOCIAL_LOGIN_GOOGLE_ENABLED
            else {}
        ),
    },
    "apple": {
        "EMAIL_AUTHENTICATION": True,
        "EMAIL_AUTHENTICATION_AUTO_CONNECT": True,
        **(
            {
                "APPS": [{
                    "client_id": APPLE_OAUTH_CLIENT_ID,
                    "secret": APPLE_OAUTH_KEY_ID,
                    "key": APPLE_OAUTH_TEAM_ID,
                    "settings": {"certificate_key": APPLE_OAUTH_PRIVATE_KEY},
                }]
            }
            if SOCIAL_LOGIN_APPLE_ENABLED
            else {}
        ),
    },
}
SOCIALACCOUNT_AUTO_SIGNUP = True
SOCIALACCOUNT_LOGIN_ON_GET = False
SOCIALACCOUNT_STORE_TOKENS = False
SOCIALACCOUNT_REQUESTS_TIMEOUT = 10
ACCOUNT_EMAIL_VERIFICATION = "none"
ACCOUNT_LOGIN_METHODS = {"username", "email"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*", "password1*", "password2*"]

# Optional Cloudflare Turnstile protection for password login/registration.
# When enabled, verification fails closed and the secret never reaches a page.
TURNSTILE_ENABLED = os.environ.get("TURNSTILE_ENABLED", "False").strip().lower() == "true"
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY", "").strip()
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
if TURNSTILE_ENABLED and not (TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY):
    raise RuntimeError("TURNSTILE_ENABLED requires TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY")

# ── CORS ─────────────────────────────────────────────────
CORS_ALLOW_ALL_ORIGINS = os.environ.get("CORS_ALLOW_ALL_ORIGINS", "False").lower() == "true"
CORS_ALLOWED_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:18000").split(",") if origin.strip()]
CSRF_TRUSTED_ORIGINS = [origin.strip() for origin in os.environ.get("CSRF_TRUSTED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,http://localhost:18000").split(",") if origin.strip()]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"]
CORS_ALLOW_HEADERS = ["accept", "authorization", "content-type", "x-csrftoken", "x-requested-with"]

# ── REST Framework ───────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework_simplejwt.authentication.JWTAuthentication",
        "rest_framework.authentication.SessionAuthentication",
    ),
    "DEFAULT_PERMISSION_CLASSES": ("rest_framework.permissions.IsAuthenticated",),
    "DEFAULT_THROTTLE_CLASSES": ["rest_framework.throttling.AnonRateThrottle", "rest_framework.throttling.UserRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"anon": "20/hour", "user": "200/hour"},
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 50,
    "NUM_PROXIES": 1,
    "DEFAULT_RENDERER_CLASSES": ("rest_framework.renderers.JSONRenderer",),
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=1),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}
