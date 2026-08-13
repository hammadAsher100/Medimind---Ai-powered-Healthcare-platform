import hashlib
from urllib.parse import urlparse

from django.conf import settings
from django.contrib.auth import login as django_login
from django.contrib.auth import logout as django_logout
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.throttling import AnonRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.tokens import AccessToken

from .serializers import LoginSerializer, RegisterSerializer, UserProfileSerializer
from .bot_protection import verify_human


class LoginThrottle(AnonRateThrottle):
    rate = "10/hour"

    def get_cache_key(self, request, view):
        identity = str(request.data.get("username", "")).strip().casefold()
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        return self.cache_format % {"scope": self.scope, "ident": f"{self.get_ident(request)}:{digest}"}


class LoginIPThrottle(AnonRateThrottle):
    """Bound password spraying across many account names from one client."""

    rate = "30/hour"


class RegisterThrottle(AnonRateThrottle):
    rate = "5/hour"


def _session_request(request):
    return getattr(request, "_request", request)


def _login_browser_session(request, user):
    django_request = _session_request(request)
    django_login(django_request, user)
    django_request.session.save()


class RegisterView(generics.CreateAPIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    serializer_class = RegisterSerializer
    throttle_classes = [RegisterThrottle]

    def create(self, request, *args, **kwargs):
        verify_human(request)
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        _login_browser_session(request, user)
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "user": UserProfileSerializer(user).data,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            },
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    throttle_classes = [LoginIPThrottle, LoginThrottle]

    def post(self, request):
        verify_human(request)
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        _login_browser_session(request, user)
        refresh = RefreshToken.for_user(user)
        return Response(
            {
                "user": UserProfileSerializer(user).data,
                "access": str(refresh.access_token),
                "refresh": str(refresh),
            }
        )


class LogoutView(APIView):
    def post(self, request):
        refresh_token = request.data.get("refresh")
        if not refresh_token:
            return Response({"detail": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            RefreshToken(refresh_token).blacklist()
        except Exception:
            return Response({"detail": "Invalid refresh token."}, status=status.HTTP_400_BAD_REQUEST)
        django_logout(_session_request(request))
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer

    def get_object(self):
        return self.request.user

    def post(self, request, *args, **kwargs):
        """Accept POST as partial update for browser convenience."""
        return self.partial_update(request, *args, **kwargs)


class AIAuthCheckView(APIView):
    """Nginx auth_request target protecting the browser-facing FastAPI route."""

    def get(self, request):
        original_method = request.headers.get("X-Original-Method", "GET").upper()
        original_uri = request.headers.get("X-Original-URI", "")
        if original_uri.startswith("/ai/index-document") and not request.user.is_staff:
            return Response(status=status.HTTP_403_FORBIDDEN)
        if request.auth is None and original_method not in {"GET", "HEAD", "OPTIONS"}:
            candidate = request.headers.get("X-Original-Origin") or request.headers.get("X-Original-Referer")
            if not candidate:
                return Response(status=status.HTTP_403_FORBIDDEN)
            parsed = urlparse(candidate)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            allowed = set(settings.CSRF_TRUSTED_ORIGINS)
            if origin not in allowed:
                return Response(status=status.HTTP_403_FORBIDDEN)
        response = Response(status=status.HTTP_204_NO_CONTENT)
        # FastAPI can use this short-lived identity when it must call an
        # ownership-checked Django endpoint. Social sessions never expose the
        # token to browser JavaScript.
        response["Authorization"] = f"Bearer {AccessToken.for_user(request.user)}"
        return response
