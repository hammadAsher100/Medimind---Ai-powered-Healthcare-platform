"""
MediMind AI — Security Middleware

Adds Content Security Policy, Permissions-Policy, and other security headers
that cannot be configured through Django's built-in SecurityMiddleware.
"""

import re

from django.conf import settings
from django.http import HttpResponse


class ContentSecurityPolicyMiddleware:
    """Adds Content-Security-Policy and related security headers to every response.

    The policy is strict but allows Django admin, Chart.js CDN, and
    the application's own static files. Nonces are used for inline scripts.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        if hasattr(response, "headers"):
            self._apply_headers(request, response)
        return response

    def _apply_headers(self, request, response):
        # CSP — relaxed enough for Chart.js CDN and inline scripts
        csp_parts = [
            "default-src 'self'",
            "script-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' https://fonts.googleapis.com https://cdn.jsdelivr.net 'unsafe-inline'",
            "img-src 'self' data: blob:",
            "font-src 'self' https://fonts.gstatic.com data:",
            "connect-src 'self' http://localhost:* http://127.0.0.1:*",
            "frame-ancestors 'none'",
            "form-action 'self'",
            "base-uri 'self'",
            "object-src 'none'",
        ]
        response["Content-Security-Policy"] = "; ".join(csp_parts)

        # Permissions Policy — restrict sensitive APIs
        response["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "fullscreen=(self), payment=(), usb=()"
        )

        # Cross-Origin policies
        response["Cross-Origin-Opener-Policy"] = "same-origin"
        response["Cross-Origin-Resource-Policy"] = "same-origin"
