from unittest.mock import Mock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from rest_framework.test import APIClient

from medimind.fields import ENCRYPTED_PREFIX


class AuthenticationSecurityTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_login_page_keeps_traditional_login_when_social_is_unconfigured(self):
        response = self.client.get(reverse("login_page"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="loginForm"')
        self.assertNotContains(response, "Continue with Google")
        self.assertNotContains(response, "Continue with Apple")

    @override_settings(
        SOCIAL_LOGIN_GOOGLE_ENABLED=True,
        SOCIAL_LOGIN_APPLE_ENABLED=True,
        SOCIALACCOUNT_PROVIDERS={
            "google": {"APPS": [{"client_id": "google-id", "secret": "google-secret", "key": ""}]},
            "apple": {
                "APPS": [{
                    "client_id": "com.example.web",
                    "secret": "APPLEKEY",
                    "key": "APPLETEAM",
                    "settings": {"certificate_key": "test-key"},
                }]
            },
        },
    )
    def test_social_login_buttons_use_post_endpoints_when_enabled(self):
        response = self.client.get(reverse("login_page"))
        self.assertContains(response, "Continue with Google")
        self.assertContains(response, "Continue with Apple")
        self.assertContains(response, reverse("google_login"))
        self.assertContains(response, reverse("apple_login"))

    def test_honeypot_blocks_registration(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "bot@example.com",
                "email": "bot@example.com",
                "password": "StrongPassword123!",
                "confirm_password": "StrongPassword123!",
                "website": "https://spam.invalid",
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(get_user_model().objects.filter(username="bot@example.com").exists())

    @override_settings(TURNSTILE_ENABLED=True, TURNSTILE_SECRET_KEY="server-secret")
    @patch("authentication.bot_protection.requests.post")
    def test_turnstile_is_verified_server_side(self, post):
        upstream = Mock()
        upstream.raise_for_status.return_value = None
        upstream.json.return_value = {"success": False}
        post.return_value = upstream
        response = self.client.post(
            reverse("login"),
            {"username": "person", "password": "password", "turnstile_token": "bad-token"},
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["data"]["secret"], "server-secret")

    def test_ai_route_auth_check_requires_login(self):
        response = self.client.get(reverse("check_ai_access"))
        self.assertEqual(response.status_code, 401)

    def test_ai_route_rejects_cross_origin_session_post(self):
        user = get_user_model().objects.create_user("patient", password="StrongPassword123!")
        self.client.force_login(user)
        response = self.client.get(
            reverse("check_ai_access"),
            HTTP_X_ORIGINAL_METHOD="POST",
            HTTP_X_ORIGINAL_ORIGIN="https://evil.example",
        )
        self.assertEqual(response.status_code, 403)

    def test_ai_auth_check_returns_server_side_identity_header(self):
        user = get_user_model().objects.create_user("social-user", password="StrongPassword123!")
        self.client.force_login(user)
        response = self.client.get(reverse("check_ai_access"))
        self.assertEqual(response.status_code, 204)
        self.assertTrue(response["Authorization"].startswith("Bearer "))

    def test_phone_number_is_encrypted_in_database(self):
        user = get_user_model().objects.create_user(
            "encrypted-patient", password="StrongPassword123!", phone_number="+1 555 0100"
        )
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT phone_number FROM authentication_user WHERE id = %s", [user.pk])
            stored = cursor.fetchone()[0]
        self.assertTrue(stored.startswith(ENCRYPTED_PREFIX))
        user.refresh_from_db()
        self.assertEqual(user.phone_number, "+1 555 0100")

    @property
    def connection(self):
        from django.db import connection

        return connection
