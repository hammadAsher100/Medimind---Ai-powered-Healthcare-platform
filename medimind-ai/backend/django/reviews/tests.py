"""Tests for reviews models."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import AuditEvent, ModelFeedback, ReviewDecision

User = get_user_model()


class ReviewDecisionTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="reviewtest", password="test")

    def test_decision_accepted(self):
        d = ReviewDecision.objects.create(
            user=self.user,
            recommendation_type="diabetes_risk",
            clinician_decision="accepted",
            ai_summary="High risk of diabetes.",
            clinician_notes="Agree with findings.",
        )
        self.assertEqual(d.get_clinician_decision_display(), "Accepted")

    def test_decision_rejected(self):
        d = ReviewDecision.objects.create(
            user=self.user,
            recommendation_type="medication_suggestion",
            clinician_decision="rejected",
        )
        self.assertEqual(d.get_clinician_decision_display(), "Rejected")


class ModelFeedbackTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="fbtest", password="test")

    def test_feedback_types(self):
        fb = ModelFeedback.objects.create(
            user=self.user,
            model_name="pneumonia_xray",
            feedback_type="correct",
            original_output="PNEUMONIA",
        )
        self.assertEqual(fb.get_feedback_type_display(), "Correct")

    def test_feedback_hallucination(self):
        fb = ModelFeedback.objects.create(
            user=self.user,
            model_name="chat_assistant",
            feedback_type="hallucination",
            corrected_output="Evidence not found in patient data.",
        )
        self.assertEqual(fb.get_feedback_type_display(), "Hallucination")


class AuditEventTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="audittest", password="test")

    def test_audit_event(self):
        event = AuditEvent.objects.create(
            user=self.user,
            event_type="prediction_made",
            source="test_suite",
            event_detail={"disease": "diabetes", "risk": 45.2},
        )
        self.assertEqual(event.get_event_type_display(), "Prediction Made")
        self.assertEqual(event.event_detail["risk"], 45.2)

    def test_emergency_event(self):
        event = AuditEvent.objects.create(
            user=self.user, event_type="emergency_detected", source="system"
        )
        self.assertEqual(event.get_event_type_display(), "Emergency Detected")
