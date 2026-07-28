"""Tests for clinical models."""

from django.test import TestCase
from django.contrib.auth import get_user_model

from .models import (
    ClinicalObservation,
    DataConflict,
    DiagnosticReportRecord,
    ObservationTrend,
    PatientStateSnapshot,
)

User = get_user_model()


class ClinicalObservationModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser", password="testpass123"
        )

    def test_create_observation(self):
        obs = ClinicalObservation.objects.create(
            user=self.user,
            test_name="Glucose",
            standardised_name="glucose",
            original_value="95",
            numeric_value=95.0,
            original_unit="mg/dL",
            abnormality_status="normal",
        )
        self.assertEqual(str(obs), "Glucose: 95 mg/dL")
        self.assertEqual(obs.get_abnormality_status_display(), "Normal")

    def test_critically_high(self):
        obs = ClinicalObservation.objects.create(
            user=self.user,
            test_name="Potassium",
            numeric_value=6.5,
            original_value="6.5",
            abnormality_status="critically_high",
        )
        self.assertEqual(obs.get_abnormality_status_display(), "Critically High")

    def test_verification_status_default(self):
        obs = ClinicalObservation.objects.create(
            user=self.user, test_name="Test", numeric_value=1.0, original_value="1"
        )
        self.assertEqual(obs.verification_status, "unverified")


class DataConflictTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="ctest", password="test")

    def test_conflict_creation(self):
        con = DataConflict.objects.create(
            user=self.user,
            conflict_type="value_discrepancy",
            severity="warning",
            first_source_label="A",
            second_source_label="B",
        )
        self.assertTrue(str(con).startswith("Value Discrepancy"))

    def test_default_unresolved(self):
        con = DataConflict.objects.create(
            user=self.user,
            conflict_type="other",
            first_source_label="A",
            second_source_label="B",
        )
        self.assertEqual(con.resolution_status, "unresolved")


class PatientStateSnapshotTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="ptest", password="test")

    def test_snapshot_priority(self):
        s = PatientStateSnapshot.objects.create(
            user=self.user, priority_level="urgent", is_current=True
        )
        self.assertEqual(s.get_priority_level_display(), "Urgent")
        self.assertTrue(s.is_current)
