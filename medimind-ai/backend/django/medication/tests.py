"""Tests for medication models."""

from django.contrib.auth import get_user_model
from django.test import TestCase

from .models import Medication, MedicationAllergy, MedicationSafetyAlert, PatientMedication

User = get_user_model()


class MedicationTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="medtest", password="test")
        self.med = Medication.objects.create(
            name="Atorvastatin",
            generic_name="Atorvastatin Calcium",
            rxnorm_code="993489",
            drug_class="statin",
            common_side_effects=["muscle pain", "headache"],
            contraindications=["liver disease"],
        )

    def test_medication_creation(self):
        self.assertEqual(self.med.name, "Atorvastatin")
        self.assertEqual(self.med.rxnorm_code, "993489")

    def test_patient_medication(self):
        pm = PatientMedication.objects.create(
            user=self.user,
            medication=self.med,
            dosage="20 mg",
            frequency="once daily",
            status="active",
        )
        self.assertEqual(str(pm), "Atorvastatin (active)")
        self.assertEqual(pm.get_status_display(), "Active")

    def test_medication_event(self):
        pm = PatientMedication.objects.create(
            user=self.user, medication=self.med, status="active"
        )
        event = pm.events.create(
            user=self.user, patient_medication=pm, event_type="dose_taken"
        )
        self.assertIn("Dose Taken", str(event))


class MedicationAlertTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alerttest", password="test")

    def test_safety_alert(self):
        alert = MedicationSafetyAlert.objects.create(
            user=self.user,
            alert_type="drug_interaction",
            severity="critical",
            title="Warfarin + Aspirin interaction",
            description="Increased bleeding risk",
        )
        self.assertEqual(alert.get_severity_display(), "Critical")
        self.assertFalse(alert.is_synthetic_data)

    def test_allergy(self):
        allergy = MedicationAllergy.objects.create(
            user=self.user,
            allergen="Penicillin",
            severity="severe",
            reaction_type="Anaphylaxis",
        )
        self.assertEqual(allergy.get_severity_display(), "Severe")
