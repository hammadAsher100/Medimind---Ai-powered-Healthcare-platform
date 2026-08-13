from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase

from reports.models import MedicalReport

from .models import Medication, MedicationExtractionRecord, PatientMedication
from .serializers import MedicationEventSerializer, MedicationExtractionRecordSerializer, PatientMedicationSerializer


class MedicationOwnershipTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.owner = User.objects.create_user("owner", password="StrongPassword123!")
        self.other = User.objects.create_user("other", password="StrongPassword123!")
        self.request = RequestFactory().post("/")
        self.request.user = self.owner

    def test_user_field_is_not_exposed_for_tampering(self):
        self.assertNotIn("user", PatientMedicationSerializer().fields)

    def test_medication_event_rejects_another_users_record(self):
        medication = Medication.objects.create(name="Example")
        other_record = PatientMedication.objects.create(user=self.other, medication=medication)
        serializer = MedicationEventSerializer(
            data={"patient_medication": other_record.pk, "event_type": "dose_taken"},
            context={"request": self.request},
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("patient_medication", serializer.errors)

    def test_extraction_rejects_another_users_report(self):
        report = MedicalReport.objects.create(user=self.other, file="reports/example.pdf")
        serializer = MedicationExtractionRecordSerializer(
            data={"source_report": report.pk, "extraction_method": "manual"},
            context={"request": self.request},
        )
        self.assertFalse(serializer.is_valid())
        self.assertIn("source_report", serializer.errors)
