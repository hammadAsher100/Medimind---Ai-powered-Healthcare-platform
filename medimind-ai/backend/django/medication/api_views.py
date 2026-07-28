"""DRF API views for Medication safety data persistence."""

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .models import (
    Medication,
    MedicationAllergy,
    MedicationEvent,
    MedicationExtractionRecord,
    MedicationSafetyAlert,
    PatientMedication,
)
from .serializers import (
    MedicationAllergySerializer,
    MedicationEventSerializer,
    MedicationExtractionRecordSerializer,
    MedicationSafetyAlertSerializer,
    MedicationSerializer,
    PatientMedicationSerializer,
)


class MedicationViewSet(viewsets.ModelViewSet):
    serializer_class = MedicationSerializer
    permission_classes = [permissions.IsAuthenticated]
    queryset = Medication.objects.all()


class PatientMedicationViewSet(viewsets.ModelViewSet):
    serializer_class = PatientMedicationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return PatientMedication.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MedicationEventViewSet(viewsets.ModelViewSet):
    serializer_class = MedicationEventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MedicationEvent.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MedicationSafetyAlertViewSet(viewsets.ModelViewSet):
    serializer_class = MedicationSafetyAlertSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MedicationSafetyAlert.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MedicationAllergyViewSet(viewsets.ModelViewSet):
    serializer_class = MedicationAllergySerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MedicationAllergy.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MedicationExtractionRecordViewSet(viewsets.ModelViewSet):
    serializer_class = MedicationExtractionRecordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MedicationExtractionRecord.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def bulk_save_alerts(request):
    """Bulk save medication safety alerts from AI analysis."""
    alerts = request.data.get("alerts", [])
    created = []
    for alert_data in alerts:
        alert_data["user"] = request.user.pk
        serializer = MedicationSafetyAlertSerializer(data=alert_data)
        if serializer.is_valid():
            serializer.save()
            created.append(serializer.data)

    return Response(
        {"created_count": len(created), "alerts": created},
        status=status.HTTP_201_CREATED,
    )
