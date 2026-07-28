"""DRF API views for Clinical intelligence data persistence."""

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .models import (
    ClinicalObservation,
    DataConflict,
    DiagnosticReportRecord,
    ObservationTrend,
    PatientStateSnapshot,
)
from .serializers import (
    ClinicalObservationSerializer,
    DataConflictSerializer,
    DiagnosticReportRecordSerializer,
    ObservationTrendSerializer,
    PatientStateSnapshotSerializer,
)


class ClinicalObservationViewSet(viewsets.ModelViewSet):
    serializer_class = ClinicalObservationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ClinicalObservation.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class DataConflictViewSet(viewsets.ModelViewSet):
    serializer_class = DataConflictSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return DataConflict.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class DiagnosticReportRecordViewSet(viewsets.ModelViewSet):
    serializer_class = DiagnosticReportRecordSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return DiagnosticReportRecord.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class PatientStateSnapshotViewSet(viewsets.ModelViewSet):
    serializer_class = PatientStateSnapshotSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = PatientStateSnapshot.objects.filter(user=self.request.user)
        if self.request.query_params.get("current") == "true":
            qs = qs.filter(is_current=True)
        return qs

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def bulk_create_observations(request):
    """Bulk create clinical observations from extracted report data."""
    observations = request.data.get("observations", [])
    if not observations:
        return Response(
            {"error": "No observations provided"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    created = []
    for obs_data in observations:
        obs_data["user"] = request.user.pk
        serializer = ClinicalObservationSerializer(data=obs_data)
        if serializer.is_valid():
            serializer.save()
            created.append(serializer.data)

    return Response(
        {
            "created_count": len(created),
            "observations": created,
        },
        status=status.HTTP_201_CREATED,
    )
