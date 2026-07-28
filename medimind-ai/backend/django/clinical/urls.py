"""URL configuration for clinical app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import api_views

router = DefaultRouter()
router.register(
    r"observations", api_views.ClinicalObservationViewSet, basename="observation"
)
router.register(
    r"conflicts", api_views.DataConflictViewSet, basename="conflict"
)
router.register(
    r"diagnostic-reports",
    api_views.DiagnosticReportRecordViewSet,
    basename="diagnostic-report",
)
router.register(
    r"state", api_views.PatientStateSnapshotViewSet, basename="patient-state"
)

urlpatterns = [
    path("", include(router.urls)),
    path(
        "observations/bulk/",
        api_views.bulk_create_observations,
        name="bulk-create-observations",
    ),
]
