"""URL configuration for medication app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import api_views as views

router = DefaultRouter()
router.register(r"medications", views.MedicationViewSet, basename="medication")
router.register(
    r"patient-medications",
    views.PatientMedicationViewSet,
    basename="patient-medication",
)
router.register(r"events", views.MedicationEventViewSet, basename="medication-event")
router.register(
    r"alerts", views.MedicationSafetyAlertViewSet, basename="medication-alert"
)
router.register(
    r"allergies", views.MedicationAllergyViewSet, basename="medication-allergy"
)
router.register(
    r"extractions",
    views.MedicationExtractionRecordViewSet,
    basename="medication-extraction",
)

urlpatterns = [
    path("", include(router.urls)),
    path("alerts/bulk/", views.bulk_save_alerts, name="bulk-save-alerts"),
]
