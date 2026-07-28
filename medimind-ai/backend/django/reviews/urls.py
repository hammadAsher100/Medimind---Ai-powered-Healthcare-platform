"""URL configuration for reviews app."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import api_views

router = DefaultRouter()
router.register(
    r"decisions", api_views.ReviewDecisionViewSet, basename="review-decision"
)
router.register(
    r"feedback", api_views.ModelFeedbackViewSet, basename="model-feedback"
)
router.register(
    r"audit", api_views.AuditEventViewSet, basename="audit-event"
)

urlpatterns = [
    path("", include(router.urls)),
    path("audit-log/", api_views.record_audit_event, name="record-audit-event"),
]
