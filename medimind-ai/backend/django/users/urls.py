from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import AllergyViewSet, FamilyHistoryViewSet, MedicalProfileViewSet

router = DefaultRouter()
router.register("medical-profile", MedicalProfileViewSet, basename="medical-profile")
router.register("allergies", AllergyViewSet, basename="allergies")
router.register("family-history", FamilyHistoryViewSet, basename="family-history")

urlpatterns = [path("", include(router.urls))]
