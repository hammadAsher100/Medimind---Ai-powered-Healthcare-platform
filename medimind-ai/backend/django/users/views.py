from rest_framework import viewsets

from .models import Allergy, FamilyHistory, MedicalProfile
from .serializers import AllergySerializer, FamilyHistorySerializer, MedicalProfileSerializer


class OwnedModelMixin:
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class MedicalProfileViewSet(OwnedModelMixin, viewsets.ModelViewSet):
    serializer_class = MedicalProfileSerializer

    def get_queryset(self):
        return MedicalProfile.objects.filter(user=self.request.user)


class AllergyViewSet(OwnedModelMixin, viewsets.ModelViewSet):
    serializer_class = AllergySerializer

    def get_queryset(self):
        return Allergy.objects.filter(user=self.request.user).order_by("allergen")


class FamilyHistoryViewSet(OwnedModelMixin, viewsets.ModelViewSet):
    serializer_class = FamilyHistorySerializer

    def get_queryset(self):
        return FamilyHistory.objects.filter(user=self.request.user).order_by("condition")
