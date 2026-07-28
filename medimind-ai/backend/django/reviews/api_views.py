"""DRF API views for Reviews and Audit."""

from rest_framework import permissions, status, viewsets
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .models import AuditEvent, ModelFeedback, ReviewDecision
from .serializers import (
    AuditEventSerializer,
    ModelFeedbackSerializer,
    ReviewDecisionSerializer,
)


class ReviewDecisionViewSet(viewsets.ModelViewSet):
    serializer_class = ReviewDecisionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ReviewDecision.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class ModelFeedbackViewSet(viewsets.ModelViewSet):
    serializer_class = ModelFeedbackSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ModelFeedback.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class AuditEventViewSet(viewsets.ModelViewSet):
    serializer_class = AuditEventSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AuditEvent.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def record_audit_event(request):
    """Record an audit event."""
    data = request.data.copy()
    data["user"] = request.user.pk
    serializer = AuditEventSerializer(data=data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
