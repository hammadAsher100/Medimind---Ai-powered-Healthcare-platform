from rest_framework import generics

from .models import TimelineEvent
from .serializers import TimelineEventSerializer


class TimelineListView(generics.ListAPIView):
    serializer_class = TimelineEventSerializer

    def get_queryset(self):
        return TimelineEvent.objects.filter(user=self.request.user).order_by("-created_at")
