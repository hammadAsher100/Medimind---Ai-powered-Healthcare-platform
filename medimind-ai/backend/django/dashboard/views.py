from rest_framework.response import Response
from rest_framework.views import APIView

from health_score.models import HealthScore
from health_score.serializers import HealthScoreSerializer
from recommendations.models import Prediction
from recommendations.serializers import PredictionSerializer
from reports.models import MedicalReport
from reports.serializers import MedicalReportSerializer
from timeline.models import TimelineEvent
from timeline.serializers import TimelineEventSerializer


class DashboardView(APIView):
    def get(self, request):
        latest_score = HealthScore.objects.filter(user=request.user).first()
        medical_profile = getattr(request.user, "medical_profile", None)
        payload = {
            "latest_health_score": HealthScoreSerializer(latest_score).data if latest_score else None,
            "recent_timeline": TimelineEventSerializer(
                TimelineEvent.objects.filter(user=request.user).order_by("-created_at")[:5],
                many=True,
            ).data,
            "recent_reports": MedicalReportSerializer(
                MedicalReport.objects.filter(user=request.user).order_by("-uploaded_at")[:3],
                many=True,
            ).data,
            "bmi": medical_profile.bmi if medical_profile else None,
            "latest_predictions": PredictionSerializer(
                Prediction.objects.filter(user=request.user).order_by("-created_at")[:4],
                many=True,
            ).data,
        }
        return Response(payload)
