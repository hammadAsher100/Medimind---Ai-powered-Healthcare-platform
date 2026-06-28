import requests
from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from timeline.services import create_timeline_event

from .models import HealthScore
from .serializers import HealthScoreSerializer


class HealthScoreView(APIView):
    def get(self, request):
        latest = HealthScore.objects.filter(user=request.user).first()
        if not latest:
            return Response({"detail": "No health score has been calculated yet."}, status=status.HTTP_404_NOT_FOUND)
        return Response(HealthScoreSerializer(latest).data)

    def post(self, request):
        try:
            response = requests.post(
                f"{settings.FASTAPI_URL}/calculate-health-score",
                json=request.data,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            return Response({"detail": f"Health score service unavailable: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

        score = HealthScore.objects.create(
            user=request.user,
            score=result["score"],
            bmi=request.data.get("bmi"),
            sugar_level=request.data.get("glucose"),
            cholesterol=request.data.get("ldl") or request.data.get("cholesterol"),
            blood_pressure_systolic=request.data.get("systolic"),
            blood_pressure_diastolic=request.data.get("diastolic"),
            lifestyle_score=result.get("lifestyle_score", 0),
            strengths=result.get("strengths", []),
            improvements=result.get("needs_improvement", []),
        )
        create_timeline_event(request.user, "score_calculated", "Health score calculated", f"New score: {score.score}", result)
        return Response(HealthScoreSerializer(score).data, status=status.HTTP_201_CREATED)


class HealthScoreHistoryView(APIView):
    def get(self, request):
        scores = HealthScore.objects.filter(user=request.user).order_by("-created_at")[:12]
        return Response(HealthScoreSerializer(reversed(list(scores)), many=True).data)
