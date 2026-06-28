from django.urls import path

from .views import HealthScoreHistoryView, HealthScoreView

urlpatterns = [
    path("", HealthScoreView.as_view(), name="health-score"),
    path("history/", HealthScoreHistoryView.as_view(), name="health-score-history"),
]
