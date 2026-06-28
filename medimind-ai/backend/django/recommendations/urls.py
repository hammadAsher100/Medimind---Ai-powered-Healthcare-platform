from django.urls import path

from .views import (
    ExplainabilityView,
    KnowledgeBaseUploadView,
    PredictionCreateView,
    PredictionListView,
    RecommendationListView,
    explainability_page,
)

urlpatterns = [
    path("predictions/", PredictionListView.as_view(), name="prediction-list"),
    path("predictions/predict/<str:disease>/", PredictionCreateView.as_view(), name="prediction-create"),
    path("recommendations/", RecommendationListView.as_view(), name="recommendation-list"),
    path("knowledge-base/upload/", KnowledgeBaseUploadView.as_view(), name="knowledge-upload"),
    path("explainability/<int:prediction_id>/", ExplainabilityView.as_view(), name="explainability"),
    path("explainability/<int:prediction_id>/chart/", explainability_page, name="explainability-page"),
]
