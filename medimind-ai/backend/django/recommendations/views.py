import requests
from django.conf import settings
from django.shortcuts import get_object_or_404, render
from rest_framework import generics, permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from timeline.services import create_timeline_event

from .models import KnowledgeDocument, Prediction, Recommendation
from .serializers import KnowledgeDocumentSerializer, PredictionSerializer, RecommendationSerializer


class PredictionCreateView(APIView):
    def post(self, request, disease):
        try:
            response = requests.post(f"{settings.FASTAPI_URL}/predict/{disease}", json=request.data, timeout=90)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            return Response({"detail": f"Prediction service unavailable: {exc}"}, status=status.HTTP_502_BAD_GATEWAY)

        prediction = Prediction.objects.create(
            user=request.user,
            disease=result["disease"],
            input_data=request.data,
            risk_percentage=result["risk_percentage"],
            risk_level=result["risk_level"],
            prediction=result["prediction"],
            shap_explanation=result["shap_explanation"],
            ai_recommendation=result.get("ai_recommendation", ""),
        )
        create_timeline_event(
            request.user,
            "prediction_made",
            f"{result['disease'].title()} risk prediction",
            f"Risk level: {result['risk_level']} ({result['risk_percentage']:.1f}%)",
            {"prediction_id": prediction.id, "disease": disease},
        )
        return Response(PredictionSerializer(prediction).data, status=status.HTTP_201_CREATED)


class PredictionListView(generics.ListAPIView):
    serializer_class = PredictionSerializer

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


class RecommendationListView(generics.ListAPIView):
    serializer_class = RecommendationSerializer

    def get_queryset(self):
        return Recommendation.objects.filter(user=self.request.user)


class KnowledgeBaseUploadView(generics.CreateAPIView):
    permission_classes = [permissions.IsAdminUser]
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = KnowledgeDocumentSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        document = serializer.save(uploaded_by=request.user)
        try:
            document.file.open("rb")
            files = {"file": (document.file.name, document.file.file, "application/pdf")}
            response = requests.post(
                f"{settings.FASTAPI_URL}/index-document",
                data={"title": document.title, "source": document.source, "document_id": document.id},
                files=files,
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            document.metadata = {"indexing_error": str(exc)}
            document.save(update_fields=["metadata"])
            return Response(KnowledgeDocumentSerializer(document).data, status=status.HTTP_202_ACCEPTED)
        finally:
            document.file.close()

        document.indexed_chunks = result.get("indexed_chunks", 0)
        document.metadata = result
        document.save(update_fields=["indexed_chunks", "metadata"])
        create_timeline_event(request.user, "knowledge_indexed", "Knowledge document indexed", document.title, result)
        return Response(KnowledgeDocumentSerializer(document).data, status=status.HTTP_201_CREATED)


class ExplainabilityView(APIView):
    def get(self, request, prediction_id):
        prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
        top_factors = prediction.shap_explanation.get("top_factors", [])
        return Response(
            {
                "disease": prediction.disease,
                "risk_percentage": prediction.risk_percentage,
                "risk_level": prediction.risk_level,
                "chart": {
                    "labels": [factor.get("feature") for factor in top_factors],
                    "values": [factor.get("contribution") for factor in top_factors],
                    "colors": ["#dc2626" if factor.get("direction") == "increases_risk" else "#16a34a" for factor in top_factors],
                },
                "explanation": prediction.shap_explanation.get("explanation_text", ""),
                "ai_recommendation": prediction.ai_recommendation,
                "rag_context": prediction.shap_explanation.get("rag_context", []),
            }
        )


def explainability_page(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    top_factors = prediction.shap_explanation.get("top_factors", [])
    return render(
        request,
        "reports/explainability.html",
        {
            "prediction": prediction,
            "labels": [factor.get("feature") for factor in top_factors],
            "values": [factor.get("contribution") for factor in top_factors],
            "colors": ["#dc2626" if factor.get("direction") == "increases_risk" else "#16a34a" for factor in top_factors],
            "rag_context": prediction.shap_explanation.get("rag_context", []),
        },
    )
