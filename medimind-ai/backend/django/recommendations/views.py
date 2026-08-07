import logging

import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from rest_framework import generics, permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from timeline.services import create_timeline_event

from .models import KnowledgeDocument, Prediction, Recommendation
from .serializers import (
    KnowledgeDocumentSerializer,
    PredictionRequestSerializer,
    PredictionSerializer,
    PredictionServiceResponseSerializer,
    RecommendationSerializer,
)


logger = logging.getLogger(__name__)


def _prediction_error(message, http_status, *, details=None):
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    return Response(payload, status=http_status)


class PredictionJSONErrorMixin:
    """Keep unexpected prediction API failures inside the JSON contract."""

    def handle_exception(self, exc):
        try:
            return super().handle_exception(exc)
        except Exception:
            logger.exception("Unexpected prediction API failure")
            return _prediction_error(
                "Unable to complete risk analysis.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PredictionCreateView(PredictionJSONErrorMixin, APIView):
    def post(self, request, disease):
        disease = disease.lower()
        request_serializer = PredictionRequestSerializer(
            data=request.data,
            context={"disease": disease},
        )
        if not request_serializer.is_valid():
            return _prediction_error(
                "Please correct the prediction form and try again.",
                status.HTTP_400_BAD_REQUEST,
                details=request_serializer.errors,
            )

        try:
            response = requests.post(
                f"{settings.FASTAPI_URL}/predict/{disease}",
                json=request_serializer.validated_data,
                timeout=90,
            )
        except requests.Timeout:
            logger.exception("FastAPI prediction request timed out for disease=%s", disease)
            return _prediction_error(
                "Risk analysis timed out. Please try again.",
                status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except requests.RequestException:
            logger.exception("FastAPI prediction request failed for disease=%s", disease)
            return _prediction_error(
                "Unable to reach the risk analysis service.",
                status.HTTP_502_BAD_GATEWAY,
            )

        if 400 <= response.status_code < 500:
            logger.warning(
                "FastAPI rejected prediction input for disease=%s with status=%s",
                disease,
                response.status_code,
            )
            return _prediction_error(
                "The risk analysis service rejected the submitted values.",
                status.HTTP_400_BAD_REQUEST,
            )
        if response.status_code >= 500:
            logger.error(
                "FastAPI prediction failed for disease=%s with status=%s",
                disease,
                response.status_code,
            )
            return _prediction_error(
                "Unable to complete risk analysis.",
                status.HTTP_502_BAD_GATEWAY,
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" not in content_type and "+json" not in content_type:
            logger.error(
                "FastAPI returned non-JSON prediction data for disease=%s content_type=%s",
                disease,
                content_type or "missing",
            )
            return _prediction_error(
                "The risk analysis service returned an invalid response.",
                status.HTTP_502_BAD_GATEWAY,
            )

        try:
            result = response.json()
        except ValueError:
            logger.exception("FastAPI returned malformed JSON for disease=%s", disease)
            return _prediction_error(
                "The risk analysis service returned an invalid response.",
                status.HTTP_502_BAD_GATEWAY,
            )

        if not isinstance(result, dict):
            logger.error("FastAPI returned a non-object JSON response for disease=%s", disease)
            return _prediction_error(
                "The risk analysis service returned an invalid response.",
                status.HTTP_502_BAD_GATEWAY,
            )

        result_serializer = PredictionServiceResponseSerializer(data=result)
        if not result_serializer.is_valid():
            logger.error(
                "FastAPI prediction response contract failed for disease=%s errors=%s",
                disease,
                result_serializer.errors,
            )
            return _prediction_error(
                "The risk analysis service returned an incomplete response.",
                status.HTTP_502_BAD_GATEWAY,
            )

        prediction_data = result_serializer.validated_data
        if prediction_data["disease"].lower() != disease:
            logger.error(
                "FastAPI prediction disease mismatch: requested=%s returned=%s",
                disease,
                prediction_data["disease"],
            )
            return _prediction_error(
                "The risk analysis service returned an inconsistent response.",
                status.HTTP_502_BAD_GATEWAY,
            )

        try:
            prediction = Prediction.objects.create(
                user=request.user,
                disease=prediction_data["disease"],
                input_data=request_serializer.validated_data,
                risk_percentage=prediction_data["risk_percentage"],
                risk_level=prediction_data["risk_level"],
                prediction=prediction_data["prediction"],
                shap_explanation=prediction_data.get("shap_explanation") or {},
                ai_recommendation=prediction_data.get("ai_recommendation", ""),
            )
            create_timeline_event(
                request.user,
                "prediction_made",
                f"{prediction_data['disease'].title()} risk prediction",
                f"Risk level: {prediction_data['risk_level']} ({prediction_data['risk_percentage']:.1f}%)",
                {"prediction_id": prediction.id, "disease": disease},
            )
        except Exception:
            logger.exception("Unable to persist prediction result for disease=%s", disease)
            return _prediction_error(
                "Unable to complete risk analysis.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_data = PredictionSerializer(prediction).data
        response_data["explanation_available"] = prediction_data.get(
            "explanation_available",
            False,
        )
        return Response(response_data, status=status.HTTP_200_OK)


class PredictionListView(PredictionJSONErrorMixin, generics.ListAPIView):
    serializer_class = PredictionSerializer

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)

    def list(self, request, *args, **kwargs):
        try:
            return super().list(request, *args, **kwargs)
        except Exception:
            logger.exception("Unable to list predictions for user=%s", request.user.pk)
            return _prediction_error(
                "Unable to load prediction history.",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


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
        shap_explanation = prediction.shap_explanation or {}
        top_factors = shap_explanation.get("top_factors", [])
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
                "explanation": shap_explanation.get("explanation_text", ""),
                "ai_recommendation": prediction.ai_recommendation,
                "rag_context": shap_explanation.get("rag_context", []),
            }
        )


@login_required
def explainability_page(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id, user=request.user)
    shap_explanation = prediction.shap_explanation or {}
    top_factors = shap_explanation.get("top_factors", [])
    return render(
        request,
        "reports/explainability.html",
        {
            "prediction": prediction,
            "labels": [factor.get("feature") for factor in top_factors],
            "values": [factor.get("contribution") for factor in top_factors],
            "colors": ["#dc2626" if factor.get("direction") == "increases_risk" else "#16a34a" for factor in top_factors],
            "rag_context": shap_explanation.get("rag_context", []),
        },
    )
