from rest_framework import serializers

from .models import KnowledgeDocument, Prediction, Recommendation


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = (
            "id",
            "disease",
            "input_data",
            "risk_percentage",
            "risk_level",
            "prediction",
            "shap_explanation",
            "ai_recommendation",
            "created_at",
        )
        read_only_fields = ("id", "created_at")


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = ("id", "category", "content", "source", "metadata", "created_at")
        read_only_fields = ("id", "created_at")


class KnowledgeDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeDocument
        fields = ("id", "file", "title", "source", "indexed_chunks", "metadata", "created_at")
        read_only_fields = ("id", "indexed_chunks", "metadata", "created_at")
