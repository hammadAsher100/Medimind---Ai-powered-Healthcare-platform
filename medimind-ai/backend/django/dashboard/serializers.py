from rest_framework import serializers


class DashboardSerializer(serializers.Serializer):
    latest_health_score = serializers.DictField(allow_null=True)
    recent_timeline = serializers.ListField()
    recent_reports = serializers.ListField()
    bmi = serializers.FloatField(allow_null=True)
    latest_predictions = serializers.ListField()
