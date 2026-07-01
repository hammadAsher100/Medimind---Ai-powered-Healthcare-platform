from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import logout as django_logout
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET

from health_score.models import HealthScore
from recommendations.models import Prediction, Recommendation
from reports.models import MedicalReport
from timeline.models import TimelineEvent
from users.models import Allergy, FamilyHistory, MedicalProfile


def _common_context(request, title):
    return {
        "page_title": title,
        "MLFLOW_URL": getattr(settings, "MLFLOW_URL", "http://localhost:15000"),
    }


@require_GET
def login_page(request):
    if request.user.is_authenticated:
        return redirect("dashboard_page")
    return render(request, "auth/login.html")


@require_GET
def register_page(request):
    if request.user.is_authenticated:
        return redirect("dashboard_page")
    return render(request, "auth/register.html")


def logout_page(request):
    django_logout(request)
    return redirect("login_page")


@login_required
def dashboard_page(request):
    latest_score = HealthScore.objects.filter(user=request.user).first()
    reports = MedicalReport.objects.filter(user=request.user).order_by("-uploaded_at")
    predictions = Prediction.objects.filter(user=request.user).order_by("-created_at")
    timeline = TimelineEvent.objects.filter(user=request.user).order_by("-created_at")[:5]
    recommendations = Recommendation.objects.filter(user=request.user).order_by("-created_at")[:3]
    score_history = list(HealthScore.objects.filter(user=request.user).order_by("created_at")[:12])
    context = _common_context(request, "Dashboard")
    context.update({
        "latest_score": latest_score,
        "reports": reports[:5],
        "reports_count": reports.count(),
        "predictions": predictions[:8],
        "predictions_count": predictions.count(),
        "timeline_events": timeline,
        "recommendations": recommendations,
        "score_history": score_history,
    })
    return render(request, "dashboard/index.html", context)


@login_required
def predictions_index(request):
    latest = {}
    for item in Prediction.objects.filter(user=request.user).order_by("-created_at"):
        latest.setdefault(item.disease, item)
    context = _common_context(request, "Risk Assessment")
    context["latest_predictions"] = latest
    return render(request, "predictions/index.html", context)


@login_required
def prediction_form(request, disease):
    titles = {
        "diabetes": "Diabetes Risk Assessment",
        "heart": "Heart Disease Risk Assessment",
        "kidney": "Kidney Disease Risk Assessment",
        "stroke": "Stroke Risk Assessment",
    }
    if disease not in titles:
        return redirect("predictions_page")
    context = _common_context(request, titles[disease])
    context["disease"] = disease
    return render(request, f"predictions/{disease}.html", context)


@login_required
def prediction_result(request):
    return render(request, "predictions/result.html", _common_context(request, "Prediction Result"))


@login_required
def reports_index(request):
    context = _common_context(request, "Medical Reports")
    context["reports"] = MedicalReport.objects.filter(user=request.user).order_by("-uploaded_at")
    return render(request, "reports/index.html", context)


@login_required
def report_upload(request):
    return render(request, "reports/upload.html", _common_context(request, "Upload Report"))


@login_required
def report_detail(request, pk):
    report = MedicalReport.objects.filter(user=request.user, pk=pk).first()
    context = _common_context(request, "Report Detail")
    context["report"] = report
    return render(request, "reports/detail.html", context)


@login_required
def assistant_chat(request):
    return render(request, "assistant/chat.html", _common_context(request, "Medical Assistant"))


@login_required
def health_score_page(request):
    context = _common_context(request, "Health Score")
    context["latest_score"] = HealthScore.objects.filter(user=request.user).first()
    context["history"] = HealthScore.objects.filter(user=request.user).order_by("-created_at")[:12]
    return render(request, "health_score/index.html", context)


@login_required
def timeline_page(request):
    context = _common_context(request, "Timeline")
    context["events"] = TimelineEvent.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "timeline/index.html", context)


@login_required
def comparison_page(request):
    context = _common_context(request, "Compare Reports")
    context["reports"] = MedicalReport.objects.filter(user=request.user).order_by("-uploaded_at")
    return render(request, "comparison/index.html", context)


@login_required
def profile_page(request):
    context = _common_context(request, "My Profile")
    context["medical_profile"] = getattr(request.user, "medical_profile", None)
    context["allergies"] = Allergy.objects.filter(user=request.user)
    context["family_history"] = FamilyHistory.objects.filter(user=request.user)
    return render(request, "profile/index.html", context)


@user_passes_test(lambda user: user.is_staff)
def mlops_page(request):
    context = _common_context(request, "MLOps Dashboard")
    context.update({
        "total_predictions": Prediction.objects.count(),
        "reports_analyzed": MedicalReport.objects.exclude(analysis_result={}).count(),
        "active_users": max(Prediction.objects.values("user").distinct().count(), 1),
    })
    return render(request, "admin_panel/mlops.html", context)
