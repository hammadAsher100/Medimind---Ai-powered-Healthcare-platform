import os

from django.conf import settings
from django.contrib.auth import logout as django_logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from health_score.models import HealthScore
from recommendations.models import Prediction, Recommendation
from reports.models import MedicalReport
from timeline.models import TimelineEvent
from users.models import Allergy, FamilyHistory, MedicalProfile

from clinical.models import (
    ClinicalObservation,
    DataConflict,
    DiagnosticReportRecord,
    PatientStateSnapshot,
)
from medication.models import (
    Medication,
    MedicationAllergy,
    MedicationSafetyAlert,
    PatientMedication,
)
from reviews.models import AuditEvent, ModelFeedback, ReviewDecision


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


@require_POST
def logout_page(request):
    """Log out via POST only — prevents CSRF logout attacks via GET links."""
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
        "score_history": HealthScore.objects.filter(user=request.user).order_by("-created_at")[:12],
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
def image_diagnosis_page(request):
    context = _common_context(request, "Image Diagnosis")
    context["model_id"] = "pneumonia_xray"
    return render(request, "predictions/image_diagnosis.html", context)


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
@require_GET
def report_download(request, pk):
    """Download a report through an ownership-checked application route."""
    report = MedicalReport.objects.filter(user=request.user, pk=pk).first()
    if not report or not report.file:
        raise Http404("Report not found")
    try:
        report.file.open("rb")
    except (FileNotFoundError, OSError):
        raise Http404("Report file not found")
    return FileResponse(
        report.file,
        as_attachment=True,
        filename=os.path.basename(report.file.name),
        content_type="application/pdf",
    )


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
    context["blood_type_choices"] = MedicalProfile.BLOOD_TYPES
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


# ─── Clinical Intelligence Pages ─────────────────────────────────────────────


@login_required
def clinical_dashboard_page(request):
    """Main clinical intelligence dashboard — patient state overview."""
    user = request.user
    latest_state = PatientStateSnapshot.objects.filter(
        user=user, is_current=True
    ).first()
    recent_conflicts = DataConflict.objects.filter(user=user)[:5]
    recent_observations = ClinicalObservation.objects.filter(user=user)[:10]
    recent_reports = DiagnosticReportRecord.objects.filter(user=user)[:5]

    context = _common_context(request, "Clinical Intelligence")
    context.update({
        "patient_state": latest_state,
        "recent_conflicts": recent_conflicts,
        "recent_observations": recent_observations,
        "recent_reports": recent_reports,
        "conflict_count": DataConflict.objects.filter(
            user=user, resolution_status="unresolved"
        ).count(),
        "observation_count": ClinicalObservation.objects.filter(user=user).count(),
    })
    return render(request, "clinical/dashboard.html", context)


@login_required
def lab_trends_page(request):
    """Longitudinal laboratory trends viewer."""
    user = request.user
    observations = ClinicalObservation.objects.filter(
        user=user
    ).order_by("-collection_date")

    # Group by test name for the frontend
    test_names = sorted(
        set(o.standardised_name or o.test_name for o in observations)
    )

    context = _common_context(request, "Lab Trends")
    context.update({
        "observations": observations[:50],
        "test_names": test_names,
    })
    return render(request, "clinical/lab_trends.html", context)


@login_required
def medications_page(request):
    """Patient medication list and management."""
    user = request.user
    medications = PatientMedication.objects.filter(
        user=user
    ).select_related("medication")
    allergies = MedicationAllergy.objects.filter(user=user)
    alerts = MedicationSafetyAlert.objects.filter(
        user=user, resolution_status="unresolved"
    )

    context = _common_context(request, "Medications")
    context.update({
        "medications": medications,
        "allergies": allergies,
        "alerts": alerts,
    })
    return render(request, "clinical/medications.html", context)


@login_required
def medication_safety_page(request):
    """Medication safety passport — interaction checks and alerts."""
    user = request.user
    medications = PatientMedication.objects.filter(
        user=user, status="active"
    ).select_related("medication")
    allergies = MedicationAllergy.objects.filter(user=user)
    alerts = MedicationSafetyAlert.objects.filter(user=user)[:20]

    context = _common_context(request, "Medication Safety")
    context.update({
        "medications": medications,
        "allergies": allergies,
        "alerts": alerts,
        "medication_payload": [
            {
                "medication_name": item.medication.name,
                "status": item.status,
                "dosage": item.dosage,
                "frequency": item.frequency,
                "drug_class": item.medication.drug_class,
            }
            for item in medications
        ],
        "allergy_payload": [
            {"allergen": item.allergen, "severity": item.severity}
            for item in allergies
        ],
    })
    return render(request, "clinical/medication_safety.html", context)


@login_required
def counterfactual_page(request):
    """Counterfactual health simulator."""
    user = request.user
    latest_predictions = {}
    for pred in Prediction.objects.filter(user=user).order_by("-created_at"):
        latest_predictions.setdefault(pred.disease, pred)

    context = _common_context(request, "Health Simulator")
    context["latest_predictions"] = latest_predictions
    return render(request, "clinical/counterfactual.html", context)


@login_required
def fhir_export_page(request):
    """FHIR export page."""
    user = request.user
    context = _common_context(request, "FHIR Export")
    context.update({
        "observation_count": ClinicalObservation.objects.filter(user=user).count(),
        "prediction_count": Prediction.objects.filter(user=user).count(),
        "has_health_score": HealthScore.objects.filter(user=user).exists(),
        "allergy_count": Allergy.objects.filter(user=user).count(),
    })
    return render(request, "clinical/fhir_export.html", context)


@login_required
def reviews_page(request):
    """Clinician review and model feedback."""
    user = request.user
    reviews = ReviewDecision.objects.filter(user=user)[:20]
    feedback = ModelFeedback.objects.filter(user=user)[:20]
    audit_events = AuditEvent.objects.filter(user=user)[:30]

    context = _common_context(request, "Reviews & Feedback")
    context.update({
        "reviews": reviews,
        "feedback": feedback,
        "audit_events": audit_events,
    })
    return render(request, "clinical/reviews.html", context)
