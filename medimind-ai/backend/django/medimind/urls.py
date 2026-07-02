from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from rest_framework_simplejwt.views import TokenRefreshView

from . import ai_proxy, page_views

urlpatterns = [
    path("", page_views.dashboard_page, name="dashboard_page"),
    path("login/", page_views.login_page, name="login_page"),
    path("register/", page_views.register_page, name="register_page"),
    path("logout/", page_views.logout_page, name="logout_page"),
    path("dashboard/", page_views.dashboard_page, name="dashboard_page"),
    path("predictions/", page_views.predictions_index, name="predictions_page"),
    path("predictions/result/", page_views.prediction_result, name="prediction_result"),
    path("predictions/<str:disease>/", page_views.prediction_form, name="prediction_form"),
    path("reports/", page_views.reports_index, name="reports_page"),
    path("reports/upload/", page_views.report_upload, name="upload_report_page"),
    path("reports/compare/", page_views.comparison_page, name="compare_reports_page"),
    path("reports/<int:pk>/", page_views.report_detail, name="report_detail_page"),
    path("assistant/", page_views.assistant_chat, name="assistant_page"),
    path("health-score/", page_views.health_score_page, name="health_score_page"),
    path("timeline/", page_views.timeline_page, name="timeline_page"),
    path("profile/", page_views.profile_page, name="profile_page"),
    path("admin-panel/mlops/", page_views.mlops_page, name="mlops_page"),
    path("admin/", admin.site.urls),
    path("api/auth/", include("authentication.urls")),
    path("api/auth/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/users/", include("users.urls")),
    path("api/reports/", include("reports.urls")),
    path("api/health-score/", include("health_score.urls")),
    path("api/timeline/", include("timeline.urls")),
    path("api/dashboard/", include("dashboard.urls")),
    path("api/", include("recommendations.urls")),
    # Proxy AI service requests (used in local dev; nginx handles this in Docker)
    path("ai/<path:path>", ai_proxy.ai_proxy, name="ai_proxy"),
    path("ai/", ai_proxy.ai_proxy, name="ai_proxy_root"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
