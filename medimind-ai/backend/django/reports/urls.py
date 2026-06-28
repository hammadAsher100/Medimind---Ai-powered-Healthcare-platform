from django.urls import path

from .views import ReportDetailView, ReportListView, ReportUploadView

urlpatterns = [
    path("", ReportListView.as_view(), name="report-list"),
    path("upload/", ReportUploadView.as_view(), name="report-upload"),
    path("<int:pk>/", ReportDetailView.as_view(), name="report-detail"),
]
