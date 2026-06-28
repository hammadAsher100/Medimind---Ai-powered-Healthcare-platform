from django.urls import path

from .views import TimelineListView

urlpatterns = [path("", TimelineListView.as_view(), name="timeline")]
