from django.contrib import admin

from .models import TimelineEvent


@admin.register(TimelineEvent)
class TimelineEventAdmin(admin.ModelAdmin):
    list_display = ("user", "event_type", "title", "created_at")
    list_filter = ("event_type", "created_at")
