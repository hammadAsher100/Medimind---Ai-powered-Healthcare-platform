from .models import TimelineEvent


def create_timeline_event(user, event_type, title, description="", metadata=None):
    return TimelineEvent.objects.create(
        user=user,
        event_type=event_type,
        title=title,
        description=description,
        metadata=metadata or {},
    )
