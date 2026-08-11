"""Low-cost activity and in-flight request tracking for EC2 idle shutdown."""

from __future__ import annotations

import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)

_client = None
_last_touch = 0


def _enabled() -> bool:
    return os.environ.get("ACTIVITY_TRACKING_ENABLED", "False").strip().lower() == "true"


def _state_key() -> dict[str, dict[str, str]]:
    instance_id = os.environ.get("EC2_INSTANCE_ID", "").strip()
    return {"pk": {"S": f"INSTANCE#{instance_id}"}, "sk": {"S": "STATE"}}


def _dynamodb():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        _client = boto3.client(
            "dynamodb",
            region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
            config=Config(connect_timeout=1, read_timeout=1, retries={"max_attempts": 2}),
        )
    return _client


def _configuration_valid() -> bool:
    return bool(os.environ.get("ACTIVITY_TABLE_NAME") and os.environ.get("EC2_INSTANCE_ID"))


def touch_activity(*, force: bool = False) -> None:
    """Refresh last activity, coalescing ordinary page requests per worker."""
    global _last_touch
    if not _enabled() or not _configuration_valid():
        return
    now = int(time.time())
    interval = int(os.environ.get("ACTIVITY_TOUCH_INTERVAL_SECONDS", "30"))
    if not force and now - _last_touch < interval:
        return
    try:
        _dynamodb().update_item(
            TableName=os.environ["ACTIVITY_TABLE_NAME"],
            Key=_state_key(),
            UpdateExpression=(
                "SET last_activity_at = :now, updated_at = :now "
                "REMOVE shutdown_lock_until, shutdown_committed_at"
            ),
            ExpressionAttributeValues={":now": {"N": str(now)}},
        )
        _last_touch = now
    except Exception:
        logger.exception("Unable to record MediMind activity")


def acquire_lease(kind: str = "django_request") -> str | None:
    """Create an expiring lease that prevents shutdown during a mutating request."""
    if not _enabled() or not _configuration_valid():
        return None
    now = int(time.time())
    lease_id = uuid.uuid4().hex
    ttl = now + int(os.environ.get("ACTIVITY_LEASE_SECONDS", "900"))
    instance_id = os.environ["EC2_INSTANCE_ID"].strip()
    try:
        touch_activity(force=True)
        _dynamodb().put_item(
            TableName=os.environ["ACTIVITY_TABLE_NAME"],
            Item={
                "pk": {"S": f"INSTANCE#{instance_id}"},
                "sk": {"S": f"LEASE#{lease_id}"},
                "kind": {"S": kind[:64]},
                "created_at": {"N": str(now)},
                "expires_at": {"N": str(ttl)},
            },
        )
        return lease_id
    except Exception:
        logger.exception("Unable to acquire MediMind activity lease")
        return None


def release_lease(lease_id: str | None) -> None:
    if not lease_id or not _enabled() or not _configuration_valid():
        return
    instance_id = os.environ["EC2_INSTANCE_ID"].strip()
    try:
        _dynamodb().delete_item(
            TableName=os.environ["ACTIVITY_TABLE_NAME"],
            Key={
                "pk": {"S": f"INSTANCE#{instance_id}"},
                "sk": {"S": f"LEASE#{lease_id}"},
            },
        )
        touch_activity(force=True)
    except Exception:
        logger.exception("Unable to release MediMind activity lease")


class ActivityTrackingMiddleware:
    """Track authenticated traffic and protect in-flight writes from shutdown."""

    _SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
    _IGNORED_PATHS = ("/readyz", "/healthz", "/static/", "/media/")

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        ignored = any(path.startswith(prefix) for prefix in self._IGNORED_PATHS)
        authenticated = bool(getattr(getattr(request, "user", None), "is_authenticated", False))
        auth_submission = request.method == "POST" and path in {"/login/", "/register/", "/api/auth/login/", "/api/auth/register/"}
        meaningful = not ignored and (authenticated or auth_submission)
        lease_id = None

        if meaningful:
            touch_activity()
            if request.method not in self._SAFE_METHODS:
                lease_id = acquire_lease("django_mutation")

        try:
            return self.get_response(request)
        finally:
            if meaningful:
                release_lease(lease_id)
