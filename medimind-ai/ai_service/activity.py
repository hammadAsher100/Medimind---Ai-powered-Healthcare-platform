"""DynamoDB-backed activity leases for safe EC2 idle shutdown."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)
_client = None


def enabled() -> bool:
    return (
        os.environ.get("ACTIVITY_TRACKING_ENABLED", "False").strip().lower() == "true"
        and bool(os.environ.get("ACTIVITY_TABLE_NAME"))
        and bool(os.environ.get("EC2_INSTANCE_ID"))
    )


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


def _start_lease(kind: str) -> str | None:
    if not enabled():
        return None
    now = int(time.time())
    lease_id = uuid.uuid4().hex
    table = os.environ["ACTIVITY_TABLE_NAME"]
    instance_id = os.environ["EC2_INSTANCE_ID"]
    try:
        _dynamodb().update_item(
            TableName=table,
            Key={"pk": {"S": f"INSTANCE#{instance_id}"}, "sk": {"S": "STATE"}},
            UpdateExpression="SET last_activity_at = :now, updated_at = :now REMOVE shutdown_lock_until, shutdown_committed_at",
            ExpressionAttributeValues={":now": {"N": str(now)}},
        )
        _dynamodb().put_item(
            TableName=table,
            Item={
                "pk": {"S": f"INSTANCE#{instance_id}"},
                "sk": {"S": f"LEASE#{lease_id}"},
                "kind": {"S": kind[:64]},
                "created_at": {"N": str(now)},
                "expires_at": {"N": str(now + int(os.environ.get("ACTIVITY_LEASE_SECONDS", "900")))},
            },
        )
        return lease_id
    except Exception:
        logger.exception("Unable to acquire FastAPI activity lease")
        return None


def _finish_lease(lease_id: str | None) -> None:
    if not lease_id or not enabled():
        return
    now = int(time.time())
    table = os.environ["ACTIVITY_TABLE_NAME"]
    instance_id = os.environ["EC2_INSTANCE_ID"]
    try:
        _dynamodb().delete_item(
            TableName=table,
            Key={"pk": {"S": f"INSTANCE#{instance_id}"}, "sk": {"S": f"LEASE#{lease_id}"}},
        )
        _dynamodb().update_item(
            TableName=table,
            Key={"pk": {"S": f"INSTANCE#{instance_id}"}, "sk": {"S": "STATE"}},
            UpdateExpression="SET last_activity_at = :now, updated_at = :now",
            ExpressionAttributeValues={":now": {"N": str(now)}},
        )
    except Exception:
        logger.exception("Unable to release FastAPI activity lease")


async def activity_middleware(request, call_next):
    ignored = (
        request.url.path in {"/health", "/readyz", "/metrics", "/cnn/models"}
        or request.headers.get("x-medimind-internal-check") == "deployment"
    )
    lease_id = None
    if enabled() and not ignored and request.method not in {"GET", "HEAD", "OPTIONS"}:
        lease_id = await asyncio.to_thread(_start_lease, "fastapi_request")
    try:
        return await call_next(request)
    finally:
        if lease_id:
            await asyncio.to_thread(_finish_lease, lease_id)
