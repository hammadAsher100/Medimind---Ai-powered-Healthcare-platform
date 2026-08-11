"""Public wake/status control plane for the MediMind EC2 instance."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONFIG = Config(connect_timeout=2, read_timeout=4, retries={"max_attempts": 3})
EC2 = boto3.client("ec2", config=CONFIG)
DDB = boto3.client("dynamodb", config=CONFIG)

INSTANCE_ID = os.environ["EC2_INSTANCE_ID"]
TABLE_NAME = os.environ["ACTIVITY_TABLE_NAME"]
APP_URL = os.environ["APP_URL"].rstrip("/")
HEALTH_CHECK_URL = os.environ["HEALTH_CHECK_URL"]
MINIMUM_RUNTIME_SECONDS = int(os.environ.get("MINIMUM_RUNTIME_MINUTES", "15")) * 60
STARTUP_TIMEOUT_SECONDS = int(os.environ.get("STARTUP_TIMEOUT_SECONDS", "900"))
STATE_KEY = {"pk": {"S": f"INSTANCE#{INSTANCE_ID}"}, "sk": {"S": "STATE"}}


def _log(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))


def _response(status_code: int, payload: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Cache-Control": "no-store, max-age=0",
            "X-Content-Type-Options": "nosniff",
        },
        "body": json.dumps(payload, separators=(",", ":")),
    }


def _asset_response(filename: str, content_type: str) -> dict:
    asset_path = Path(__file__).parents[2] / "site" / filename
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": content_type,
            "Cache-Control": "public, max-age=300",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "same-origin",
            "Content-Security-Policy": (
                "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; object-src 'none'; "
                "base-uri 'self'; frame-ancestors 'none'"
            ),
        },
        "body": asset_path.read_text(encoding="utf-8"),
    }


def _request_context(event: dict) -> dict:
    http = (event.get("requestContext") or {}).get("http") or {}
    headers = {str(k).lower(): str(v) for k, v in (event.get("headers") or {}).items()}
    return {
        "method": http.get("method", ""),
        "path": http.get("path", ""),
        "source_ip": http.get("sourceIp", "unknown"),
        "user_agent": headers.get("user-agent", "")[:200],
        "request_id": (event.get("requestContext") or {}).get("requestId", ""),
    }


def _number(item: dict, name: str, default: int = 0) -> int:
    try:
        return int(item.get(name, {}).get("N", default))
    except (TypeError, ValueError):
        return default


def _state_item() -> dict:
    return DDB.get_item(TableName=TABLE_NAME, Key=STATE_KEY, ConsistentRead=True).get("Item", {})


def _instance_state() -> tuple[str, object]:
    result = EC2.describe_instances(InstanceIds=[INSTANCE_ID])
    reservations = result.get("Reservations", [])
    if not reservations or not reservations[0].get("Instances"):
        return "terminated", None
    instance = reservations[0]["Instances"][0]
    return instance["State"]["Name"], instance.get("LaunchTime")


def _record_wake_request(now: int, context: dict) -> None:
    DDB.update_item(
        TableName=TABLE_NAME,
        Key=STATE_KEY,
        UpdateExpression=(
            "SET wake_requested_at = :now, last_activity_at = :now, updated_at = :now, "
            "minimum_runtime_until = :minimum, requested_by = :source "
            "REMOVE shutdown_lock_until, shutdown_committed_at"
        ),
        ExpressionAttributeValues={
            ":now": {"N": str(now)},
            ":minimum": {"N": str(now + MINIMUM_RUNTIME_SECONDS)},
            ":source": {"S": f"public:{context['source_ip']}"[:200]},
        },
    )


def _try_start(now: int, context: dict) -> bool:
    try:
        DDB.update_item(
            TableName=TABLE_NAME,
            Key=STATE_KEY,
            UpdateExpression="SET wake_lock_until = :lock, start_requested_at = :now, updated_at = :now",
            ConditionExpression="attribute_not_exists(wake_lock_until) OR wake_lock_until < :now",
            ExpressionAttributeValues={
                ":lock": {"N": str(now + 60)},
                ":now": {"N": str(now)},
            },
        )
    except DDB.exceptions.ConditionalCheckFailedException:
        _log("start_suppressed_by_lock", **context)
        return False

    EC2.start_instances(InstanceIds=[INSTANCE_ID])
    _log("instance_start_requested", **context)
    return True


def _readiness() -> tuple[bool, str | None]:
    request = urllib.request.Request(
        HEALTH_CHECK_URL,
        headers={"User-Agent": "MediMindWakeController/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status != 200 or "application/json" not in content_type:
                return False, f"unexpected_status_{response.status}"
            payload = json.loads(response.read(32_768).decode("utf-8"))
            return payload.get("ready") is True, None
    except urllib.error.HTTPError as exc:
        return False, f"http_{exc.code}"
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return False, "unreachable"


def _starting_payload(now: int, started_at: int) -> dict:
    elapsed = max(0, now - started_at)
    if elapsed < 45:
        stage = "Starting secure server"
    elif elapsed < 150:
        stage = "Loading MediMind services"
    elif elapsed < 360:
        stage = "Loading AI models"
    else:
        stage = "Almost ready"
    return {"status": "starting", "stage": stage, "retry_after": 7}


def handler(event, _context):
    scheduled_wake = event.get("source") == "aws.events" and event.get("detail-type") == "Scheduled Event"
    context = (
        {"method": "POST", "path": "/control/wake", "source_ip": "eventbridge", "user_agent": "maintenance", "request_id": event.get("id", "")}
        if scheduled_wake
        else _request_context(event)
    )

    if not scheduled_wake and context["method"].upper() == "GET":
        static_routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
        }
        asset = static_routes.get(context["path"])
        if asset:
            return _asset_response(*asset)

    now = int(time.time())
    method = context["method"].upper()
    path = context["path"].rstrip("/")
    is_wake = method == "POST" and path.endswith("/control/wake")
    is_status = method == "GET" and path.endswith("/control/status")
    if not (is_wake or is_status):
        return _response(404, {"status": "error", "message": "Not found."})

    try:
        if is_wake:
            _record_wake_request(now, context)
            _log("wake_requested", **context)

        item = _state_item()
        state, launch_time = _instance_state()
        wake_requested_at = _number(item, "wake_requested_at")
        start_requested_at = _number(item, "start_requested_at", wake_requested_at or now)

        if state == "stopped":
            if is_wake or (wake_requested_at and now - wake_requested_at <= STARTUP_TIMEOUT_SECONDS):
                _try_start(now, context)
                return _response(202, _starting_payload(now, start_requested_at or now))
            return _response(200, {"status": "offline", "stage": "Ready to start"})

        if state == "stopping":
            return _response(202, {"status": "starting", "stage": "Preparing a secure restart", "retry_after": 8})

        if state == "pending":
            return _response(202, _starting_payload(now, start_requested_at))

        if state in {"shutting-down", "terminated"}:
            _log("instance_unrecoverable_state", state=state, **context)
            return _response(503, {"status": "error", "message": "MediMind is temporarily unavailable."})

        if state != "running":
            return _response(503, {"status": "error", "message": "MediMind could not be started."})

        ready, readiness_error = _readiness()
        if ready:
            first_ready = _number(item, "ready_at") == 0
            DDB.update_item(
                TableName=TABLE_NAME,
                Key=STATE_KEY,
                UpdateExpression="SET ready_at = :now, updated_at = :now REMOVE wake_lock_until",
                ExpressionAttributeValues={":now": {"N": str(now)}},
            )
            if first_ready:
                _log("application_ready", startup_seconds=max(0, now - start_requested_at), **context)
            return _response(200, {"status": "ready", "url": APP_URL})

        if now - start_requested_at > STARTUP_TIMEOUT_SECONDS:
            _log("startup_timeout", readiness_error=readiness_error, **context)
            return _response(503, {"status": "error", "message": "MediMind took longer than expected to start. Please retry."})

        _log("application_not_ready", reason=readiness_error, **context)
        return _response(202, _starting_payload(now, start_requested_at))
    except ClientError as exc:
        _log("aws_control_error", code=exc.response.get("Error", {}).get("Code"), **context)
        return _response(503, {"status": "error", "message": "MediMind could not be started right now."})
    except Exception:
        logger.exception("Unexpected wake controller failure")
        return _response(500, {"status": "error", "message": "MediMind could not be started right now."})
