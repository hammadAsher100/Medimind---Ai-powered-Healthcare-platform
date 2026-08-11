"""Scheduled, lease-aware shutdown controller for MediMind EC2."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

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
IDLE_SECONDS = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30")) * 60
MINIMUM_RUNTIME_SECONDS = int(os.environ.get("MINIMUM_RUNTIME_MINUTES", "15")) * 60
STATE_KEY = {"pk": {"S": f"INSTANCE#{INSTANCE_ID}"}, "sk": {"S": "STATE"}}


def _log(event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))


def _number(item: dict, name: str, default: int = 0) -> int:
    try:
        return int(item.get(name, {}).get("N", default))
    except (TypeError, ValueError):
        return default


def _instance():
    result = EC2.describe_instances(InstanceIds=[INSTANCE_ID])
    reservations = result.get("Reservations", [])
    if not reservations or not reservations[0].get("Instances"):
        return None
    return reservations[0]["Instances"][0]


def _active_leases(now: int) -> list[dict]:
    result = DDB.query(
        TableName=TABLE_NAME,
        KeyConditionExpression="pk = :pk AND begins_with(sk, :lease)",
        ExpressionAttributeValues={
            ":pk": {"S": f"INSTANCE#{INSTANCE_ID}"},
            ":lease": {"S": "LEASE#"},
        },
        ConsistentRead=True,
    )
    return [item for item in result.get("Items", []) if _number(item, "expires_at") > now]


def handler(_event, _context):
    now = int(time.time())
    token = uuid.uuid4().hex
    try:
        instance = _instance()
        if not instance or instance["State"]["Name"] != "running":
            _log("idle_check_skipped", reason="instance_not_running")
            return {"action": "none", "reason": "instance_not_running"}

        item = DDB.get_item(TableName=TABLE_NAME, Key=STATE_KEY, ConsistentRead=True).get("Item", {})
        launch_epoch = int(instance["LaunchTime"].timestamp())
        last_activity = max(
            _number(item, "last_activity_at"),
            _number(item, "wake_requested_at"),
            _number(item, "ready_at"),
            launch_epoch,
        )
        minimum_runtime_until = max(
            _number(item, "minimum_runtime_until"),
            launch_epoch + MINIMUM_RUNTIME_SECONDS,
        )
        deployment_until = _number(item, "deployment_until")

        if deployment_until > now:
            _log("idle_check_skipped", reason="deployment_active")
            return {"action": "none", "reason": "deployment_active"}
        if now < minimum_runtime_until:
            _log("idle_check_skipped", reason="minimum_runtime", remaining=minimum_runtime_until - now)
            return {"action": "none", "reason": "minimum_runtime"}
        if now - last_activity < IDLE_SECONDS:
            _log("idle_check_skipped", reason="recent_activity", idle_seconds=now - last_activity)
            return {"action": "none", "reason": "recent_activity"}

        DDB.update_item(
            TableName=TABLE_NAME,
            Key=STATE_KEY,
            UpdateExpression="SET shutdown_lock_until = :until, shutdown_token = :token, updated_at = :now",
            ConditionExpression=(
                "(attribute_not_exists(last_activity_at) OR last_activity_at <= :cutoff) AND "
                "(attribute_not_exists(deployment_until) OR deployment_until < :now) AND "
                "(attribute_not_exists(shutdown_lock_until) OR shutdown_lock_until < :now)"
            ),
            ExpressionAttributeValues={
                ":until": {"N": str(now + 120)},
                ":token": {"S": token},
                ":now": {"N": str(now)},
                ":cutoff": {"N": str(now - IDLE_SECONDS)},
            },
        )

        leases = _active_leases(now)
        refreshed = DDB.get_item(TableName=TABLE_NAME, Key=STATE_KEY, ConsistentRead=True).get("Item", {})
        if leases or _number(refreshed, "last_activity_at") > now - IDLE_SECONDS:
            DDB.update_item(
                TableName=TABLE_NAME,
                Key=STATE_KEY,
                UpdateExpression="REMOVE shutdown_lock_until, shutdown_token",
            )
            reason = "active_request" if leases else "new_activity"
            _log("shutdown_cancelled", reason=reason, active_leases=len(leases))
            return {"action": "none", "reason": reason}

        DDB.update_item(
            TableName=TABLE_NAME,
            Key=STATE_KEY,
            UpdateExpression="SET shutdown_committed_at = :now, stop_reason = :reason, updated_at = :now",
            ConditionExpression="shutdown_token = :token AND shutdown_lock_until > :now",
            ExpressionAttributeValues={
                ":now": {"N": str(now)},
                ":token": {"S": token},
                ":reason": {"S": "idle_timeout"},
            },
        )
        EC2.stop_instances(InstanceIds=[INSTANCE_ID])
        _log("instance_stop_requested", idle_seconds=now - last_activity, reason="idle_timeout")
        return {"action": "stop", "reason": "idle_timeout", "idle_seconds": now - last_activity}
    except DDB.exceptions.ConditionalCheckFailedException:
        _log("idle_check_skipped", reason="state_changed_or_locked")
        return {"action": "none", "reason": "state_changed_or_locked"}
    except ClientError as exc:
        _log("idle_check_aws_error", code=exc.response.get("Error", {}).get("Code"))
        raise
    except Exception:
        logger.exception("Unexpected idle shutdown failure")
        raise
