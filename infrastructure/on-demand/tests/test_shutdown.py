import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


FUNCTION = Path(__file__).parents[1] / "functions" / "shutdown" / "app.py"


def load_module():
    os.environ.update(
        {
            "AWS_DEFAULT_REGION": "ap-south-1",
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "EC2_INSTANCE_ID": "i-0123456789abcdef0",
            "ACTIVITY_TABLE_NAME": "activity",
            "IDLE_TIMEOUT_MINUTES": "30",
            "MINIMUM_RUNTIME_MINUTES": "15",
        }
    )
    spec = importlib.util.spec_from_file_location("shutdown_test_module", FUNCTION)
    module = importlib.util.module_from_spec(spec)
    boto3_module = types.ModuleType("boto3")
    boto3_module.client = MagicMock(return_value=MagicMock())
    botocore_module = types.ModuleType("botocore")
    config_module = types.ModuleType("botocore.config")
    config_module.Config = lambda **_kwargs: object()
    exceptions_module = types.ModuleType("botocore.exceptions")
    exceptions_module.ClientError = type("ClientError", (Exception,), {})
    with patch.dict(
        sys.modules,
        {
            "boto3": boto3_module,
            "botocore": botocore_module,
            "botocore.config": config_module,
            "botocore.exceptions": exceptions_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class IdleShutdownTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.now = 2_000_000_000
        self.module.time.time = MagicMock(return_value=self.now)
        self.module._instance = MagicMock(
            return_value={
                "State": {"Name": "running"},
                "LaunchTime": datetime.fromtimestamp(self.now - 7200, tz=timezone.utc),
            }
        )
        self.module.DDB.get_item = MagicMock(
            return_value={"Item": {"last_activity_at": {"N": str(self.now - 3600)}}}
        )
        self.module.DDB.update_item = MagicMock()
        self.module._active_leases = MagicMock(return_value=[])
        self.module.EC2.stop_instances = MagicMock()

    def test_stops_only_after_idle_timeout(self):
        result = self.module.handler({}, None)
        self.assertEqual(result["action"], "stop")
        self.module.EC2.stop_instances.assert_called_once()

    def test_active_lease_cancels_shutdown(self):
        self.module._active_leases.return_value = [{"sk": {"S": "LEASE#1"}}]
        result = self.module.handler({}, None)
        self.assertEqual(result["reason"], "active_request")
        self.module.EC2.stop_instances.assert_not_called()

    def test_deployment_lease_prevents_shutdown(self):
        self.module.DDB.get_item.return_value = {
            "Item": {
                "last_activity_at": {"N": str(self.now - 3600)},
                "deployment_until": {"N": str(self.now + 600)},
            }
        }
        result = self.module.handler({}, None)
        self.assertEqual(result["reason"], "deployment_active")
        self.module.EC2.stop_instances.assert_not_called()

    def test_recent_activity_prevents_shutdown(self):
        self.module.DDB.get_item.return_value = {
            "Item": {"last_activity_at": {"N": str(self.now - 60)}}
        }
        result = self.module.handler({}, None)
        self.assertEqual(result["reason"], "recent_activity")
        self.module.EC2.stop_instances.assert_not_called()


if __name__ == "__main__":
    unittest.main()
