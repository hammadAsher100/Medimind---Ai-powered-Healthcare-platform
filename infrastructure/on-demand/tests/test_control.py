import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


FUNCTION = Path(__file__).parents[1] / "functions" / "control" / "app.py"


def load_module():
    os.environ.update(
        {
            "AWS_DEFAULT_REGION": "ap-south-1",
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "EC2_INSTANCE_ID": "i-0123456789abcdef0",
            "ACTIVITY_TABLE_NAME": "activity",
            "APP_URL": "https://app.medimind-ai.online",
            "HEALTH_CHECK_URL": "https://app.medimind-ai.online/readyz",
            "ORIGIN_TOKEN": "x" * 32,
        }
    )
    spec = importlib.util.spec_from_file_location("wake_control_test_module", FUNCTION)
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


class WakeControllerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.event = {
            "headers": {"x-medimind-origin-token": "x" * 32},
            "requestContext": {
                "requestId": "request-1",
                "http": {"method": "POST", "path": "/control/wake", "sourceIp": "203.0.113.10"},
            },
        }
        self.module._record_wake_request = MagicMock()
        self.module._state_item = MagicMock(return_value={})

    def body(self, response):
        import json

        return json.loads(response["body"])

    def test_rejects_requests_without_cloudfront_token(self):
        self.event["headers"] = {}
        response = self.module.handler(self.event, None)
        self.assertEqual(response["statusCode"], 403)

    def test_stopped_instance_is_started(self):
        self.module._instance_state = MagicMock(return_value=("stopped", None))
        self.module._try_start = MagicMock(return_value=True)
        response = self.module.handler(self.event, None)
        self.assertEqual(response["statusCode"], 202)
        self.assertEqual(self.body(response)["status"], "starting")
        self.module._try_start.assert_called_once()

    def test_pending_instance_is_not_started_again(self):
        self.module._instance_state = MagicMock(return_value=("pending", None))
        self.module._try_start = MagicMock()
        response = self.module.handler(self.event, None)
        self.assertEqual(self.body(response)["status"], "starting")
        self.module._try_start.assert_not_called()

    def test_running_instance_redirects_only_after_readiness(self):
        self.module._instance_state = MagicMock(return_value=("running", None))
        self.module._readiness = MagicMock(return_value=(True, None))
        self.module.DDB.update_item = MagicMock()
        response = self.module.handler(self.event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(self.body(response), {"status": "ready", "url": "https://app.medimind-ai.online"})

    def test_terminated_instance_returns_clean_failure(self):
        self.module._instance_state = MagicMock(return_value=("terminated", None))
        response = self.module.handler(self.event, None)
        self.assertEqual(response["statusCode"], 503)
        self.assertNotIn("instance", self.body(response)["message"].lower())


if __name__ == "__main__":
    unittest.main()
