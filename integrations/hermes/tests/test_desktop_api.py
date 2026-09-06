import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


class DesktopApiTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "hermes_dashboard_plugin_magnitude_test",
            Path(__file__).parents[1] / "dashboard" / "plugin_api.py",
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")), \
             patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Unexpected network")):
            spec.loader.exec_module(self.module)
        app = FastAPI()
        app.include_router(self.module.router, prefix="/api/plugins/magnitude")
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        for key in list(sys.modules):
            if key.startswith(self.module.__name__):
                del sys.modules[key]

    def test_setup_is_available_without_cli_or_service(self):
        with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")), \
             patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Unexpected network")):
            response = self.client.get("/api/plugins/magnitude/setup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prompt": self.module.companion.SETUP_PROMPT})

    def test_explicit_command_dispatches_once(self):
        with patch.object(self.module.commands, "load", return_value="Loaded fixture") as load:
            response = self.client.post("/api/plugins/magnitude/command", json={"operation": "load", "args": "fixture"})
        load.assert_called_once_with("fixture")
        self.assertEqual(response.json(), {"message": "Loaded fixture"})

    def test_invalid_commands_cannot_select_arbitrary_methods(self):
        for body in [{"operation": "__init__"}, {"operation": "stop", "extra": True},
                     {"operation": "load", "args": "x" * 2049}, {"operation": "load", "args": 1}]:
            with self.subTest(body=str(body)[:100]):
                response = self.client.post("/api/plugins/magnitude/command", json=body)
                self.assertEqual(response.status_code, 422)

    def test_get_cannot_invoke_command(self):
        self.assertEqual(self.client.get("/api/plugins/magnitude/command").status_code, 405)


if __name__ == "__main__":
    unittest.main()
