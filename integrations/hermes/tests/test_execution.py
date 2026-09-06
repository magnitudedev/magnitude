import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

spec = importlib.util.spec_from_file_location(
    "magnitude_execution_tests", Path(__file__).parents[1] / "__init__.py",
    submodule_search_locations=[str(Path(__file__).parents[1])],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from magnitude_execution_tests.progress import InferenceProgress
from magnitude_execution_tests.terminal import TerminalOutput


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.progress = InferenceProgress(capture=lambda: TerminalOutput(self.lines.append),
                                         reader=SimpleNamespace(read=lambda _: None))
        self.event = dict(session_id="session", api_request_id="api", model="local",
                          base_url="http://127.0.0.1:10100/inference/v1")
        self.home = patch.dict(sys.modules, {"hermes_constants": SimpleNamespace(get_hermes_home=lambda: Path("/fixture"))})
        self.home.start()

    def tearDown(self):
        self.progress.close()
        self.home.stop()

    def test_exactly_once_transparent_result_and_fresh_attempt_ids(self):
        result = object()
        received = []
        request = {"messages": [{"role": "user", "content": "private"}], "stream": True,
                   "extra_headers": {"Keep": "unchanged", "magnitude-include-progress": "false"}}
        def call(payload):
            received.append(payload)
            return result
        for _ in range(2):
            self.assertIs(self.progress.execute(request=request, next_call=call, **self.event), result)
        self.assertEqual(len(received), 2)
        self.assertIs(received[0]["messages"], request["messages"])
        self.assertEqual(request["extra_headers"]["magnitude-include-progress"], "false")
        headers = [item["extra_headers"] for item in received]
        self.assertEqual(headers[0]["Keep"], "unchanged")
        self.assertEqual(headers[0]["Magnitude-Include-Progress"], "true")
        self.assertNotIn("magnitude-include-progress", headers[0])
        self.assertEqual(headers[0]["Magnitude-Observation-Group"], headers[1]["Magnitude-Observation-Group"])
        self.assertNotEqual(headers[0]["Magnitude-Observation-Id"], headers[1]["Magnitude-Observation-Id"])
        UUID(headers[0]["Magnitude-Observation-Id"])
        self.assertNotIn("private", "\n".join(self.lines))

    def test_exception_identity_and_no_retry(self):
        failure = RuntimeError("provider failed")
        calls = []
        def call(payload):
            calls.append(payload)
            raise failure
        with self.assertRaises(RuntimeError) as raised:
            self.progress.execute(request={}, next_call=call, **self.event)
        self.assertIs(raised.exception, failure)
        self.assertEqual(len(calls), 1)
        self.assertIn("Request failed", self.lines[-1])
        self.assertNotIn("Request finished", "\n".join(self.lines))

    def test_other_provider_request_is_untouched(self):
        request = {}
        received = []
        self.progress.execute(request=request, next_call=received.append,
                              **dict(self.event, base_url="https://other.example/v1"))
        self.assertEqual(received, [request])
        self.assertIs(received[0], request)
        self.assertEqual(self.lines, [])

    def test_broken_presentation_cannot_change_inference(self):
        self.progress._capture = lambda: (_ for _ in ()).throw(RuntimeError("UI failure"))
        calls = []
        self.progress.execute(request={}, next_call=calls.append, **self.event)
        self.assertEqual(len(calls), 1)

    def test_headless_correlates_without_terminal_or_observation_reads(self):
        self.progress._capture = lambda: None
        self.progress._reader = SimpleNamespace(read=lambda _: self.fail("Headless request must not poll"))
        received = []
        self.progress.execute(request={}, next_call=received.append, **self.event)
        self.assertIn("Magnitude-Observation-Id", received[0]["extra_headers"])
        self.assertEqual(self.lines, [])


if __name__ == "__main__":
    unittest.main()
