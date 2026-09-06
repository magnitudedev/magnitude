import importlib.util
from pathlib import Path
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event


spec = importlib.util.spec_from_file_location(
    "magnitude_progress_tests", Path(__file__).parents[1] / "__init__.py",
    submodule_search_locations=[str(Path(__file__).parents[1])],
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from magnitude_progress_tests.progress import InferenceProgress, is_magnitude_endpoint, register_progress
from magnitude_progress_tests.terminal import TerminalOutput


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.now = 10
        self.progress = InferenceProgress(
            capture=lambda: TerminalOutput(self.lines.append), clock=lambda: self.now,
        )
        self.event = dict(session_id="session", api_request_id="request",
                          model="local", base_url="http://127.0.0.1:10100/inference/v1")

    def test_endpoint_does_not_match_other_providers(self):
        self.assertTrue(is_magnitude_endpoint(self.event["base_url"] + "/"))
        for url in [None, "", "https://127.0.0.1:10100/inference/v1",
                    "http://127.0.0.1:10100.evil/inference/v1",
                    "http://user@127.0.0.1:10100/inference/v1",
                    "http://127.0.0.1:bad/inference/v1",
                    self.event["base_url"] + "?other=1", self.event["base_url"] + "#other"]:
            with self.subTest(url=url):
                self.assertFalse(is_magnitude_endpoint(url))

    def test_request_finishes_once(self):
        self.progress.start(**self.event)
        self.progress.start(**self.event)
        self.now = 12.5
        self.progress.finish(**self.event)
        self.progress.finish(**self.event)
        self.assertEqual(len(self.lines), 2)
        self.assertIn("Request finished · 2.5s", self.lines[-1])

    def test_failure_is_not_success(self):
        self.progress.start(**self.event)
        self.progress.finish(failed=True, **self.event)
        self.assertIn("Request failed", self.lines[-1])

    def test_sessions_do_not_finish_each_other(self):
        other = dict(self.event, session_id="other")
        self.progress.start(**self.event)
        self.progress.start(**other)
        self.progress.end_session(**self.event)
        self.progress.finish(**self.event)
        self.assertEqual(len(self.lines), 2)
        self.progress.finish(**other)
        self.assertEqual(len(self.lines), 3)

    def test_unload_closes_existing_and_future_requests(self):
        self.progress.start(**self.event)
        self.progress.close()
        self.progress.close()
        self.progress.finish(**self.event)
        self.progress.start(**dict(self.event, api_request_id="next"))
        self.assertEqual(len(self.lines), 1)

    def test_missing_identity_and_non_magnitude_requests_are_silent(self):
        for changes in [dict(session_id=None), dict(api_request_id=""),
                        dict(base_url="https://api.example.com/v1")]:
            self.progress.start(**dict(self.event, **changes))
        self.assertEqual(self.lines, [])

    def test_headless_requests_are_silent(self):
        progress = InferenceProgress(capture=lambda: None)
        progress.start(**self.event)
        progress.finish(**self.event)
        self.assertEqual(progress._requests, {})

    def test_concurrent_duplicate_callbacks(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda _: self.progress.start(**self.event), range(100)))
            list(workers.map(lambda _: self.progress.finish(**self.event), range(100)))
        self.assertEqual(len(self.lines), 2)

    def test_session_end_suppresses_completion_waiting_for_final_metadata(self):
        reading, release = Event(), Event()
        class Reader:
            def read(self, group):
                reading.set()
                if not release.wait(timeout=2):
                    raise AssertionError("Metadata read was not released")
                return []
        progress = InferenceProgress(capture=lambda: TerminalOutput(self.lines.append), reader=Reader())
        # Keep the background poller dormant: exercise the final metadata read alone.
        progress._stop.set()
        event = dict(self.event, group_id="group", observation_id="observation")
        progress.start(**event)
        with ThreadPoolExecutor(max_workers=1) as workers:
            finishing = workers.submit(progress.finish, **event)
            try:
                self.assertTrue(reading.wait(timeout=2))
                progress.end_session(**event)
            finally:
                release.set()
            finishing.result(timeout=2)
        progress.close()
        self.assertEqual(len(self.lines), 1)

    def test_registration_is_lazy_and_observers_fail_soft(self):
        class Context:
            def __init__(self):
                self.hooks = {}
            def register_hook(self, name, callback):
                self.hooks[name] = callback
            def register_middleware(self, name, callback):
                self.middleware = callback
            def on_unload(self, callback):
                self.close = callback
        ctx = Context()
        register_progress(ctx)
        for callback in ctx.hooks.values():
            callback(session_id=[] , api_request_id={})
        ctx.close()


if __name__ == "__main__":
    unittest.main()
