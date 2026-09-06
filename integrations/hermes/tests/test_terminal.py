import importlib.util
from pathlib import Path
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor


spec = importlib.util.spec_from_file_location(
    "magnitude_terminal", Path(__file__).parents[1] / "terminal.py"
)
terminal = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = terminal
spec.loader.exec_module(terminal)


class TerminalStatusTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.status = terminal.TerminalStatus(terminal.TerminalOutput(self.lines.append))

    def test_only_phase_transitions_print(self):
        self.status.phase("loading", "Loading 10%")
        self.status.phase("loading", "Loading 90%")
        self.status.phase("prefill", "Prefilling")
        self.assertEqual(self.lines, ["Loading 10%", "Prefilling"])

    def test_completion_is_exactly_once_and_closes_observation(self):
        self.status.finish("Complete")
        self.status.finish("Duplicate")
        self.status.phase("loading", "Stale")
        self.assertEqual(self.lines, ["Complete"])

    def test_silent_cancellation(self):
        self.status.finish()
        self.status.phase("generating", "Stale")
        self.assertEqual(self.lines, [])

    def test_concurrent_updates_coalesce(self):
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(lambda _: self.status.phase("prefill", "Prefilling"), range(100)))
        self.assertEqual(self.lines, ["Prefilling"])

    def test_terminal_control_characters_and_length_are_bounded(self):
        self.assertEqual(terminal.plain_line("a\x1b[2J\r\nb\x00"), "a [2J  b ")
        self.assertEqual(len(terminal.plain_line("x" * 1000)), 300)

    def test_no_terminal_application_means_no_output_surface(self):
        self.assertIsNone(terminal.TerminalOutput.capture())


if __name__ == "__main__":
    unittest.main()
