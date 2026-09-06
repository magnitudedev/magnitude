import importlib.util
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest


spec = importlib.util.spec_from_file_location(
    "magnitude_commands_test", Path(__file__).parents[1] / "__init__.py",
    submodule_search_locations=[str(Path(__file__).parents[1])],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from magnitude_commands_test.commands import ModelCommands, format_catalog


class CommandTests(unittest.TestCase):
    def test_initializing_and_empty_catalog(self):
        self.assertIn("discovering", format_catalog({"_tag": "Initializing"}))
        self.assertIn("/magnitude-setup", format_catalog({"_tag": "Ready", "models": []}))

    def test_installed_models_preserve_acquisition_and_residency_information(self):
        def local(model_id, state):
            return {"_tag": "Local", "product": {
                "_tag": "Catalog", "modelId": model_id, "acquisitionState": state,
            }}
        result = format_catalog({"_tag": "Ready", "models": [
            {"_tag": "Remote"},
            local("not-installed", {"_tag": "NotInstalled"}),
            local("z-model", {"_tag": "Installed", "residencyState": {"_tag": "Ready"}}),
            local("a-model", {"_tag": "UpdateAvailable", "residencyState": {"_tag": "Unloaded"}}),
            {"_tag": "Local", "product": {"_tag": "Discovered", "modelId": "b-model",
                "state": {"_tag": "Ready", "residencyState": {"_tag": "Loading"}}}},
        ]})
        self.assertEqual(result, "Magnitude local models:\na-model · Unloaded · Update available\nb-model · Loading\nz-model · Ready")

    def test_usage_errors_do_not_construct_client(self):
        commands = ModelCommands(lambda: self.fail("Unexpected client construction"))
        for call, args in ((commands.status, "extra"), (commands.stop, "extra"), (commands.load, " ")):
            self.assertTrue(call(args).startswith("Usage:"))

    def test_concurrent_commands_share_client_without_serializing_requests(self):
        barrier = Barrier(2)
        constructed = []
        class Client:
            def call(self, operation, payload):
                barrier.wait(timeout=2)
                return {"_tag": "Initializing"}
        def factory():
            client = Client()
            constructed.append(client)
            return client
        commands = ModelCommands(factory)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(commands.status, ["", ""]))
        self.assertEqual(len(constructed), 1)
        self.assertTrue(all("discovering" in result for result in results))


if __name__ == "__main__":
    unittest.main()
