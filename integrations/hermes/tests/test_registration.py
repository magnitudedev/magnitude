import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RegistrationTests(unittest.TestCase):
    def test_registration_and_setup_need_neither_cli_nor_service(self):
        for shared_skill in (False, True):
            with self.subTest(shared_skill=shared_skill), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                if shared_skill:
                    skill = home / "skills" / "magnitude" / "SKILL.md"
                    skill.parent.mkdir(parents=True)
                    skill.write_text("User-owned instructions")
                commands, skills, hooks, unload = {}, [], {}, []
                ctx = SimpleNamespace(
                    register_command=lambda name, callback, **_: commands.__setitem__(name, callback),
                    register_hook=lambda name, callback: hooks.__setitem__(name, callback),
                    register_middleware=lambda name, callback: None,
                    register_skill=lambda *args: skills.append(args),
                    on_unload=unload.append,
                )
                spec = importlib.util.spec_from_file_location(
                    "magnitude_registration_test", Path(__file__).parents[1] / "__init__.py",
                    submodule_search_locations=[str(Path(__file__).parents[1])],
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                with patch.dict(sys.modules, {"hermes_constants": SimpleNamespace(get_hermes_home=lambda: home)}), \
                     patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")), \
                     patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("Unexpected network")):
                    spec.loader.exec_module(module)
                    module.register(ctx)
                    self.assertIn(module.SETUP_PROMPT, commands["magnitude-setup"](""))
                    with patch.dict(sys.modules, {"jsonschema": None}):
                        self.assertIn("missing JSON Schema support", commands["magnitude"](""))
                        self.assertIn(module.SETUP_PROMPT, commands["magnitude-setup"](""))
                    for callback in unload:
                        callback()
                self.assertEqual(set(commands), {"magnitude", "load-model", "stop-model", "magnitude-setup"})
                self.assertEqual(len(skills), 0 if shared_skill else 1)
                if shared_skill:
                    self.assertEqual(skill.read_text(), "User-owned instructions")
                else:
                    self.assertEqual(skills[0][0], "usage")
                    self.assertEqual(skills[0][1].read_text(), (
                        Path(__file__).parents[3] / "cli/src/harness-connections/magnitude-skill.md"
                    ).read_text())


if __name__ == "__main__":
    unittest.main()
