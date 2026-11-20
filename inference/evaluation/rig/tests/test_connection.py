"""Remote command boundaries preserve native arguments and process status."""

import ast
import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from magnitude_rig.connection import HOST_PACKAGE, CommandFailed, HostRun, Remote, windows_command


class RemoteCommandTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX shell working directory")
    def test_posix_working_directory_is_literal(self):
        remote = Remote("local", True, False, "Linux", "x86_64", sys.executable,
                        "/tmp", "/tmp", "/tmp/agent/package.zip", {}, ("cpu",), True)
        command = remote.command([sys.executable, "-c", "import os; print(os.getcwd())"],
                                 cwd=os.getcwd())
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), os.getcwd())

    def test_windows_command_has_no_shell_metacharacters_outside_encoding(self):
        command = windows_command(["python", "-c", 'print("$HOME; & literal")'])
        prefix = "powershell.exe -NoProfile -NonInteractive -EncodedCommand "
        self.assertTrue(command.startswith(prefix))
        script = base64.b64decode(command[len(prefix):]).decode("utf-16-le")
        self.assertIn("ProcessStartInfo", script)
        self.assertIn("exit $child.ExitCode", script)

    @unittest.skipUnless(os.name == "nt", "native Windows argument boundary")
    def test_windows_native_arguments_and_working_directory(self):
        values = ["{}", '{"key":"value"}', "", "two words", "a'b", 'a"b',
                  "trailing\\", "dollar$; & literal", "line\nbreak", "模型"]
        program = "import os,json,sys; print(json.dumps([os.getcwd(),sys.argv[1:]]))"
        command = windows_command([sys.executable, "-c", program, *values], cwd=os.getcwd())
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [os.getcwd(), values])

    @unittest.skipUnless(os.name == "nt", "native Windows exit status")
    def test_windows_native_exit_status(self):
        command = windows_command([sys.executable, "-c", "raise SystemExit(23)"])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 23)

    @unittest.skipUnless(os.name == "posix", "POSIX shell")
    def test_following_skips_shell_output_that_parses_as_json_and_names_a_failed_agent(self):
        """A login shell's bare number or list is no event; an agent that ends without the run's
        end failed as a command, not as a connection."""
        printed = "print(5); print('[1]'); print(json.dumps({'event': 'output', 'seq': 1})); "

        def follow(end: str) -> list:
            script = f"import json, sys; {printed}{end}"
            remote = Remote("local", True, False, "Linux", "x86_64", sys.executable, "/tmp", "/tmp",
                            "/tmp/agent/package.zip", {}, ("cpu",), True)
            run = HostRun(remote, "run", ["/tmp/agent/package.zip", "magnitude_rig.host.agent"])
            run._argv = lambda *arguments: [sys.executable, "-c", script]
            return list(run.follow(0))

        events = follow("print(json.dumps({'event': 'ended', 'seq': 2}))")
        self.assertEqual([event["event"] for event in events], ["output", "ended"])
        with self.assertRaises(CommandFailed):
            follow("sys.exit(3)")

    def test_instrument_remote_imports_stay_inside_host_package(self):
        """Every `magnitude_rig.*` module named by an instrument's remote command string must be
        packaged in `HOST_PACKAGE`, and `magnitude_rig.host.served.prompts` must import from that
        zip alone (isolated from the controller source tree)."""
        with zipfile.ZipFile(io.BytesIO(HOST_PACKAGE)) as archive:
            packaged = set()
            for name in archive.namelist():
                if name.endswith("/__init__.py"):
                    packaged.add(name[:-len("/__init__.py")].replace("/", "."))
                elif name.endswith(".py") and name != "__main__.py":
                    packaged.add(name[:-len(".py")].replace("/", "."))
        instruments_dir = Path(__file__).resolve().parent.parent / "src" / "magnitude_rig" / "instruments"
        missing = []
        for path in sorted(instruments_dir.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for module in re.findall(r"magnitude_rig(?:\.[A-Za-z_][A-Za-z0-9_]*)+", node.value):
                        if module not in packaged:
                            missing.append((path.name, module))
        self.assertEqual(missing, [], f"instrument remote command references unpackaged module(s): {missing}")

        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "agent.zip"
            zip_path.write_bytes(HOST_PACKAGE)
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = str(zip_path)
            result = subprocess.run(
                [sys.executable, "-S", "-c",
                 "import sys; sys.path = [p for p in sys.path if 'magnitude' not in p or p.endswith('agent.zip')]; "
                 "from magnitude_rig.host.served import prompts; print(prompts.__name__)"],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "prompts")


if __name__ == "__main__":
    unittest.main()
