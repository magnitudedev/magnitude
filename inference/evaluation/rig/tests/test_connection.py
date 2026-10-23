"""Remote command boundaries preserve native arguments and process status."""

import base64
import json
import os
import subprocess
import sys
import unittest

from magnitude_rig.connection import Remote, windows_command


class RemoteCommandTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX shell working directory")
    def test_posix_working_directory_is_literal(self):
        remote = Remote("local", True, False, "Linux", "x86_64", sys.executable,
                        "/tmp", "/tmp", "/tmp/agent.py", {}, ("cpu",), True)
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


if __name__ == "__main__":
    unittest.main()
