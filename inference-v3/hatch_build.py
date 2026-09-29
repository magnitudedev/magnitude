"""Bundle the native templates library in the existing engine distribution."""

import platform
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class NativeTemplatesHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        output = root / (
            "src/templates/_native" if version == "editable" else "build/templates-wheel/bundle"
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "native/templates/tools/build.py"),
                "--output",
                str(output),
                "--build-dir",
                str(root / "build/templates-wheel"),
                "--without-tests",
            ],
            check=True,
        )
        build_data["pure_python"] = False
        if sys.platform == "darwin":
            # Python's platform tag can claim an older deployment target than the
            # C++ compiler actually emitted. Derive it from this exact binary.
            metadata = subprocess.check_output(
                ["otool", "-l", str(output / "libtemplates.dylib")], text=True
            )
            match = re.search(r"\bminos (\d+)\.(\d+)", metadata)
            if match is None:
                raise RuntimeError("Native library has no verifiable macOS deployment target")
            target = f"macosx_{match[1]}_{match[2]}_{platform.machine()}"
        elif sys.platform == "linux":
            # A manylinux claim requires auditwheel qualification in a matching
            # build environment; the raw platform wheel makes no such claim.
            target = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        else:
            raise RuntimeError("Native templates packaging supports macOS and Linux only")
        build_data["tag"] = f"py3-none-{target}"
        if version != "editable":
            build_data["force_include"][str(output)] = "templates/_native"
