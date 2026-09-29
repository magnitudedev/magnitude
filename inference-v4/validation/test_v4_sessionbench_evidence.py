"""Focused source identity checks for the native Session Bench launcher."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from v4_sessionbench import runner_source_evidence, source_evidence, source_hash


def write(root: Path, name: str, content: str = "before") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class SourceEvidenceTests(unittest.TestCase):
    def test_engine_includes_native_template_sources_and_build_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = write(root, "validation/v4_sessionbench.py")
            cpp = write(root, "engine/templates/native/src/abi.cpp")
            header = write(root, "engine/templates/native/include/templates.h")
            write(root, "engine/templates/build.rs")
            write(root, "Cargo.lock")
            write(root, "target/release/output", "ignored")

            evidence = source_evidence(root, launcher)
            self.assertIn("engine/templates/native/src/abi.cpp", evidence)
            self.assertIn("engine/templates/native/include/templates.h", evidence)
            self.assertIn("engine/templates/build.rs", evidence)
            self.assertIn("Cargo.lock", evidence)
            self.assertNotIn("target/release/output", evidence)
            baseline = source_hash(evidence)
            cpp.write_text("after")
            self.assertNotEqual(baseline, source_hash(source_evidence(root, launcher)))
            cpp.write_text("before")
            header.write_text("after")
            self.assertNotEqual(baseline, source_hash(source_evidence(root, launcher)))

    def test_runner_includes_fixture_data_and_excludes_environment(self):
        with TemporaryDirectory() as directory:
            source = Path(directory)
            write(source, "pyproject.toml")
            write(source, "uv.lock")
            write(source, "src/session_bench/runner.py")
            fixture = write(source, "src/benchmark_fixtures/data/moby-dick.lock.json")
            write(source, ".venv/lib/installed.py", "ignored")

            evidence = runner_source_evidence(source)
            self.assertIn("src/session_bench/runner.py", evidence)
            self.assertIn("src/benchmark_fixtures/data/moby-dick.lock.json", evidence)
            self.assertNotIn(".venv/lib/installed.py", evidence)
            baseline = source_hash(evidence)
            fixture.write_text("after")
            self.assertNotEqual(baseline, source_hash(runner_source_evidence(source)))


if __name__ == "__main__":
    unittest.main()
