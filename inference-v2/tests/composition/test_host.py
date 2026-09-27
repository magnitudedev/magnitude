import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_complete_catalog_roundtrip_has_no_device_or_tokenizer_imports():
    script = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'mlx_vlm', 'transformers', 'numpy'}:
            raise AssertionError('host imported ' + fullname)
sys.meta_path.insert(0, Block())
from magnitude_engine import blueprints as bp
from magnitude_engine.composition import Catalog
from magnitude_engine.worker import Worker
source = bp.model.programs.qwen35.Program(
    artifact=bp.model.artifacts.Local(path='/model'),
    embedding=bp.model.embeddings.Streamed(cache_bytes=8192),
    feedforward=bp.model.feedforward.qwen35.MoE(experts=bp.model.experts.Streamed(slots=2)),
)
head = bp.model.programs.mtp.Head(artifact=bp.model.artifacts.Local(path='/head'), target=source)
engine = bp.engine.Engine(generation=bp.generation.Generation(
    target=bp.model.Executor(program=source, state=bp.model.state.PagedHybrid()),
    method=bp.generation.methods.MTP(drafter=bp.model.Executor(
        program=head, state=bp.model.state.Native(source=head))),
))
restored = bp.loads(bp.dumps(engine))
assert restored.generation.target.program is restored.generation.method.drafter.program.target
assert restored.generation.method.drafter.program is restored.generation.method.drafter.state.source
assert len(Catalog.exports(bp).types) >= 30
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_blueprint_api_is_statically_typed_without_generated_stubs(tmp_path):
    project = Path(__file__).resolve().parents[2]
    source = tmp_path / "consumer.py"
    source.write_text("""
from magnitude_engine import blueprints as bp
source = bp.model.programs.qwen35.Program(artifact=bp.model.artifacts.Local(path='/model'))
target = bp.model.Executor(program=source, state=bp.model.state.PagedHybrid())
engine = bp.engine.Engine(generation=bp.generation.Generation(target=target))
bad_dependency = bp.model.Executor(program=bp.resources.io.PositionalReader(), state=target.state)
bad_scalar = bp.model.embeddings.Streamed(cache_bytes='large')
bad_field = bp.engine.scheduling.TimeShared(unknown=1)
""")
    result = subprocess.run(
        [
            str(Path(sys.executable).parent / "pyright"),
            "--pythonpath",
            sys.executable,
            "--project",
            str(project / "pyproject.toml"),
            "--outputjson",
            str(source),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    errors = [d for d in report["generalDiagnostics"] if d["severity"] == "error"]
    assert len(errors) == 3, report
    assert {e["range"]["start"]["line"] for e in errors} == {5, 6, 7}


@pytest.mark.parametrize("override", ["--context-tokens", "--retained-prefixes"])
def test_serving_rejects_ignored_blueprint_overrides(override):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "magnitude_engine.serving",
            "--model",
            "test",
            "--engine-blueprint",
            "/not/opened.json",
            override,
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "inside --engine-blueprint" in result.stderr
