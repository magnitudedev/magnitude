"""The installed public surface delegates to the ordinary engine contracts."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_import_is_lazy_and_public_exports_are_engine_contracts():
    source = """
import sys
import magnitude
assert "ops" not in sys.modules
assert "torch" not in sys.modules
assert "engine.serving.runtime" not in sys.modules
from magnitude import load_model
assert "ops" not in sys.modules
from magnitude import ModelRequest, LogitsSelection, SpecialTokens
from engine.models.sequence import ModelRequest as InternalRequest
assert ModelRequest is InternalRequest
assert LogitsSelection.NONE == "none"
assert SpecialTokens.LITERAL == "literal"
assert "fastapi" not in sys.modules
assert "uvicorn" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_invalid_load_fails_before_device_discovery(tmp_path):
    from magnitude import load_model

    with pytest.raises(FileNotFoundError):
        with load_model(tmp_path / "missing", memory_bytes=1024):
            pytest.fail("invalid path was loaded")
    with pytest.raises(ValueError, match="memory_bytes"):
        with load_model(tmp_path, memory_bytes=0):
            pytest.fail("invalid budget was loaded")


def test_wheel_manifest_includes_public_module_and_original_packages():
    import tomllib

    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert {"src/magnitude", "src/engine", "src/ops", "src/templates"} <= set(packages)


def test_public_request_can_read_logits_without_sampling():
    import ops
    from engine.models.qwen35.runtime import DenseRuntime
    from magnitude import LoadedModel, ModelRequest, TokenId
    from tests.models.test_forced_advance_numerics import description
    from tests.models.test_qwen35_runtime import ModelResidency
    from tests.ops.test_compiler import Runtime

    class ReadableRuntime(Runtime):
        def download(self, view):
            import struct

            _, spec, _ = view
            return struct.pack(f"={spec.elements}f", *range(spec.elements))

    device = ops.DeviceRuntime(ReadableRuntime(), budget_bytes=1 << 24)
    residency = ModelResidency(device)
    executor = DenseRuntime(description(), device, residency)
    # The fake native executor supplies known readback values, but exercises real
    # input assembly, graph construction, readback, and ownership boundaries.
    loaded = LoadedModel(executor, None)
    source = loaded.input((TokenId(1),))
    sequence = source.open()
    try:
        batch = executor.prepare((ModelRequest(sequence, (TokenId(1),)),))
        try:
            values = batch.advances[0].read_logits()
            assert len(values) == 1
            assert values[0] == tuple(float(i) for i in range(executor.geometry.vocabulary))
            assert batch.advances[0].read_sample() is None
            batch.advances[0].commit()
            assert sequence.position == 1
        finally:
            batch.close()
    finally:
        sequence.close()
        source.close()
        executor.close()
        for resource in residency.resources:
            resource.close()
        device.close()
