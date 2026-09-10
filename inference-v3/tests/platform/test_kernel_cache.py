"""Artifact identity and malformed storage cannot substitute a launch contract."""

import json

from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.kernel_cache import ArtifactCache, ArtifactKey
from magnitude_engine.platform.metal_compiler import MetalSource


def source():
    return MetalSource(
        source="kernel void f() {}",
        entry="f",
        signature=(TensorSpec((2, 3), DType.BF16), TensorSpec((3,), DType.F32)),
        permutation=(1, 0),
        groups=(1, 2, 3),
        threads=(32, 1, 1),
    )


def test_artifact_identity_checksum_and_typed_launch_roundtrip(tmp_path):
    cache = ArtifactCache(MetalSource, tmp_path)
    key = ArtifactKey(program="1" * 64, compiler="2" * 64, lowering="3" * 64)
    assert cache.read(key) is None
    cache.write(key, source())
    restored = ArtifactCache(MetalSource, tmp_path).read(key)
    assert restored == source()
    assert restored.signature[0].dtype is DType.BF16
    for field in ("program", "compiler", "lowering"):
        assert cache.read(key.model_copy(update={field: "4" * 64})) is None
    path = tmp_path / (key.digest + ".json")
    data = json.loads(path.read_bytes())
    data["artifact"]["permutation"] = [0, 1]
    path.write_text(json.dumps(data))
    assert cache.read(key) is None
    assert cache.statistics.invalid == 1
    cache.write(key, source())
    assert cache.read(key) == source()
    path.write_text('{"partial":')
    assert cache.read(key) is None
    assert cache.statistics.invalid == 2


def test_cache_storage_failure_is_observable_and_does_not_fail_execution(tmp_path):
    path = tmp_path / "not-a-directory"
    path.write_text("occupied")
    cache = ArtifactCache(MetalSource, path)
    key = ArtifactKey(program="1" * 64, compiler="2" * 64, lowering="3" * 64)
    assert cache.read(key) is None
    cache.write(key, source())
    assert cache.statistics.read_errors == cache.statistics.write_errors == 1
    assert cache.statistics.last_error


def test_specialization_normalizes_binding_without_erasing_parameter_meaning():
    from magnitude_engine.platform.specialization import Specialization

    def kernel(width, *, dtype=DType.F32, scale=0.0):
        raise AssertionError("identity lookup must not build IR")

    first = Specialization.bind(kernel, 1)
    assert first == Specialization.bind(kernel, width=1, dtype=DType.F32, scale=0.0)
    assert first != Specialization.bind(kernel, True)
    assert first != Specialization.bind(kernel, 1, scale=-0.0)
    assert first != Specialization.bind(kernel, 1, dtype="float32")
    import pytest

    with pytest.raises(TypeError, match="immutable"):
        Specialization.bind(kernel, [1])
