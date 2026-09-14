from dataclasses import FrozenInstanceError

import pytest

import ops
from engine.platform.storage import FileSource


def binding(source):
    return ops.Binding(
        ops.TensorSpec((4,), ops.DType.U8), "same-logical-weight", ops.Residency.STREAMED,
        (ops.SourcePlane(ops.SourceSpan(source, 0, 4), 1, 1),),
    )


def test_equal_values_on_different_paths_have_distinct_prepared_import_identity(tmp_path):
    first_path, second_path = tmp_path / "first.bin", tmp_path / "second.bin"
    first_path.write_bytes(b"same")
    second_path.write_bytes(b"same")
    with FileSource(first_path) as first, FileSource(second_path) as second:
        assert first.read(0, 4) == second.read(0, 4)
        assert binding(first).value_identity == binding(second).value_identity
        assert binding(first).fingerprint != binding(second).fingerprint
        assert first.info.kind == ops.SourceKind.FILE


def test_file_snapshot_identity_survives_path_replacement(tmp_path):
    path, replacement = tmp_path / "artifact", tmp_path / "replacement"
    path.write_bytes(b"old!")
    with FileSource(path) as original:
        revision = original.info.revision
        replacement.write_bytes(b"new!")
        replacement.replace(path)
        with FileSource(path) as updated:
            assert original.read(0, 4) == b"old!"
            assert updated.read(0, 4) == b"new!"
            assert original.info.identity == updated.info.identity
            assert original.info.revision == revision != updated.info.revision


def test_composite_source_retains_backing_provenance_and_region_identity():
    first, second = ops.MemorySource(b"abcd"), ops.ZeroSource(4)
    source = ops.SegmentedSource((ops.SourceSpan(first, 2, 2), ops.SourceSpan(second, 0, 2)))
    alternate = ops.SegmentedSource((ops.SourceSpan(first, 0, 2), ops.SourceSpan(second, 0, 2)))
    assert source.read(0, 4) == b"cd\x00\x00"
    assert source.info.dependencies == (first.info, second.info)
    assert source.info.kind == ops.SourceKind.COMPOSITE
    assert source.info.fingerprint != alternate.info.fingerprint


def test_segmented_source_fills_caller_storage_without_allocating_child_reads():
    class IntoOnly(ops.MemorySource):
        def read(self, offset, length):
            raise AssertionError("execution must use caller-owned staging")

    first = IntoOnly(b"abcdefgh")
    source = ops.SegmentedSource((ops.SourceSpan(first, 2, 3), ops.SourceSpan(ops.ZeroSource(4), 0, 4)))
    storage = bytearray(b"xxxxxx")
    assert source.read_into(1, memoryview(storage)) == 6
    assert storage == b"de\x00\x00\x00\x00"


def test_memory_source_content_and_provenance_are_immutable():
    mutable = bytearray(b"data")
    source = ops.MemorySource(mutable)
    mutable[0] = 0
    assert source.read(0, 4) == b"data"
    with pytest.raises(FrozenInstanceError):
        source.content = b"oops"


def test_reopened_artifact_replaces_prepared_source_handle(monkeypatch, tmp_path):
    from ops.runtime.imports import PreparedImport
    from tests.ops.test_compiler import Runtime

    monkeypatch.setattr(PreparedImport, "prepare", lambda self: self)
    path = tmp_path / "weight"
    path.write_bytes(b"same")
    with ops.DeviceRuntime(Runtime(), budget_bytes=1024) as device:
        with FileSource(path) as first:
            initial = device.prepare_binding(binding(first))
        with FileSource(path) as reopened:
            current = device.prepare_binding(binding(reopened))
            assert current is not initial
            assert current.plan.binding.fingerprint == initial.plan.binding.fingerprint
            assert current.plan.binding.planes[0].span.source.read(0, 4) == b"same"


def test_root_eviction_retires_regions_but_not_execution_leases(monkeypatch):
    from ops.runtime.imports import PreparedImport
    from tests.ops.test_compiler import Runtime

    monkeypatch.setattr(PreparedImport, "prepare", lambda self: self)
    root = binding(ops.MemorySource(b"abcd"))
    region = root.region(1, 2, shape=(2,))
    assert region.root_identity == root.root_identity
    with ops.DeviceRuntime(Runtime(), budget_bytes=1024) as device:
        prepared = device.prepare_binding(region)
        cached = device.upload(region.spec, b"bc")
        device._bindings[region.fingerprint] = cached
        execution = cached.fork()
        device.evict_binding(root)
        assert region.fingerprint not in device._imports
        assert region.fingerprint not in device._bindings
        assert not prepared._closed
        assert device.allocated_bytes == 2
        execution.close()
        assert device.allocated_bytes == 0
