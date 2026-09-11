"""One packed owner serves every consumer without retaining container backing."""

import os
from contextlib import ExitStack

import gguf
import numpy as np
import pytest

from magnitude_engine.kernels.precision import NATIVE_BF16, REFERENCE_F32
from magnitude_engine.operations.parameters import ResidentParameter
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec, Ticket
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.weights.descriptor import StoredQuantized, WeightDescriptor
from magnitude_engine.weights.formats.gguf import Encoding, GGUFFormat, quantization
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.representation import (
    Affine,
    HierarchicalCoefficients,
    resident_bytes,
)
from magnitude_engine.weights.residency import Weights
from performance.precision import decode, encode, rounded
from tests.kernels.test_encoded import packed
from tests.support import bind


def write(path, raw, encoding, name="weight"):
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType(encoding))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_many(path, tensors, encoding):
    writer = gguf.GGUFWriter(path, "qwen35")
    for name, raw in tensors:
        writer.add_tensor(name, raw, raw_dtype=gguf.GGMLQuantizationType(encoding))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_chunked_import_failure_does_not_publish_partial_residency(monkeypatch):
    from magnitude_engine.weights import residency as residency_module

    raw = packed(Encoding.Q4_K, 4, 512)
    events = []

    original_wait = Ticket.wait

    def tracked_wait(ticket):
        events.append("wait")
        return original_wait(ticket)

    monkeypatch.setattr(Ticket, "wait", tracked_wait)

    class FailingSource:
        size = raw.nbytes
        reads = 0

        def read(self, offset, length):
            events.append("read")
            self.reads += 1
            if self.reads == 2:
                raise OSError("deliberate source failure")
            return raw.tobytes()[offset : offset + length]

    class Format:
        identity = ArtifactIdentity("failing-import")
        source = FailingSource()

        def stored(self, descriptor):
            representation, codec = quantization(Encoding.Q4_K)
            return StoredQuantized(representation, codec, self.source, 0)

        def close(self):
            pass

    monkeypatch.setattr(residency_module, "IMPORT_CHUNK_BYTES", 2 * Encoding.Q4_K.block_bytes)
    context = open_context(Backend.LLVM, 1024**2, 0)
    weights = Weights(Format(), context)
    with pytest.raises(OSError, match="deliberate source failure"):
        weights.resident(WeightDescriptor(name="weight", shape=(4, 512)))
    assert weights.existing(WeightDescriptor(name="weight", shape=(4, 512))) is None
    assert context.allocated_bytes == 0
    assert events[:2] == ["read", "read"]
    assert "wait" in events[2:]
    weights.close()
    context.close()


@pytest.mark.device
@pytest.mark.parametrize("encoding", (Encoding.Q4_K, Encoding.Q5_K, Encoding.Q6_K))
@pytest.mark.parametrize("dtype", (DType.F32, DType.BF16))
def test_compact_hierarchy_serves_every_consumer_and_survives_retirement(tmp_path, dtype, encoding):
    n, k = 9, 512
    raw = packed(encoding, n, k).reshape(n, -1)
    weights = gguf.dequantize(raw, gguf.GGMLQuantizationType(encoding))
    path = tmp_path / "weight.gguf"
    write(path, raw, encoding)
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 8 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFFormat(str(path))
        cleanup.callback(artifact.close)
        binding = bind(artifact, context, REFERENCE_F32)
        cleanup.callback(binding.close)
        descriptor = WeightDescriptor(name="weight", shape=(n, k))
        resident = binding.weights.resident(descriptor)
        assert isinstance(resident.representation, Affine)
        assert isinstance(resident.representation.coefficients, HierarchicalCoefficients)
        assert (
            context.allocated_bytes
            == resident.nbytes
            == resident_bytes(resident.representation, n * k)
            == raw.nbytes
        )
        linear = binding.operations.linear(descriptor)
        embedding = binding.operations.embedding(descriptor)
        parameter = ResidentParameter(resident)
        cleanup.callback(parameter.close)
        dense = parameter.acquire()
        cleanup.callback(dense.close)
        commands, checks = [], []
        for rows in (1, 8):
            inputs = rounded(
                np.random.default_rng(rows).normal(size=(rows, k)).astype(np.float32), dtype
            )
            a = context.upload(TensorSpec(inputs.shape, dtype), encode(inputs, dtype))
            b = context.allocate(TensorSpec((rows, n), dtype))
            cleanup.callback(a.close)
            cleanup.callback(b.close)
            commands.extend(linear.prepare(a, b))
            # Matrix BF16 weights round once at the shared matrix operand.
            w = rounded(weights, dtype) if rows == 8 else weights
            expected = inputs.astype(np.float64) @ w.astype(np.float64).T
            bound = (
                2e-6 * (np.abs(inputs.astype(np.float64)) @ np.abs(w.astype(np.float64)).T) + 1e-6
            )
            if dtype == DType.BF16:
                bound = (1 + 2**-8) * bound + 2**-8 * np.abs(expected)
            checks.append((b, expected, bound))
        ids = np.array([8, 0, 8], np.int32)
        indices = context.upload(TensorSpec((3,), DType.I32), ids.tobytes())
        selected = context.allocate(TensorSpec((3, k), dtype))
        cleanup.callback(indices.close)
        cleanup.callback(selected.close)
        commands.extend(embedding.prepare(indices, selected))
        # Commands own the final planes even after every supplying owner retires.
        binding.close()
        artifact.close()
        ticket = context.submit(commands)
        ticket.wait()
        for output, expected, bound in checks:
            actual = decode(context.read(output, after=ticket), dtype).reshape(expected.shape)
            assert np.all(np.abs(actual - expected) <= bound)
        actual = decode(context.read(selected, after=ticket), dtype).reshape(3, k)
        np.testing.assert_array_equal(actual, rounded(weights[ids], dtype))
        actual_dense = np.frombuffer(context.read(dense, after=ticket), np.float32).reshape(n, k)
        np.testing.assert_array_equal(actual_dense, weights)


@pytest.mark.device
def test_blocked_representation_when_a_row_cannot_fill_a_fold(tmp_path):
    """Compact hierarchical residency does not depend on one schedule's fold width."""
    n, k = 4, 256
    raw = packed(Encoding.Q4_K, n, k).reshape(n, -1)
    path = tmp_path / "narrow.gguf"
    write(path, raw, Encoding.Q4_K)
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 4 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFFormat(str(path))
        cleanup.callback(artifact.close)
        binding = bind(artifact, context, REFERENCE_F32)
        cleanup.callback(binding.close)
        resident = binding.weights.resident(WeightDescriptor(name="weight", shape=(n, k)))
        assert isinstance(resident.representation, Affine)
        assert isinstance(resident.representation.coefficients, HierarchicalCoefficients)
        assert resident.nbytes == raw.nbytes


@pytest.mark.device
def test_compact_hierarchies_group_by_rows_without_changing_bytes(tmp_path):
    n, k = 5, 512
    gate = packed(Encoding.Q4_K, n, k).reshape(n, -1)
    up = packed(Encoding.Q4_K, n, k).reshape(n, -1)
    path = tmp_path / "group.gguf"
    write_many(path, (("gate", gate), ("up", up)), Encoding.Q4_K)
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 4 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFFormat(str(path))
        cleanup.callback(artifact.close)
        binding = bind(artifact, context, NATIVE_BF16)
        cleanup.callback(binding.close)
        descriptors = (
            WeightDescriptor(name="gate", shape=(n, k)),
            WeightDescriptor(name="up", shape=(n, k)),
        )
        group = binding.weights.group(descriptors)
        assert group is not None
        assert isinstance(group.representation, Affine)
        assert isinstance(group.representation.coefficients, HierarchicalCoefficients)
        assert context.allocated_bytes == gate.nbytes + up.nbytes
        operation = binding.operations.gated_linear(*descriptors)
        assert operation.plan(1, DType.BF16, DType.BF16).candidate == (
            "gated.hierarchical_fused_vector"
        )
        for first_row, descriptor in zip((0, n), descriptors, strict=True):
            resident = binding.weights.existing(descriptor)
            assert resident is not None
            assert resident.layout.first_row == first_row
            assert resident.layout.logical_rows == n
            view = resident.acquire((TensorSpec((gate.nbytes + up.nbytes,), DType.U8),))[0]
            try:
                assert view.offset == 0
            finally:
                view.close()
