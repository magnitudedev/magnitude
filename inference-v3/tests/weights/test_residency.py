"""One packed owner serves every consumer without retaining container backing."""

import os
from contextlib import ExitStack

import gguf
import numpy as np
import pytest

from magnitude_engine.kernels.precision import NATIVE_BF16, REFERENCE_F32
from magnitude_engine.operations.parameters import ResidentParameter
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.formats.gguf import Encoding, GGUFFormat
from magnitude_engine.weights.representation import HierarchicalAffine, resident_bytes
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
        assert isinstance(resident.representation, HierarchicalAffine)
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
        assert isinstance(resident.representation, HierarchicalAffine)
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
        assert isinstance(group.representation, HierarchicalAffine)
        assert context.allocated_bytes == gate.nbytes + up.nbytes
        operation = binding.operations.gated_linear(*descriptors)
        assert operation.plan(1, DType.BF16, DType.BF16).candidate == (
            "gated.hierarchical_fused_vector"
        )
        for start, descriptor, expected in zip(
            (0, gate.nbytes), descriptors, (gate, up), strict=True
        ):
            resident = binding.weights.existing(descriptor)
            assert resident is not None
            view = resident.acquire((TensorSpec((expected.nbytes,), DType.U8),))[0]
            try:
                assert view.offset == start
            finally:
                view.close()
