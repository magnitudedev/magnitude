"""A single packed owner serves every consumer without retaining GGUF backing."""

from contextlib import ExitStack

import gguf
import numpy as np
import pytest

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.numerics.encoded_layout import EncodedLayout
from magnitude_engine.operations.embedding import EncodedEmbedding
from magnitude_engine.operations.linear import EncodedLinear, ProjectionSchedule
from magnitude_engine.operations.parameters import DenseParameter
from magnitude_engine.operations.weights import ResidentWeight
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.machine import open_context
from performance.precision import decode, encode, rounded
from tests.numerics.test_encoded import packed


@pytest.mark.device
@pytest.mark.parametrize("encoding", (Encoding.Q4_K, Encoding.Q5_K, Encoding.Q6_K))
@pytest.mark.parametrize("dtype", (DType.F32, DType.BF16))
def test_weight_layout_shared_consumers_and_retirement(tmp_path, dtype, encoding):
    n, k = 9, 512
    raw = packed(encoding, n, k).reshape(n, -1)
    weights = gguf.dequantize(raw, gguf.GGMLQuantizationType(encoding))
    path = tmp_path / "weight.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor("weight", raw, raw_dtype=gguf.GGMLQuantizationType(encoding))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 8 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        owner = ResidentWeight(artifact, context, "weight", layout=EncodedLayout.AFFINE_PLANES)
        cleanup.callback(owner.close)
        assert owner.layout == EncodedLayout.AFFINE_PLANES
        assert (
            context.allocated_bytes
            == owner.nbytes
            == n * k * {Encoding.Q4_K: 6, Encoding.Q5_K: 7, Encoding.Q6_K: 8}[encoding] // 8
        )
        linear = EncodedLinear(owner, schedule=ProjectionSchedule.BASELINE)
        embedding = EncodedEmbedding(owner)
        parameter = DenseParameter(owner)
        for operation in (linear, embedding, parameter):
            cleanup.callback(operation.close)
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
        linear.close()
        embedding.close()
        owner.close()
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
def test_weight_layout_failed_later_chunk_releases_unpublished_backing(tmp_path, monkeypatch):
    # More than one bounded staging chunk; fail after the first has completed.
    raw = packed(Encoding.Q4_K, 2057, 512).reshape(2057, -1)
    path = tmp_path / "weight.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor("weight", raw, raw_dtype=gguf.GGMLQuantizationType.Q4_K)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 4 * 1024**2, 0)
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        original_read = artifact.source.read
        calls = 0

        def short_later_read(offset, size):
            nonlocal calls
            calls += 1
            value = original_read(offset, size)
            return value if calls == 1 else value[:-1]

        monkeypatch.setattr(artifact.source, "read", short_later_read)
        with pytest.raises(ValueError, match="short read"):
            ResidentWeight(artifact, context, "weight")
        assert calls == 2
        assert context.allocated_bytes == 0
