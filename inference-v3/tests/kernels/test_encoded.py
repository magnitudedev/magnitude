"""Independent GGUF interpretation followed by an ordinary reference contraction."""

import os
from pathlib import Path

import gguf
import numpy as np
import pytest

from magnitude_engine.kernels.precision import NATIVE_BF16
from magnitude_engine.kernels.projection.serial import projection as projection_cpu
from magnitude_engine.kernels.projection.threadgroup import projection
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.platform.storage import FileSource
from magnitude_engine.weights.formats.gguf import Encoding, block_layout, read_directory
from performance.precision import decode, encode, rounded


def packed(encoding: Encoding, outputs: int, inputs: int, *, seed: int = 284) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = outputs * inputs // encoding.block_elements
    data = rng.integers(0, 256, (blocks, encoding.block_bytes), dtype=np.uint8)
    if encoding == Encoding.F32:
        return rng.normal(size=(outputs, inputs)).astype(np.float32).view(np.uint8).ravel()
    if encoding == Encoding.F16:
        return rng.normal(size=(outputs, inputs)).astype(np.float16).view(np.uint8).ravel()
    offsets = (
        (0, 2)
        if encoding in (Encoding.Q4_K, Encoding.Q5_K)
        else ((208,) if encoding == Encoding.Q6_K else (0,))
    )
    for offset in offsets:
        scales = rng.uniform(-0.02, 0.02, blocks).astype(np.float16)
        data[:, offset : offset + 2] = scales.view(np.uint8).reshape(-1, 2)
    return data.ravel()


def check_projection(
    data: np.ndarray,
    encoding: Encoding,
    outputs: int,
    inputs: int,
    rows: int,
    *,
    matrix=False,
    subgroup=False,
    packed_k=False,
    groupwise=False,
    row_tile=1,
) -> None:
    backend = os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")
    weights = gguf.dequantize(data.reshape(outputs, -1), gguf.GGMLQuantizationType(encoding))
    x = np.random.default_rng(765).normal(size=(rows, inputs)).astype(np.float32)
    expected = x.astype(np.float64) @ weights.astype(np.float64).T
    context = open_context(
        Backend(backend), x.nbytes + data.nbytes + rows * outputs * 4 + 1024**2, 0
    )
    a = context.upload(TensorSpec((rows, inputs), DType.F32), x.tobytes())
    b = context.upload(TensorSpec((data.size,), DType.U8), data.tobytes())
    c = context.upload(
        TensorSpec((rows, outputs), DType.F32),
        np.full((rows, outputs), np.nan, np.float32).tobytes(),
    )
    # A test that wants one schedule names its factory; the candidate table is
    # exercised separately, by tests/kernels/test_selection.py.
    capability = context.capability
    representation = block_layout(encoding)
    if matrix:
        from magnitude_engine.kernels.projection.matrix import projection as matrix_projection

        program = matrix_projection(rows, outputs, inputs, representation, capability=capability)
    elif subgroup:
        from magnitude_engine.kernels.projection.subgroup import projection as subgroup_projection

        program = subgroup_projection(
            rows, outputs, inputs, representation, capability=capability, row_tile=row_tile
        )
    elif packed_k:
        from magnitude_engine.kernels.projection.packed_k import projection as k_projection

        program = k_projection(
            rows, outputs, inputs, representation, capability=capability, row_tile=row_tile
        )
    elif groupwise:
        from magnitude_engine.kernels.projection.blocks import projection as group_projection

        program = group_projection(
            rows, outputs, inputs, representation, capability=capability, row_tile=row_tile
        )
    elif backend == "llvm":
        program = projection_cpu(rows, outputs, inputs, representation, row_tile=row_tile)
    else:
        program = projection(rows, outputs, inputs, representation, row_tile=row_tile)
    kernel = context.compile(program)
    weight_view = b.view(kernel.signature[1])
    ticket = context.submit([Prepared(context, kernel, [a, weight_view, c])])
    weight_view.close()
    a.close()
    b.close()
    actual = np.frombuffer(context.read(c, after=ticket), np.float32).reshape(rows, outputs)
    c.close()
    assert context.allocated_bytes == 0
    context.close()
    # Absolute error scales with the sum of magnitudes, including cancellation.
    # This checks FP32 reduction error without excusing a relative error at zero.
    bound = np.abs(x.astype(np.float64)) @ np.abs(weights.astype(np.float64)).T
    assert np.all(np.abs(actual - expected) <= 2e-6 * bound + 1e-6), (
        np.max(np.abs(actual - expected)),
        np.max(bound),
    )


@pytest.mark.device
@pytest.mark.parametrize("encoding", [Encoding.Q6_K, Encoding.IQ4_XS, Encoding.Q8_0, Encoding.F16])
def test_groupwise_contraction(encoding):
    check_projection(packed(encoding, 5, 768), encoding, 5, 768, 3, groupwise=True, row_tile=2)


@pytest.mark.device
@pytest.mark.parametrize("encoding", tuple(Encoding))
@pytest.mark.parametrize("rows,inputs", [(1, 256), (3, 768)])
def test_projection(encoding: Encoding, rows: int, inputs: int):
    check_projection(packed(encoding, 5, inputs), encoding, 5, inputs, rows)


@pytest.mark.device
@pytest.mark.parametrize("encoding", tuple(Encoding))
@pytest.mark.parametrize("row_tile", [2, 4])
def test_row_reuse_projection(encoding, row_tile):
    check_projection(packed(encoding, 5, 768), encoding, 5, 768, 3, row_tile=row_tile)


@pytest.mark.device
@pytest.mark.parametrize("encoding", tuple(Encoding))
def test_tiled_projection(encoding):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") == "llvm":
        pytest.skip("GPU matrix schedule; LLVM uses its CPU contraction")
    check_projection(packed(encoding, 19, 256), encoding, 19, 256, 17, matrix=True)


@pytest.mark.device
@pytest.mark.parametrize("encoding", tuple(Encoding))
@pytest.mark.parametrize("row_tile", [1, 2, 4])
def test_metal_subgroup_projection(encoding, row_tile):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") != "metal":
        pytest.skip("Metal subgroup schedule")
    check_projection(
        packed(encoding, 5, 768), encoding, 5, 768, 3, subgroup=True, row_tile=row_tile
    )


@pytest.mark.device
@pytest.mark.parametrize("encoding", (Encoding.Q4_K, Encoding.Q5_K))
@pytest.mark.parametrize("row_tile", [1, 2, 4])
def test_metal_packed_k_projection(encoding, row_tile):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") != "metal":
        pytest.skip("Metal packed K schedule")
    check_projection(
        packed(encoding, 5, 768), encoding, 5, 768, 3, packed_k=True, row_tile=row_tile
    )


@pytest.mark.device
@pytest.mark.parametrize("matrix", [False, True])
def test_grouped_compact_projection_layout(matrix):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") != "metal":
        pytest.skip("Metal compact projection schedules")
    from magnitude_engine.kernels.projection.matrix import projection as matrix_projection
    from magnitude_engine.kernels.projection.packed_k import projection as vector_projection

    encoding = Encoding.Q4_K
    rows, widths, inputs = (9 if matrix else 3), (5, 7), 512
    outputs = sum(widths)
    data = packed(encoding, outputs, inputs)
    weights = gguf.dequantize(data.reshape(outputs, -1), gguf.GGMLQuantizationType(encoding))
    values = np.random.default_rng(765).normal(size=(rows, inputs)).astype(np.float32)
    logical = values.astype(np.float64) @ weights.astype(np.float64).T
    expected = np.concatenate((logical[:, : widths[0]].ravel(), logical[:, widths[0] :].ravel()))
    context = open_context(Backend.METAL, data.nbytes + values.nbytes + 2 * 1024**2, 0)
    a = context.upload(TensorSpec(values.shape, DType.F32), values.tobytes())
    b = context.upload(TensorSpec((data.size,), DType.U8), data.tobytes())
    c = context.allocate(TensorSpec((rows, outputs), DType.F32))
    factory = matrix_projection if matrix else vector_projection
    kernel = context.compile(
        factory(rows, widths, inputs, block_layout(encoding), capability=context.capability)
    )
    weight_view = b.view(kernel.signature[1])
    ticket = context.submit((Prepared(context, kernel, (a, weight_view, c)),))
    weight_view.close()
    a.close()
    b.close()
    actual = np.frombuffer(context.read(c, after=ticket), np.float32)
    c.close()
    context.close()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.device
@pytest.mark.parametrize("encoding", (Encoding.Q4_K, Encoding.Q5_K))
def test_hierarchical_fused_gate(encoding):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") != "metal":
        pytest.skip("Metal hierarchical fused schedule")
    from magnitude_engine.kernels.projection.hierarchical_fused import gated_vector

    rows, outputs, inputs = 2, 5, 512
    gate = packed(encoding, outputs, inputs)
    up = packed(encoding, outputs, inputs, seed=285)
    raw = np.concatenate((gate, up))
    wire = gguf.GGMLQuantizationType(encoding)
    gate_weights = gguf.dequantize(gate.reshape(outputs, -1), wire)
    up_weights = gguf.dequantize(up.reshape(outputs, -1), wire)
    values = rounded(
        np.random.default_rng(765).normal(0, 0.1, size=(rows, inputs)).astype(np.float32),
        DType.BF16,
    )
    gate_result = rounded(values @ gate_weights.T, DType.BF16)
    up_result = rounded(values @ up_weights.T, DType.BF16)
    activated = rounded(gate_result / (1 + np.exp(-gate_result)), DType.BF16)
    expected = rounded(activated * up_result, DType.BF16)
    context = open_context(Backend.METAL, raw.nbytes + 2 * 1024**2, 0)
    a = context.upload(TensorSpec(values.shape, DType.BF16), encode(values, DType.BF16))
    b = context.upload(TensorSpec((raw.nbytes // 4,), DType.U32), raw.tobytes())
    c = context.allocate(TensorSpec((rows, outputs), DType.BF16))
    kernel = context.compile(
        gated_vector(
            rows,
            outputs,
            inputs,
            block_layout(encoding),
            capability=context.capability,
            precision=NATIVE_BF16,
        )
    )
    ticket = context.submit((Prepared(context, kernel, (a, b, c)),))
    a.close()
    b.close()
    actual = decode(context.read(c, after=ticket), DType.BF16).reshape(rows, outputs)
    c.close()
    context.close()
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.model
@pytest.mark.device
def test_model_q4_projection():
    artifact = os.environ.get("MAGNITUDE_TEST_GGUF")
    if artifact is None:
        pytest.skip("set MAGNITUDE_TEST_GGUF to qualify the pinned model weights")
    with FileSource(Path(artifact)) as source:
        directory = read_directory(source)
        tensor = directory.tensor("blk.0.ffn_gate.weight")
        data = np.frombuffer(
            source.read(directory.data_offset + tensor.offset, tensor.nbytes), dtype=np.uint8
        )
    check_projection(data, tensor.encoding, *tensor.shape, rows=1)
