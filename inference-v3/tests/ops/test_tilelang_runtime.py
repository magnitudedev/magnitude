from __future__ import annotations

from typing import cast

import gguf
import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.runtime.tilelang import TileLangRuntime
from engine.weights.descriptor import StoredQuantized, WeightDescriptor
from engine.weights.formats.gguf import Encoding, quantization
from engine.weights.identity import ArtifactIdentity
from engine.weights.tensor_residency import TensorWeights


class _QuantizedFormat:
    identity = ArtifactIdentity("0" * 64)

    def __init__(self, content: bytes, encoding: Encoding):
        self.source = ops.MemorySource(content)
        self.encoding = encoding

    def stored(self, descriptor):
        representation, codec = quantization(self.encoding)
        return StoredQuantized(representation, codec, self.source, 0)

    def close(self):
        pass


@pytest.mark.device
def test_metal_reshape_preserves_every_element():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal reshape regression check requires MPS")
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    source_spec = ops.TensorSpec((2, 3, 4), ops.DType.F16)
    expected = np.arange(source_spec.elements, dtype=np.float16).reshape(
        cast(tuple[int, ...], source_spec.shape)
    )
    source = device.upload(source_spec, expected.tobytes())
    compiled = execution = None
    try:
        compiled = ops.compile(
            lambda value: ops.reshape(value, (4, 6)),
            signature=ops.Signature((ops.Argument(source_spec, "source"),)),
            device=device,
            constants={},
            options=ops.CompileOptions(mode="decode", precision="reference"),
        )
        execution = compiled.submit(source)
        execution.completion.wait()
        np.testing.assert_array_equal(
            execution.outputs[0].native.cpu().numpy(), expected.reshape(4, 6)
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        source.close()
        device.close()


@pytest.mark.device
def test_metal_runtime_composes_kernels_and_binds_static_arguments_natively():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal sanity check requires MPS")

    runtime = TileLangRuntime("metal")
    device = ops.DeviceRuntime(runtime, budget_bytes=1 << 20)
    assert "gemm.runtime_valid_m" in device.capabilities.features
    hidden_spec = ops.TensorSpec((2, 8), ops.DType.F16)
    weight_spec = ops.TensorSpec((8, 8), ops.DType.F16)
    hidden_host = torch.randn(cast(tuple[int, ...], hidden_spec.shape), dtype=torch.float16)
    weight_host = torch.randn(cast(tuple[int, ...], weight_spec.shape), dtype=torch.float16)
    hidden = device.allocate(hidden_spec)
    hidden.native.copy_(hidden_host.to("mps"))
    weight = device.upload(weight_spec, weight_host.numpy().tobytes())

    compiled = ops.compile(
        lambda value, matrix: ops.silu(ops.linear(value, matrix, output_dtype=ops.DType.F16)),
        signature=ops.Signature(
            (
                ops.Argument(hidden_spec, "hidden"),
                ops.Argument(weight_spec, "weight", ops.ValueKind.CONSTANT),
            )
        ),
        device=device,
        constants={"weight": weight},
        options=ops.CompileOptions(mode="decode"),
    )
    assert len(compiled.diagnostics.submissions) == 1
    assert len(compiled.diagnostics.submissions[0]) == 2

    execution = compiled.submit(hidden)
    execution.completion.wait()
    expected = torch.nn.functional.silu(hidden_host.float() @ weight_host.float().T)
    torch.testing.assert_close(
        execution.outputs[0].native.cpu().float(), expected, atol=2e-2, rtol=2e-2
    )

    for output in execution.outputs:
        output.close()
    compiled.close()
    hidden.close()
    weight.close()
    device.close()


@pytest.mark.device
def test_metal_sampling_matches_the_semantic_reference():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal sampling check requires MPS")
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    logits_spec = ops.TensorSpec((4, 4), ops.DType.F32)
    draws_spec = ops.TensorSpec((4, 6), ops.DType.U32)
    logits_host = np.asarray(
        [
            [1.0, 3.0, 3.0, -np.inf],
            [0.0, 0.0, 0.0, 0.0],
            [-np.inf, -np.inf, -np.inf, -np.inf],
            [0.0, np.nan, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    draws_host = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 17, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    logits = device.upload(logits_spec, logits_host.tobytes())
    draws = device.upload(draws_spec, draws_host.tobytes())
    compiled = ops.compile(
        lambda values, random: ops.sample(values, random),
        signature=ops.Signature(
            (ops.Argument(logits_spec, "logits"), ops.Argument(draws_spec, "draws"))
        ),
        device=device,
        constants={},
        options=ops.CompileOptions(mode="decode"),
    )
    execution = compiled.submit(logits, draws)
    actual = np.frombuffer(
        device.read(execution.outputs[0], after=execution.completion), dtype=np.int32
    ).reshape(4, 2)
    reference = ops.evaluate_reference(
        compiled.graph, {"logits": logits_host, "draws": draws_host}
    ).outputs[0]
    assert np.array_equal(actual, reference)
    for output in execution.outputs:
        output.close()
    compiled.close()
    draws.close()
    logits.close()
    device.close()


@pytest.mark.device
def test_metal_q8_residency_import_produces_canonical_packed_storage():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal residency check requires MPS")
    scales = (np.float16(0.25).tobytes(), np.float16(-0.5).tobytes())
    codes = (
        np.arange(-16, 16, dtype=np.int8).tobytes(),
        np.arange(15, -17, -1, dtype=np.int8).tobytes(),
    )
    wire = b"".join(scale + code for scale, code in zip(scales, codes, strict=True))
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    weights = TensorWeights(_QuantizedFormat(wire, Encoding.Q8_0), device)
    resident = device.resolve(weights.bind(WeightDescriptor(name="weight", shape=(2, 32)), ops.DType.F16))
    assert device.read(resident) == b"".join(codes) + b"".join(scales)
    resident.close()
    weights.close()
    device.close()


def _packed(encoding: Encoding, outputs: int, inputs: int) -> np.ndarray:
    rng = np.random.default_rng(284)
    blocks = outputs * inputs // encoding.block_elements
    data = rng.integers(0, 256, (blocks, encoding.block_bytes), dtype=np.uint8)
    offsets = (
        (0, 2)
        if encoding in (Encoding.Q4_K, Encoding.Q5_K)
        else ((208,) if encoding == Encoding.Q6_K else (0,))
    )
    for offset in offsets:
        scales = rng.uniform(-0.02, 0.02, blocks).astype(np.float16)
        data[:, offset : offset + 2] = scales.view(np.uint8).reshape(-1, 2)
    return data.ravel()


@pytest.mark.device
@pytest.mark.parametrize(
    "encoding",
    (Encoding.Q4_K, Encoding.Q5_K, Encoding.Q6_K, Encoding.Q8_0),
)
@pytest.mark.parametrize("rows,floating,mode", (
    (1, ops.DType.BF16, "decode"),
    (2, ops.DType.F32, "prefill"),
    (9, ops.DType.BF16, "prefill"),
    (9, ops.DType.F32, "prefill"),
))
def test_metal_quantized_import_and_projection_match_gguf(encoding, rows, floating, mode):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal encoded projection check requires MPS")
    outputs, inputs = 2, 256
    packed = _packed(encoding, outputs, inputs)
    expected_weight = gguf.dequantize(
        packed.reshape(outputs, -1), gguf.GGMLQuantizationType(encoding)
    )
    host = np.random.default_rng(765).normal(size=(rows, inputs)).astype(np.float32)
    if floating == ops.DType.BF16:
        tensor = torch.from_numpy(host).to(torch.bfloat16)
        payload = tensor.view(torch.uint16).numpy().tobytes()
        host = tensor.float().numpy()
    else:
        payload = host.tobytes()
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=4 << 20))
    weights = TensorWeights(_QuantizedFormat(packed.tobytes(), encoding), device)
    weight = device.resolve(weights.bind(
        WeightDescriptor(name="weight", shape=(outputs, inputs)),
        ops.DType.BF16 if floating == ops.DType.BF16 else ops.DType.F16,
    ))
    source_spec = ops.TensorSpec(host.shape, floating)
    source = device.upload(source_spec, payload)
    compiled = ops.compile(
        lambda value, matrix: ops.linear(value, matrix, output_dtype=ops.DType.F32),
        signature=ops.Signature(
            (
                ops.Argument(source_spec, "source"),
                ops.Argument(weight.spec, "weight", ops.ValueKind.CONSTANT),
            )
        ),
        device=device,
        constants={"weight": weight},
        options=ops.CompileOptions(mode=mode),
    )
    execution = compiled.submit(source)
    actual = np.frombuffer(
        device.read(execution.outputs[0], after=execution.completion), np.float32
    ).reshape(rows, outputs)
    expected = host.astype(np.float64) @ expected_weight.astype(np.float64).T
    bound = np.abs(host.astype(np.float64)) @ np.abs(expected_weight.astype(np.float64)).T
    assert np.all(np.abs(actual - expected) <= 2e-5 * bound + 1e-5)
    for output in execution.outputs:
        output.close()
    compiled.close()
    source.close()
    weight.close()
    weights.close()
    device.close()
