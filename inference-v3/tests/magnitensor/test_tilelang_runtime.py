from __future__ import annotations

import gguf
import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.runtime.tilelang import TileLangRuntime
from magnitude_engine.weights.descriptor import StoredQuantized, WeightDescriptor
from magnitude_engine.weights.formats.gguf import Encoding, quantization
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.tensor_residency import TensorWeights


class _Bytes:
    def __init__(self, content: bytes):
        self.content = content
        self.size = len(content)

    def read(self, offset: int, length: int) -> bytes:
        return self.content[offset : offset + length]


class _QuantizedFormat:
    identity = ArtifactIdentity("0" * 64)

    def __init__(self, content: bytes, encoding: Encoding):
        self.source = _Bytes(content)
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
    device = mt.device("metal", budget_bytes=1 << 20)
    source_spec = mt.TensorSpec((2, 3, 4), mt.DType.F16)
    expected = np.arange(source_spec.elements, dtype=np.float16).reshape(source_spec.shape)
    source = device.upload(source_spec, expected.tobytes())
    compiled = execution = None
    try:
        compiled = mt.compile(
            lambda value: mt.reshape(value, (4, 6)),
            signature=mt.Signature((mt.Argument(source_spec, "source"),)),
            device=device,
            constants={},
            options=mt.CompileOptions(mode="decode"),
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
    device = mt.Device(runtime, budget_bytes=1 << 20)
    assert "gemm.runtime_valid_m" in device.capabilities.features
    hidden_spec = mt.TensorSpec((2, 8), mt.DType.F16)
    weight_spec = mt.TensorSpec((8, 8), mt.DType.F16)
    hidden_host = torch.randn(hidden_spec.shape, dtype=torch.float16)
    weight_host = torch.randn(weight_spec.shape, dtype=torch.float16)
    hidden = device.allocate(hidden_spec)
    hidden.native.copy_(hidden_host.to("mps"))
    weight = device.upload(weight_spec, weight_host.numpy().tobytes())

    compiled = mt.compile(
        lambda value, matrix: mt.silu(mt.linear(value, matrix, output_dtype=mt.DType.F16)),
        signature=mt.Signature(
            (
                mt.Argument(hidden_spec, "hidden"),
                mt.Argument(weight_spec, "weight", mt.ValueKind.CONSTANT),
            )
        ),
        device=device,
        constants={"weight": weight},
        options=mt.CompileOptions(mode="decode"),
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
    device = mt.device("metal", budget_bytes=1 << 20)
    logits_spec = mt.TensorSpec((4, 4), mt.DType.F32)
    draws_spec = mt.TensorSpec((4, 6), mt.DType.U32)
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
    compiled = mt.compile(
        lambda values, random: mt.sample(values, random),
        signature=mt.Signature(
            (mt.Argument(logits_spec, "logits"), mt.Argument(draws_spec, "draws"))
        ),
        device=device,
        constants={},
        options=mt.CompileOptions(mode="decode"),
    )
    execution = compiled.submit(logits, draws)
    actual = np.frombuffer(
        device.read(execution.outputs[0], after=execution.completion), dtype=np.int32
    ).reshape(4, 2)
    reference = mt.evaluate_reference(
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
    device = mt.device("metal", budget_bytes=1 << 20)
    weights = TensorWeights(_QuantizedFormat(wire, Encoding.Q8_0), device)
    resident = weights.resident(WeightDescriptor(name="weight", shape=(2, 32)), mt.DType.F16)
    assert device.read(resident) == b"".join(codes) + b"".join(scales)
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
    (Encoding.Q4_K, Encoding.Q5_K, Encoding.Q6_K, Encoding.Q8_0, Encoding.IQ4_XS),
)
def test_metal_quantized_import_and_projection_match_gguf(encoding):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal encoded projection check requires MPS")
    outputs, inputs, rows = 2, 256, 2
    packed = _packed(encoding, outputs, inputs)
    expected_weight = gguf.dequantize(
        packed.reshape(outputs, -1), gguf.GGMLQuantizationType(encoding)
    )
    host = np.random.default_rng(765).normal(size=(rows, inputs)).astype(np.float32)
    device = mt.device("metal", budget_bytes=4 << 20)
    weights = TensorWeights(_QuantizedFormat(packed.tobytes(), encoding), device)
    weight = weights.resident(
        WeightDescriptor(name="weight", shape=(outputs, inputs)), mt.DType.F16
    )
    source_spec = mt.TensorSpec(host.shape, mt.DType.F32)
    source = device.upload(source_spec, host.tobytes())
    compiled = mt.compile(
        lambda value, matrix: mt.linear(value, matrix, output_dtype=mt.DType.F32),
        signature=mt.Signature(
            (
                mt.Argument(source_spec, "source"),
                mt.Argument(weight.spec, "weight", mt.ValueKind.CONSTANT),
            )
        ),
        device=device,
        constants={"weight": weight},
        options=mt.CompileOptions(mode="prefill"),
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
    weights.close()
    device.close()
