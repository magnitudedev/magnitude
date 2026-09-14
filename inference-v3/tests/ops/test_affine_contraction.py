"""Coefficient-group contraction eligibility and bounded source publication."""

from dataclasses import replace

import numpy as np
import pytest

import ops
from engine import DevicePlan
from ops.compiler.lowering import LoweringContext
from ops.kernels.matrix import (_packed_vector_geometry, _packet_matrix_instruction,
                                _packet_reduction_width, matrix_geometry)
from ops.kernels.packed import affine_shared_bytes, packet_format
from ops.kernels.grouped_experts import _continuous_contraction
from tests.ops.test_attention_normalization_lowering import CAPABILITIES


def test_continuous_operand_tiles_require_native_precision_and_bounded_layout():
    instruction = ops.MatrixInstruction(8, 8, 8, ops.DType.BF16, ops.DType.F32)
    context = LoweringContext(replace(CAPABILITIES, features=CAPABILITIES.features | {"gemm.shared_instruction_tiles"}),
                              "prefill", "model", "test", 1 << 20)
    assert _continuous_contraction(context, ops.DType.BF16, instruction)
    assert not _continuous_contraction(context, ops.DType.F16, instruction)
    assert not _continuous_contraction(replace(context, capabilities=CAPABILITIES), ops.DType.BF16, instruction)
    assert affine_shared_bytes(32, 64, 32, ops.DType.BF16, decoded=True) == 6144


def test_direct_group_packets_do_not_claim_unsupported_signed_nibbles():
    spec = ops.TensorSpec((32, 256), ops.DType.BF16).with_representation(
        ops.Affine(ops.Code(4, interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),
                   32, ops.DirectCoefficients(ops.DType.F32)))
    assert packet_format(spec) is None


@pytest.mark.parametrize("outputs,capacity,threads", [
    (1, 128, 32), (3, 128, 32), (17, 128, 64), (17, 64, 64), (17, 32, 32), (17, 31, None),
])
def test_packet_vector_geometry_respects_device_limits(outputs, capacity, threads):
    context = LoweringContext(replace(CAPABILITIES, threads_per_group=capacity),
                              "decode", "model", "test", 1 << 20)
    spec = ops.TensorSpec((outputs, 512), ops.DType.BF16).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16)))
    assert _packed_vector_geometry(spec, context) == (None if threads is None else (threads, 4))


def test_matrix_input_precision_does_not_round_quantization_coefficients():
    context = LoweringContext(CAPABILITIES, "prefill", "model", "test", 1 << 20)
    assert _packet_matrix_instruction(context, ops.DType.F16).input_dtype == ops.DType.F16
    assert _packet_matrix_instruction(context, ops.DType.F32).input_dtype == ops.DType.F32
    unsupported = replace(context, capabilities=replace(CAPABILITIES, matrix_instructions=(
        ops.MatrixInstruction(8, 8, 8, ops.DType.F16, ops.DType.F16),)))
    assert _packet_matrix_instruction(unsupported, ops.DType.F16) is None


def test_shared_capacity_includes_coefficient_pairs_and_group_boundaries():
    context = LoweringContext(replace(CAPABILITIES, shared_memory_bytes=4096),
                              "prefill", "model", "test", 1 << 20)
    instruction = _packet_matrix_instruction(context, ops.DType.F16)
    bm, bn, _ = matrix_geometry(context, instruction, 2048, 2560, 64, coefficient_bytes=8)
    assert affine_shared_bytes(bm, bn, 64, instruction.input_dtype) <= 4096
    specs = tuple(ops.TensorSpec((32, 512), ops.DType.F16).with_representation(
        ops.Affine(ops.Code(4), group, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16)))
        for group in (16, 32, 64))
    assert _packet_reduction_width(*specs) == 16


@pytest.mark.device
@pytest.mark.parametrize("rows", [1, 9])
def test_streamed_affine_matrix_covers_offset_tails_without_invocation_compilation(rows):
    from ops.runtime.observation import Activity

    width, outputs = 512, 257
    values = ((np.arange(rows * width).reshape(rows, width) % 13 - 6) / 16).astype(np.float16)
    codes = (np.arange(outputs * width).reshape(outputs, width) % 16).astype(np.uint8)
    packed = (codes.ravel()[::2] | codes.ravel()[1::2] << 4).tobytes()
    groups = codes.size // 64
    scales = np.full(groups, 0x3B40, np.uint16)  # 3/1024, not a power of two
    biases = np.full(groups, 0xBCB4, np.uint16)  # -22.5/1024
    contents = (packed, scales.tobytes(), biases.tobytes())
    planes = tuple(ops.SourcePlane(ops.SourceSpan(ops.MemorySource(content), 0, len(content)), group, size)
                   for content, group, size in zip(contents, (2, 64, 64), (1, 2, 2), strict=True))
    spec = ops.TensorSpec((outputs, width), ops.DType.F16).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16)))
    binding = ops.Binding(spec, "streamed-affine-tail", ops.Residency.STREAMED, planes, ops.CanonicalImport())
    decoded = codes.astype(np.float32) * (3 / 1024) - 22.5 / 1024
    expected = (values.astype(np.float32) @ decoded.T).astype(np.float16)
    signature = ops.Signature((ops.Argument(ops.TensorSpec(values.shape, ops.DType.F16), "x"),
                               ops.Argument(spec, "weight", ops.ValueKind.CONSTANT)))
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=128 << 10)) as device:
        source = device.upload(signature.args[0].spec, values.tobytes())
        program = None
        try:
            program = ops.compile(ops.linear, signature=signature, device=device,
                                  constants={"weight": binding},
                                  options=ops.CompileOptions(mode="decode" if rows == 1 else "prefill"))
            with device.observe() as observed:
                execution = program.submit(source)
                execution.completion.wait()
            try:
                actual = np.frombuffer(device.read(execution.outputs[0]), np.float16).reshape(expected.shape)
                np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
                reads = [item for item in observed.result.activities if item.kind == Activity.SOURCE_READ]
                assert sum(item.bytes_completed for item in reads) == sum(map(len, contents))
                assert len(reads) > len(planes)
                assert not any(item.kind == Activity.COMPILE for item in observed.result.activities)
            finally:
                for output in execution.outputs:
                    output.close()
        finally:
            if program is not None:
                program.close()
            source.close()
