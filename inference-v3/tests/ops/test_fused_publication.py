"""Fused regions retain observable logical publication boundaries."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from tests.ops import composed_formulas


@pytest.mark.device
@pytest.mark.parametrize(
    "kind",
    [
        "attention",
        "attention-partitioned",
        "recurrent",
        "recurrent-decode",
        "dense",
        "dense-decode",
    ],
)
def test_bfloat_fusion_keeps_intermediate_publications(kind):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rows = 1 if kind.endswith("decode") else 9
    width, heads = 128, 4
    channels = width * heads
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=32 << 20))
    resources = []
    compiled = execution = None
    try:
        codes = np.eye(channels, dtype=np.uint8)
        packed = codes.ravel()[::2] | (codes.ravel()[1::2] << 4)
        groups = codes.size // 64
        weight_spec = ops.TensorSpec(codes.shape, ops.DType.BF16).with_representation(
            ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))
        )
        weight = device.upload(
            weight_spec,
            packed.tobytes()
            + np.full(groups, 0x3F80, np.uint16).tobytes()
            + np.zeros(groups, np.uint16).tobytes(),
        )
        resources.append(weight)
        function: Callable[..., ops.Tensor]
        if kind.startswith("attention"):
            capacity = 8192 if kind.endswith("partitioned") else 16
            q = torch.zeros((rows, heads, width), dtype=torch.bfloat16)
            history = torch.zeros((2, capacity, 1, width), dtype=torch.bfloat16)
            history[1, 0] = 1.0
            history[1, 1] = 1.0078125
            visible = torch.tensor([[0, 2]] * rows, dtype=torch.int32)
            gate = torch.full_like(q, 0.4)
            arrays = (q, history, visible, gate)

            function = composed_formulas.attention_output

            attended = ((history[1, 0].float() + history[1, 1].float()) / 2).bfloat16()
            expected = (attended * torch.sigmoid(gate.float()).bfloat16()).reshape(rows, channels)
            family = "attention.matrix-streaming-gated-output"
        elif kind.startswith("recurrent"):
            mixed = torch.linspace(0.5, 1.5, width).bfloat16().repeat(rows, heads, 1)
            norm = torch.ones(width, dtype=torch.bfloat16)
            gate = torch.full((rows, channels), 0.4, dtype=torch.bfloat16)
            arrays = (mixed, norm, gate)

            function = composed_formulas.recurrent_output

            normalized = (
                mixed.float() * torch.rsqrt(mixed.float().square().mean(-1, keepdim=True) + 1e-6)
            ).bfloat16()
            activated_gate = (gate.float() * torch.sigmoid(gate.float())).bfloat16()
            expected = normalized.reshape(rows, channels) * activated_gate
            family = "recurrent.output-"
        else:
            value = torch.full((rows, channels), 0.4, dtype=torch.bfloat16)
            arrays = (value,)

            def dense_function(value, w):
                return composed_formulas.dense_feedforward(value, w, w, w)

            function = dense_function

            expected = (value.float() * torch.sigmoid(value.float())).bfloat16() * value
            family = "dense_swiglu.packet-"

        inputs = []
        for array in arrays:
            dtype = ops.DType.I32 if array.dtype == torch.int32 else ops.DType.BF16
            payload = (
                array.numpy().tobytes()
                if dtype == ops.DType.I32
                else array.view(torch.uint16).numpy().tobytes()
            )
            resource = device.upload(ops.TensorSpec(tuple(array.shape), dtype), payload)
            inputs.append(resource)
            resources.append(resource)
        signature = ops.Signature(
            tuple(
                ops.Argument(
                    value.spec,
                    f"x{i}",
                    ops.ValueKind.RESOURCE
                    if kind.startswith("attention") and i == 1
                    else ops.ValueKind.INPUT,
                )
                for i, value in enumerate(inputs)
            )
            + (ops.Argument(weight.spec, "w", ops.ValueKind.CONSTANT),)
        )
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={"w": weight},
            options=ops.CompileOptions(mode="decode" if kind.endswith("decode") else "prefill"),
        )
        assert any(
            c.name.startswith(family) for c in compiled.diagnostics.operations
        )
        execution = (
            compiled.submit(inputs[0], *inputs[2:], resources={"x1": inputs[1]})
            if kind.startswith("attention")
            else compiled.submit(*inputs)
        )
        execution.completion.wait()
        torch.testing.assert_close(execution.outputs[0].native.cpu(), expected, rtol=0, atol=0)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
def test_float_residual_and_bfloat_normalized_publications_share_one_region():
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    left = torch.ones((2, 128), dtype=torch.float32)
    update = torch.full((2, 128), 0.003, dtype=torch.bfloat16)
    gain = torch.ones(128, dtype=torch.bfloat16)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    resources = []
    compiled = execution = None
    try:
        for array in (left, update, gain):
            dtype = ops.DType.F32 if array.dtype == torch.float32 else ops.DType.BF16
            payload = (
                array.numpy().tobytes()
                if dtype == ops.DType.F32
                else array.view(torch.uint16).numpy().tobytes()
            )
            resources.append(device.upload(ops.TensorSpec(tuple(array.shape), dtype), payload))

        compiled = ops.compile(
            composed_formulas.residual_normalization,
            signature=ops.Signature(
                tuple(ops.Argument(value.spec, f"v{i}") for i, value in enumerate(resources))
            ),
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.dispatches == 1
        assert any(
            c.name.startswith("residual_rms.fused")
            for c in compiled.diagnostics.operations
        )
        execution = compiled.submit(*resources)
        execution.completion.wait()
        residual = left + update.float()
        expected = (
            residual * torch.rsqrt(residual.square().mean(-1, keepdim=True) + 1e-6)
        ).bfloat16()
        torch.testing.assert_close(execution.outputs[0].native.cpu(), residual, rtol=0, atol=0)
        torch.testing.assert_close(execution.outputs[1].native.cpu(), expected, rtol=0, atol=0)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()
