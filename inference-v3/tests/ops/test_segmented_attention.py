"""Segmented history and branch masks use one global attention normalization."""

from contextlib import ExitStack

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.kv_codecs import decode_kv_reference, encode_kv_reference
from ops.tensor.ops import _persistent_attention_reference


@pytest.mark.device
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("dtype,width", [(ops.DType.F16, 32), (ops.DType.BF16, 256)])
def test_segmented_shared_prefix_attention(mode, dtype, width):
    rng = np.random.default_rng(44)
    rows, heads = 4, 4
    torch_dtype = getattr(torch, dtype.value)
    tensors = [
        torch.from_numpy(rng.normal(0, 0.3, shape).astype(np.float32)).to(torch_dtype)
        for shape in ((rows, heads, width), (rows, 1, width), (rows, 1, width))
    ]
    q, k, v = (tensor.float().numpy() for tensor in tensors)
    contents = [tensor.view(torch.uint8).numpy().tobytes() for tensor in tensors]
    history_spec = ops.kv_state_spec(24, 1, dtype, ops.default_kv_representation(width, width))
    physical = encode_kv_reference(
        rng.normal(0, 0.3, history_spec.shape).astype(np.float32), history_spec.representation
    )
    logical = decode_kv_reference(physical, history_spec)
    visible = np.array(
        [[0, 5, 8, 2, 0, 1], [0, 5, 12, 3, 1, 2], [0, 0, 0, 0, 0, 0], [0, 5, 20, 1, 3, 1]], np.int32
    )
    expected = _persistent_attention_reference((q, logical, k, v, visible), {"scale": width**-0.5})
    specs = (
        ops.TensorSpec(q.shape, dtype),
        history_spec,
        ops.TensorSpec(k.shape, dtype),
        ops.TensorSpec(v.shape, dtype),
        ops.TensorSpec(visible.shape, ops.DType.I32),
    )
    with (
        ops.DeviceRuntime.open(
            DevicePlan.discover(backend="metal", maximum_bytes=128 << 20)
        ) as device,
        ExitStack() as owned,
    ):
        resources = []
        for spec, data in zip(
            specs, (contents[0], physical, contents[1], contents[2], visible.tobytes()), strict=True
        ):
            resource = device.upload(spec, data)
            owned.callback(resource.close)
            resources.append(resource)
        signature = ops.Signature(
            tuple(
                ops.Argument(
                    spec,
                    ("query", "history", "keys", "values", "visible")[i],
                    ops.ValueKind.RESOURCE if i == 1 else ops.ValueKind.INPUT,
                )
                for i, spec in enumerate(specs)
            )
        )
        program = ops.compile(
            lambda q, h, k, v, r: ops.persistent_attention(q, h, k, v, r, sequence_count=4),
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode=mode),
        )
        owned.callback(program.close)
        execution = program.submit(
            resources[0], *resources[2:], resources={"history": resources[1]}
        )
        for resource in execution.outputs:
            owned.callback(resource.close)
        actual = (
            torch.frombuffer(
                bytearray(device.read(execution.outputs[0], after=execution.completion)),
                dtype=torch_dtype,
            )
            .float()
            .numpy()
            .reshape(q.shape)
        )
        # Match the existing BF16 persistent-attention qualification tolerance.
        rtol, atol = (0.03, 0.003) if dtype == ops.DType.BF16 else (4e-3, 4e-4)
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        assert device.read(resources[1]) == physical
