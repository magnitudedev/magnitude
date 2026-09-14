"""Gate coverage for matrix-chunk recurrence and its state boundaries."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch

import ops
from ops.operation import build_operations
from engine import DevicePlan
from ops.compiler.lowering import LoweringContext


def _program(rows, batch, dtype, mapping):
    specs = (
        ops.TensorSpec((rows, 2, 128), dtype),
        ops.TensorSpec((rows, 2, 128), dtype),
        ops.TensorSpec((rows, 4, 35), dtype),
        ops.TensorSpec((rows, 4), ops.DType.F32),
        ops.TensorSpec((rows, 4), dtype),
        ops.TensorSpec((batch, 4, 35, 128), ops.DType.F32),
        ops.TensorSpec((batch + 1,), ops.DType.I32),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, f"v{i}", ops.ValueKind.RESOURCE if i == 5 else ops.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(q, k, v, decay, beta, state, offsets):
        return ops.gated_delta_recurrence(q, k, v, decay, beta, state, offsets, mapping=mapping)

    return function, signature, specs


def test_chunked_recurrence_selection_accounts_for_workspace_and_capabilities():
    function, signature, _ = _program(193, 3, ops.DType.F32, "tiled")
    graph = ops.trace(function, signature)
    capabilities = ops.Capabilities(
        32,
        256,
        32768,
        matrix_instructions=(ops.MatrixInstruction(8, 8, 8, ops.DType.F32, ops.DType.F32),),
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 8 << 20)

    def selected(ctx):
        return build_operations(graph, ctx)[0]

    candidate = selected(context)
    assert candidate.name == "gated_delta.chunked-matrix@0"
    assert candidate.kernel_count == 3
    assert candidate.aliases == ()
    assert candidate.workspace_bytes == 3 * 7 * 4 * (32 * 32 + 2 * 32 + 32 * 128 + 32 * 35 + 35 * 128) * 4
    assert selected(replace(context, workspace_limit=candidate.workspace_bytes)).kernel_count == 3
    assert selected(replace(context, workspace_limit=candidate.workspace_bytes - 1)).name == "gated_delta.register-state@0"
    decoded = selected(replace(context, mode="decode"))
    assert decoded.name == "gated_delta.register-state@0"
    assert decoded.aliases == ()
    assert selected(replace(context, workspace_limit=1)).name == "gated_delta.register-state@0"
    assert (
        selected(replace(context, capabilities=replace(capabilities, matrix_instructions=()))).name
        == "gated_delta.register-state@0"
    )


@pytest.mark.device
@pytest.mark.parametrize("dtype,mapping", [(ops.DType.F32, "tiled"), (ops.DType.BF16, "grouped")])
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.parametrize("chunked", [True, False])
def test_chunked_recurrence_tails_resets_and_empty_sequence(dtype, mapping, reset, chunked):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rows, batch = 193, 3
    function, signature, specs = _program(rows, batch, dtype, mapping)
    rng = np.random.default_rng(139)
    arrays: list[np.ndarray] = [
        rng.normal(0, 0.1, cast(tuple[int, ...], spec.shape)).astype(np.float32)
        for spec in specs[:-1]
    ]
    arrays[3] = rng.uniform(0.85, 1, (rows, 4)).astype(np.float32)
    if reset:
        arrays[3][[0, 31, 32, 64, 192]] = 0  # Resets on and inside chunk boundaries.
    else:
        arrays[3].fill(1)  # No reset/decay may hide an incorrect carried chunk state.
    arrays[4] = rng.uniform(0, 1, (rows, 4)).astype(np.float32)
    arrays.append(np.asarray([0, 65, 65, 193], dtype=np.int32))
    natives = [
        torch.from_numpy(array).to(
            torch.bfloat16
            if spec.dtype == ops.DType.BF16
            else torch.int32
            if spec.dtype == ops.DType.I32
            else torch.float32
        )
        for spec, array in zip(specs, arrays, strict=True)
    ]
    # Independent sequential FP64 equations, starting from exact stored operands.
    q, k, v, decay, beta, initial, offsets = [x.double().numpy() for x in natives]
    state = initial.copy()
    expected = np.empty((rows, 4, 35), dtype=np.float64)
    for sequence in range(batch):
        for row in range(int(offsets[sequence]), int(offsets[sequence + 1])):
            for head in range(4):
                kh = head % 2 if mapping == "tiled" else head // 2
                state[sequence, head] *= decay[row, head]
                residual = beta[row, head] * (v[row, head] - state[sequence, head] @ k[row, kh])
                state[sequence, head] += residual[:, None] * k[row, kh]
                expected[row, head] = state[sequence, head] @ q[row, kh]
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=16 << 20))
    resources = []
    compiled = execution = None
    try:
        for spec, native in zip(specs, natives, strict=True):
            payload = (
                native.view(torch.uint16).numpy().tobytes()
                if spec.dtype == ops.DType.BF16
                else native.numpy().tobytes()
            )
            resources.append(device.upload(spec, payload))
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill", workspace_limit=None if chunked else 0),
        )
        selected = "chunked-matrix" if chunked else "register-state"
        assert compiled.diagnostics.submissions == ((f"gated_delta.{selected}@0",),)
        execution = compiled.submit(*resources[:5], resources[6], resources={"v5": resources[5]})
        execution.completion.wait()
        actual_output, actual_state = [
            out.native.float().cpu().numpy() for out in execution.outputs
        ]
        np.testing.assert_allclose(
            actual_output, expected, rtol=8e-3 if dtype == ops.DType.BF16 else 3e-4, atol=3e-5
        )
        np.testing.assert_allclose(actual_state, state, rtol=3e-4, atol=3e-5)
        np.testing.assert_array_equal(resources[5].native.cpu().numpy(), initial.astype(np.float32))
        np.testing.assert_array_equal(actual_state[1], initial[1].astype(np.float32))
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()
