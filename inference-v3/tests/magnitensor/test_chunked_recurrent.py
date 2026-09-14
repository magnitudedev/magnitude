"""Gate coverage for matrix-chunk recurrence and its state boundaries."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, select_cover


def _program(rows, batch, dtype, mapping):
    specs = (
        mt.TensorSpec((rows, 2, 128), dtype),
        mt.TensorSpec((rows, 2, 128), dtype),
        mt.TensorSpec((rows, 4, 35), dtype),
        mt.TensorSpec((rows, 4), mt.DType.F32),
        mt.TensorSpec((rows, 4), dtype),
        mt.TensorSpec((batch, 4, 35, 128), mt.DType.F32),
        mt.TensorSpec((batch + 1,), mt.DType.I32),
    )
    signature = mt.Signature(
        tuple(
            mt.Argument(spec, f"v{i}", mt.ValueKind.RESOURCE if i == 5 else mt.ValueKind.INPUT)
            for i, spec in enumerate(specs)
        )
    )

    def function(q, k, v, decay, beta, state, offsets):
        return mt.gated_delta_recurrence(q, k, v, decay, beta, state, offsets, mapping=mapping)

    return function, signature, specs


def test_chunked_recurrence_selection_accounts_for_workspace_and_capabilities():
    function, signature, _ = _program(193, 3, mt.DType.F32, "tiled")
    graph = mt.trace(function, signature)
    capabilities = mt.Capabilities(
        32,
        256,
        32768,
        matrix_instructions=(mt.MatrixInstruction(8, 8, 8, mt.DType.F32, mt.DType.F32),),
        memory_scopes=frozenset({"global", "shared", "local"}),
        native_multi_launch=True,
    )
    context = LoweringContext(capabilities, "prefill", "model", "test", 8 << 20)

    def selected(ctx):
        return select_cover(graph, mt.lowerings.enumerate(graph, ctx)).candidates[0]

    candidate = selected(context)
    assert candidate.name == "gated_delta.chunked-matrix@0"
    assert candidate.kernel_count == 2
    assert candidate.workspace_bytes == 3 * 7 * 4 * 2 * (32 * 32 + 32) * 4
    assert selected(replace(context, mode="decode")).name == "gated_delta.register-state@0"
    assert selected(replace(context, workspace_limit=1)).name == "gated_delta.register-state@0"
    assert (
        selected(replace(context, capabilities=replace(capabilities, matrix_instructions=()))).name
        == "gated_delta.register-state@0"
    )


@pytest.mark.device
@pytest.mark.parametrize("dtype,mapping", [(mt.DType.F32, "tiled"), (mt.DType.BF16, "grouped")])
def test_chunked_recurrence_tails_resets_and_empty_sequence(dtype, mapping):
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
    arrays[3][[0, 31, 32, 64, 192]] = 0  # Resets on and inside chunk boundaries.
    arrays[4] = rng.uniform(0, 1, (rows, 4)).astype(np.float32)
    arrays.append(np.asarray([0, 65, 65, 193], dtype=np.int32))
    natives = [
        torch.from_numpy(array).to(
            torch.bfloat16
            if spec.dtype == mt.DType.BF16
            else torch.int32
            if spec.dtype == mt.DType.I32
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
    device = mt.device("metal", budget_bytes=16 << 20)
    resources = []
    compiled = execution = None
    try:
        for spec, native in zip(specs, natives, strict=True):
            payload = (
                native.view(torch.uint16).numpy().tobytes()
                if spec.dtype == mt.DType.BF16
                else native.numpy().tobytes()
            )
            resources.append(device.upload(spec, payload))
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=mt.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.submissions == (("gated_delta.chunked-matrix@0",),)
        execution = compiled.submit(*resources[:5], resources[6], resources={"v5": resources[5]})
        execution.completion.wait()
        actual_output, actual_state = [
            out.native.float().cpu().numpy() for out in execution.outputs
        ]
        np.testing.assert_allclose(
            actual_output, expected, rtol=8e-3 if dtype == mt.DType.BF16 else 3e-4, atol=3e-5
        )
        np.testing.assert_allclose(actual_state, state, rtol=3e-4, atol=3e-5)
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
