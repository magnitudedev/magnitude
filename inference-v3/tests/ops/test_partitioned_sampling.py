"""Exact vocabulary partitions, position-addressed draws and bounded scratch."""

from dataclasses import replace

import numpy as np
import pytest

import ops
from engine import DevicePlan
from ops.compiler.lowering import LoweringContext
from ops.kernels.sampling import SamplingRule


def _definition(rows, vocabulary):
    signature = ops.Signature((
        ops.Argument(ops.TensorSpec((rows, vocabulary), ops.DType.F32), "logits"),
        ops.Argument(ops.TensorSpec((rows, 6), ops.DType.U32), "draws"),
    ))
    return ops.trace(ops.sample, signature), signature


def test_sampling_geometry_accounts_for_all_summaries_and_reduction_threads():
    graph, _ = _definition(7, 4099)
    compiler_target = ops.CompilerTarget(32, 96, 1536,
                                     )
    context = LoweringContext(compiler_target, "decode", "model", "test", 1 << 20)
    selected, = SamplingRule().build(graph, 0, context)
    assert selected.emitter.threads == 64  # A non-power-of-two limit cannot break the reduction tree.
    assert selected.emitter.partitions == 9
    assert selected.workspace_bytes == 7 * 9 * 12
    assert selected.kernel_count == 2
    limited, = SamplingRule().build(graph, 0, replace(context, workspace_limit=7 * 3 * 12))
    assert limited.emitter.partitions == 3
    single, = SamplingRule().build(graph, 0, replace(context, workspace_limit=7 * 2 * 12 - 1))
    assert single.kernel_count == 1 and single.workspace == ()


@pytest.mark.device
@pytest.mark.parametrize("workspace", [0, 1 << 20])
def test_sampling_partitions_preserve_rng_ties_failures_and_regrouping(workspace):
    rows, vocabulary = 7, 4099
    graph, signature = _definition(rows, vocabulary)
    rng = np.random.default_rng(193)
    logits = rng.uniform(-2, 2, size=(rows, vocabulary)).astype(np.float32)
    logits[0].fill(-np.inf)
    logits[0, [31, 2048, 4098]] = 3  # Equal winners in different partitions.
    logits[1, ::19] = -np.inf
    logits[2].fill(-np.inf)
    logits[3, -1] = np.nan
    logits[4, -1] = np.inf
    logits[5].fill(-1)
    logits[5, 100], logits[5, 4000] = -0.0, 0.0
    logits[6] = logits[1]
    draws = np.zeros((rows, 6), np.uint32)
    draws[1] = (1, 0xFFFFFFF1, 0xEFFFFFFF, 0xFFFFFFFF, 0xAAAAAAAA, 3)
    draws[6] = draws[1]
    expected = ops.evaluate_reference(graph, {"logits": logits, "draws": draws}).outputs[0]
    assert expected[0].tolist() == [31, 0]
    assert expected[2].tolist() == [-1, 1]
    assert expected[3].tolist() == expected[4].tolist() == [-1, 2]
    assert expected[5].tolist() == [100, 0]
    np.testing.assert_array_equal(expected[1], expected[6])
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=8 << 20)) as device:
        program = ops.compile(ops.sample, signature=signature, device=device, constants={},
                              options=ops.CompileOptions(mode="decode", workspace_limit=workspace))
        try:
            # Reuse one compiled shape while moving complete logical draws with
            # their rows. This must not change RNG counters or selected tokens.
            for order in (np.arange(rows), np.asarray([6, 2, 4, 0, 1, 5, 3])):
                left = device.upload(signature.args[0].spec, logits[order].tobytes())
                right = device.upload(signature.args[1].spec, draws[order].tobytes())
                execution = None
                try:
                    execution = program.submit(left, right)
                    actual = np.frombuffer(device.read(execution.outputs[0], after=execution.completion),
                                            np.int32).reshape(rows, 2)
                    np.testing.assert_array_equal(actual, expected[order])
                finally:
                    if execution is not None:
                        for output in execution.outputs:
                            output.close()
                    right.close()
                    left.close()
        finally:
            program.close()
