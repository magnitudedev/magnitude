"""Packed mask selection, mixed rows, and preservation of invalid-source policy."""

import numpy as np
import pytest

import ops


def definition(rows, vocabulary, masked_rows):
    signature = ops.Signature(
        (
            ops.Argument(ops.TensorSpec((rows, vocabulary), ops.DType.F32), "logits"),
            ops.Argument(ops.TensorSpec((rows, 6), ops.DType.U32), "draws"),
            ops.Argument(
                ops.TensorSpec((masked_rows, (vocabulary + 31) // 32), ops.DType.U32), "masks"
            ),
            ops.Argument(ops.TensorSpec((rows,), ops.DType.I32), "mask_rows"),
        )
    )
    return ops.trace(ops.sample_constrained, signature), signature


def fixture(vocabulary=67):
    logits = np.zeros((6, vocabulary), np.float32)
    logits[:, -1] = 100
    logits[:, 31] = 4
    logits[:, 32] = 8
    logits[4, -1] = np.nan
    logits[5, -1] = np.inf
    draws = np.zeros((6, 6), np.uint32)
    masks = np.zeros((3, (vocabulary + 31) // 32), np.uint32)
    masks[0, 0] = 1 << 31
    masks[1, 1] = 1
    # Deliberately set projection-tail bits; none denote an actual logit.
    masks[:, -1] = 0xFFFFFFF8
    mask_rows = np.asarray([-1, 0, 1, 2, 0, 0], np.int32)
    return dict(logits=logits, draws=draws, masks=masks, mask_rows=mask_rows)


def test_mask_reference_preserves_mixed_rows_and_source_failures():
    graph, _ = definition(6, 67, 3)
    result = ops.evaluate_reference(graph, fixture()).outputs[0]
    assert result.tolist() == [[66, 0], [31, 0], [32, 0], [-1, 1], [-1, 2], [-1, 2]]


@pytest.mark.device
@pytest.mark.parametrize("workspace", [0, 1 << 20])
@pytest.mark.parametrize("vocabulary", [67, 4099])
def test_tilelang_packed_masks_match_reference_with_both_partition_schedules(workspace, vocabulary):
    from engine import DevicePlan

    graph, signature = definition(6, vocabulary, 3)
    values = fixture(vocabulary)
    expected = ops.evaluate_reference(graph, values).outputs[0]
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend="metal", maximum_bytes=8 << 20)
    ) as device:
        program = ops.compile(
            ops.sample_constrained,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="decode", workspace_limit=workspace),
        )
        resources = []
        execution = None
        try:
            for argument in signature.args:
                resources.append(device.upload(argument.spec, values[argument.name].tobytes()))
            execution = program.submit(*resources)
            actual = np.frombuffer(
                device.read(execution.outputs[0], after=execution.completion), np.int32
            ).reshape(6, 2)
            np.testing.assert_array_equal(actual, expected)
        finally:
            if execution is not None:
                for output in execution.outputs:
                    output.close()
            for resource in resources:
                resource.close()
            program.close()


def test_categorical_masking_keeps_request_draws_when_rows_are_regrouped():
    rows, vocabulary = 4, 67
    graph, _ = definition(rows, vocabulary, 1)
    rng = np.random.default_rng(183)
    logits = rng.normal(size=(rows, vocabulary)).astype(np.float32)
    draws = np.asarray([(1, 78, 23, i, 456, 0) for i in range(rows)], np.uint32)
    masks = np.asarray([[0x80000011, 0x01000003, 0x00000004]], np.uint32)
    row_map = np.asarray([0, -1, 0, -1], np.int32)
    values = dict(logits=logits, draws=draws, masks=masks, mask_rows=row_map)
    actual = ops.evaluate_reference(graph, values).outputs[0]
    ordinary_signature = ops.Signature(
        (
            ops.Argument(ops.TensorSpec((rows, vocabulary), ops.DType.F32), "logits"),
            ops.Argument(ops.TensorSpec((rows, 6), ops.DType.U32), "draws"),
        )
    )
    ordinary = ops.trace(ops.sample, ordinary_signature)
    # Independently state the allowed IDs represented by the packed fixture.
    masked = logits.copy()
    prohibited = sorted(set(range(vocabulary)) - {0, 4, 31, 32, 33, 56, 66})
    masked[np.ix_([0, 2], prohibited)] = -np.inf
    expected = ops.evaluate_reference(ordinary, dict(logits=masked, draws=draws)).outputs[0]
    np.testing.assert_array_equal(actual, expected)
    order = np.asarray([3, 0, 2, 1])
    moved = ops.evaluate_reference(
        graph, dict(logits=logits[order], draws=draws[order], masks=masks, mask_rows=row_map[order])
    ).outputs[0]
    np.testing.assert_array_equal(moved, actual[order])
