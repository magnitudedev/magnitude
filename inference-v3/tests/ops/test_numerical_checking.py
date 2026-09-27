"""Compression thresholds do not hide bad producers or bad physical state."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import ops
from ops.kv_codecs import decode_kv_reference, encode_kv_reference
from ops.lab.checking import CheckedExecution, checked_append
from ops.lab.preparation import NumericalMismatch
from ops.lab.records import MeasurementProtocol


@pytest.mark.parametrize("packing", [1, 2])
def test_checked_cache_write_separates_accuracy_from_encoding(packing):
    rep = replace(ops.affine_k8_uniform_v4(32, 32), packing_version=packing)
    state = ops.kv_state_spec(5, 2, ops.DType.F32, rep)
    dense = ops.TensorSpec((2, 2, 32), ops.DType.F32)
    indices = ops.TensorSpec((2,), ops.DType.I32)
    initial = encode_kv_reference(np.zeros(state.shape, np.float32), rep)
    values = np.full(dense.shape, 1.49, np.float32)
    values[..., 0], values[..., 1] = 0, 15  # Exact unit quantization step.
    actual = values.copy()
    actual[..., 2] = 1.51  # Small permitted error crosses a code boundary.
    keys = np.zeros(dense.shape, np.float32)
    destinations = np.array([3, -1], np.int32)
    protocol = MeasurementProtocol(absolute_tolerance=0.03, relative_tolerance=0)
    expected = (keys, values, destinations)
    produced = (keys, actual, destinations)
    contents = checked_append(initial, state, expected, produced, (dense, dense, indices), protocol)
    decoded = decode_kv_reference(contents, state)
    assert decoded[3, 0, 34] == 2
    assert np.all(decoded[[0, 1, 2, 4]] == 0)
    # The numerical check rejects a bad producer even when its storage is valid.
    wrong = actual.copy()
    wrong[..., 3] += 1
    with pytest.raises(NumericalMismatch):
        checked_append(
            initial, state, expected, (keys, wrong, destinations), (dense, dense, indices), protocol
        )
    # Replay comparison must reject one corrupted byte, including untouched rows.
    checked = CheckedExecution((contents,), {}, 1, 1)
    resource = SimpleNamespace(spec=state, content=contents)
    device = SimpleNamespace(read=lambda resource: resource.content)
    checked.compare(device, (resource,), {}, protocol)
    damaged = bytearray(contents)
    damaged[0] ^= 1
    resource.content = bytes(damaged)
    with pytest.raises(NumericalMismatch, match="encoded output"):
        checked.compare(device, (resource,), {}, protocol)
