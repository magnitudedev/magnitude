"""Semantic representation/state contracts independent of GPU schedules."""
from dataclasses import replace

import numpy as np
import pytest

import ops
from ops.kv_codecs import rotate_reference, encode_kv_reference, decode_kv_reference
from ops.lab.fixtures import Fixture, tensor_identity
from ops.tensor.ops import _kv_copy_reference


@pytest.mark.parametrize('width', [32, 128, 256, 512])
def test_signed_rotation_preserves_dot_products_and_inverse(width):
    rng = np.random.default_rng(903)
    x, q = rng.normal(size=(2, 3, width)).astype(np.float32)
    rx, rq = rotate_reference(x, 42), rotate_reference(q, 42)
    np.testing.assert_allclose(np.sum(rx * rq, axis=-1), np.sum(x * q, axis=-1), rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(rotate_reference(rx, 42, inverse=True), x, atol=1e-6)


@pytest.mark.parametrize('factory', [ops.affine_k8_uniform_v4, ops.rotated_k4_uniform_v4])
def test_packing_conventions_preserve_reconstruction_and_change_identity(factory):
    rng = np.random.default_rng(11)
    logical = rng.normal(size=(11, 3, 512)).astype(np.float32)
    rep = factory(256, 256)
    first, second = replace(rep, packing_version=1), replace(rep, packing_version=2)
    a, b = encode_kv_reference(logical, first), encode_kv_reference(logical, second)
    assert a != b and first.digest != second.digest
    np.testing.assert_array_equal(decode_kv_reference(a, ops.kv_state_spec(11, 3, ops.DType.BF16, first)),
                                  decode_kv_reference(b, ops.kv_state_spec(11, 3, ops.DType.BF16, second)))
    planes = rep.planes(33)
    assert all(plane.offset % 16 == 0 for plane in planes)
    assert all(left.offset + left.nbytes <= right.offset for left, right in zip(planes, planes[1:]))


def test_copy_reference_preserves_reconstructed_precision_and_original_sources():
    rep = ops.affine_k8_uniform_v4(256, 256)
    spec = ops.kv_state_spec(8, 2, ops.DType.BF16, rep)
    ranges = ops.TensorSpec((1, 3), ops.DType.I32)
    graph = ops.trace(lambda state, spans: ops.kv_copy(state, spans, max_count=2),
                      ops.Signature((ops.Argument(spec, 'state', ops.ValueKind.RESOURCE),
                                     ops.Argument(ranges, 'spans'))))
    state = np.full(spec.shape, np.float32(1.001), np.float32)
    state[4:6] = 0
    actual, = ops.evaluate_reference(graph, {'state': state, 'spans': np.array([[0, 4, 2]], np.int32)}).outputs
    np.testing.assert_array_equal(actual[4:6], state[:2])
    assert np.all(state[4:6] == 0)


@pytest.mark.parametrize('ranges', [[[0, 1, 2]], [[0, 4, 2], [2, 5, 2]], [[7, 0, 2]]])
def test_copy_reference_rejects_overlapping_or_out_of_range_transitions(ranges):
    rep = ops.affine_k8_uniform_v4(256, 256)
    with pytest.raises(ValueError):
        _kv_copy_reference((np.zeros((8, 2, 512), np.float32), np.asarray(ranges, np.int32)),
                           {'representation': rep, 'max_count': 2})


def test_fixture_retains_original_packed_bytes_without_requantization():
    rep = ops.rotated_k4_uniform_v4(256, 256)
    spec = ops.kv_state_spec(3, 2, ops.DType.BF16, rep)
    rng = np.random.default_rng(82)
    physical = encode_kv_reference(rng.normal(size=spec.shape).astype(np.float32), rep)
    logical = decode_kv_reference(physical, spec)
    graph = ops.trace(lambda state: state, ops.Signature((ops.Argument(spec, 'state', ops.ValueKind.RESOURCE),)))
    identity = graph.resources[0]
    fixture = Fixture(graph, {identity: logical}, bindings={identity: physical})
    tensor = fixture._tensor(identity)
    assert tensor.physical is physical
    np.testing.assert_array_equal(tensor.reference, logical)
    changed = bytearray(physical)
    changed[-1] ^= 1
    assert tensor_identity(spec, logical, physical) != tensor_identity(spec, logical, bytes(changed))
