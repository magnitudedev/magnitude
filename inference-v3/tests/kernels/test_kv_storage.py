import os

import numpy as np
import pytest

from magnitude_engine.kernels.kv.append import append_kv
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.state.kv import KVLayout, KVPool


@pytest.mark.device
def test_real_slab_offsets_and_consumer_lifetimes():
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 1024**2, 0)
    pool = KVPool(context, KVLayout(3, 2, 32), page_tokens=4, slab_pages=8)
    (peer,) = pool.reserve(8)
    (run,) = pool.reserve(12, after=peer)
    assert peer.adjacent(run)
    rng = np.random.default_rng(462)
    arrays = [rng.normal(size=(7, 2, 32)).astype(np.float32) for _ in range(2)]
    inputs = [context.upload(TensorSpec(a.shape, DType.F32), a.tobytes()) for a in arrays]
    offset = context.upload(TensorSpec((1,), DType.I32), np.array([3], np.int32).tobytes())
    keys, values = run.views(2)
    program = append_kv(7, 2, 32, run.capacity, capability=context.capability)
    prepared = Prepared(context, context.compile(program), [*inputs, offset, keys, values])
    read_keys = keys.view(TensorSpec((7, 2, 32), DType.F32), 3 * 2 * 32 * 4)
    read_values = values.view(TensorSpec((7, 2, 32), DType.F32), 3 * 2 * 32 * 4)
    peer.close()
    run.close()
    keys.close()
    values.close()
    ticket = context.submit([prepared])
    for tensor in [*inputs, offset]:
        tensor.close()
    for view, expected in zip((read_keys, read_values), arrays, strict=True):
        actual = np.frombuffer(context.read(view, after=ticket), np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(actual, expected)
    assert pool.slab_count == 1
    read_keys.close()
    read_values.close()
    assert pool.slab_count == 0
    assert context.allocated_bytes == 0
    pool.close()
    context.close()
