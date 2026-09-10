"""Real dependency, ABI binding, subview, and completion checks on a device."""

import os

import numpy as np
import pytest
import tilelang.language as T

from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context


def affine(size: int, cpu: bool):
    @T.prim_func
    def main(Z: T.Tensor((size,), "float32"), A: T.Tensor((size,), "float32")):
        if cpu:
            for i in T.Parallel(size):
                A[i] = Z[i] * 2 + 1
        else:
            with T.Kernel(T.ceildiv(size, 64), threads=64) as block:
                lane = T.get_thread_binding(0)
                i = block * 64 + lane
                if i < size:
                    A[i] = Z[i] * 2 + 1

    return main


@pytest.mark.device
def test_batch_dependencies_reordered_arguments_and_owned_subviews():
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    size = 129
    spec = TensorSpec((size,), DType.F32)
    backing = TensorSpec((size * 2,), DType.F32)
    context = open_context(backend, 4 * spec.nbytes, 0)
    values = np.arange(size, dtype=np.float32)
    source = context.upload(
        backing, np.concatenate((np.full(size, -999), values)).astype(np.float32).tobytes()
    )
    selected = source.view(spec, spec.nbytes)
    middle, output = context.allocate(spec), context.allocate(spec)
    kernel = context.compile(affine(size, backend == Backend.LLVM))
    assert context.compile(affine(size, backend == Backend.LLVM)) is kernel
    first = Prepared(context, kernel, [selected, middle])
    second = Prepared(context, kernel, [middle, output])
    source.close()
    selected.close()
    middle.close()
    ticket = context.submit([first, second], timing=backend != Backend.LLVM)
    del kernel, first, second
    actual = np.frombuffer(context.read(output, after=ticket), np.float32)
    np.testing.assert_array_equal(actual, values * 4 + 3)
    if backend != Backend.LLVM:
        assert ticket.device_seconds is not None and ticket.device_seconds > 0
    assert context.allocated_bytes == spec.nbytes
    output.close()
    assert context.allocated_bytes == 0
    context.close()
