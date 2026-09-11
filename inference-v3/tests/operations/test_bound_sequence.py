import os
from contextlib import ExitStack

import numpy as np
import pytest

from magnitude_engine.kernels.pointwise.portable import pointwise
from magnitude_engine.kernels.precision import REFERENCE_F32
from magnitude_engine.kernels.semantics import Pointwise
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.binding import BoundSequence
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.host.machine import open_context


@pytest.mark.device
def test_nested_binding_replays_and_invocation_outlives_every_owner():
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 1024**2, 0)
    spec = TensorSpec((67,), DType.F32)
    values = np.arange(67, dtype=np.float32) - 13
    x = context.upload(spec, values.tobytes())
    middle, output, final = (context.allocate(spec) for _ in range(3))
    add = context.compile(pointwise(
            67, Pointwise.ADD, capability=context.capability, precision=REFERENCE_F32
        ))
    first = BoundSequence(
        context,
        (
            Prepared(context, add, (x, x, middle)),
            Prepared(context, add, (middle, x, output)),
        ),
    )
    outer = BoundSequence(context, (first.prepare(), Prepared(context, add, (output, x, final))))
    first.close()
    for tensor in (x, middle, output):
        tensor.close()
    for _ in range(2):
        invocation = outer.prepare()
        assert invocation.dispatches == 3
        ticket = context.submit((invocation,))
        actual = np.frombuffer(context.read(final, after=ticket), np.float32)
        np.testing.assert_array_equal(actual, values * 4)
    invocation = outer.prepare()
    outer.close()
    final.close()
    retained = context.allocated_bytes
    assert retained > 0
    ticket = context.submit((invocation,))
    assert context.allocated_bytes == retained
    ticket.wait()
    assert context.allocated_bytes == 0
    context.close()


@pytest.mark.device
def test_parameterized_sequence_releases_old_state_and_rebinds_subviews():
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 1024**2, 0)
        cleanup.callback(context.close)
        spec, row = TensorSpec((2, 67), DType.F32), TensorSpec((67,), DType.F32)
        values = np.arange(134, dtype=np.float32).reshape(2, 67)
        source = context.upload(spec, values.tobytes())
        output, scratch = context.allocate(row), context.allocate(row)
        tail = source.view(row, row.nbytes)
        add = context.compile(pointwise(
            67, Pointwise.ADD, capability=context.capability, precision=REFERENCE_F32
        ))
        binding = BoundSequence(
            context,
            (
                Prepared(context, add, (tail, tail, scratch)),
                Prepared(context, add, (scratch, tail, output)),
            ),
            (source, output),
        )
        inner = binding
        binding = BoundSequence(context, (inner.prepare((source, output)),), (source, output))
        inner.close()
        tail.close()
        source.close()
        output.close()
        retained = context.allocated_bytes
        # Changing input/output allocations are absent from the retained plan.
        scratch.close()
        assert context.allocated_bytes == retained
        for scale in (2, 5):
            source = context.upload(spec, (values * scale).tobytes())
            output = context.allocate(row)
            invocation = binding.prepare((source, output))
            source.close()
            ticket = context.submit((invocation,))
            actual = np.frombuffer(context.read(output, after=ticket), np.float32)
            np.testing.assert_array_equal(actual, values[1] * scale * 3)
            output.close()
            assert context.allocated_bytes == retained
        binding.close()
        assert context.allocated_bytes == 0


@pytest.mark.device
def test_independent_region_joins_producer_and_consumer_with_rebinding():
    from magnitude_engine.platform.execution import ExecutionOrder

    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 4 * 1024**2, 0)
        cleanup.callback(context.close)
        count = 65539
        spec = TensorSpec((count,), DType.F32)
        values = (np.arange(count, dtype=np.float32) % 101) - 50
        source = context.upload(spec, values.tobytes())
        x, a, b, output = (context.allocate(spec) for _ in range(4))
        add = context.compile(pointwise(
            count, Pointwise.ADD, capability=context.capability, precision=REFERENCE_F32
        ))
        multiply = context.compile(
            pointwise(
                count, Pointwise.MULTIPLY, capability=context.capability, precision=REFERENCE_F32
            )
        )
        branches = BoundSequence(
            context,
            (Prepared(context, multiply, (x, x, a)), Prepared(context, add, (x, x, b))),
            (x, a, b),
            order=ExecutionOrder.INDEPENDENT,
        )
        binding = BoundSequence(
            context,
            (
                Prepared(context, add, (source, source, x)),
                branches.prepare((x, a, b)),
                Prepared(context, add, (a, b, output)),
            ),
            (source, output),
        )
        branches.close()
        for tensor in (source, output, x, a, b):
            tensor.close()
        for scale in (1, 3):
            source = context.upload(spec, (values * scale).tobytes())
            output = context.allocate(spec)
            invocation = binding.prepare((source, output))
            assert invocation.dispatches == 4
            source.close()
            if scale == 3:
                binding.close()
            ticket = context.submit((invocation,))
            actual = np.frombuffer(context.read(output, after=ticket), np.float32)
            np.testing.assert_array_equal(actual, (values * scale * 2) ** 2 + values * scale * 4)
            output.close()
        assert context.allocated_bytes == 0
