import gc
import weakref

import pytest

from magnitude_engine.platform.execution import (
    CapacityError,
    DeviceContext,
    DType,
    Executable,
    Prepared,
    SubmissionError,
    TensorSpec,
)

SPEC = TensorSpec((16,), DType.F32)


class Buffer:
    def __init__(self, size):
        self.content = bytes(size)
        self.released = False

    @property
    def allocated_bytes(self):
        return len(self.content)

    def close(self):
        assert not self.released
        self.released = True


class Event:
    def __init__(self):
        self.complete = False
        self.fail = False

    def ready(self):
        return self.complete

    def wait(self):
        if self.fail:
            raise RuntimeError("device lost")
        self.complete = True


class Driver:
    def __init__(self):
        self.buffers = []
        self.events = []
        self.fail_drain = False

    def allocate(self, size):
        result = Buffer(size)
        self.buffers.append(result)
        return result

    def upload(self, content):
        result = self.allocate(len(content))
        result.content = content
        return result

    def read(self, buffer, offset, size):
        return buffer.content[offset : offset + size]

    def write(self, buffer, offset, content):
        buffer.content = buffer.content[:offset] + content + buffer.content[offset + len(content) :]

    def record(self, *, timing=False):
        event = Event()
        self.events.append(event)
        return event

    def dispatch(self, commands):
        for command in commands:
            command.launch()

    def drain(self):
        if self.fail_drain:
            raise RuntimeError("device lost during drain")
        for event in self.events:
            event.wait()


class Kernel:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def launch(self, args):
        self.calls += 1
        assert all(not buffer.released for buffer, _, _ in args)
        if self.fail:
            raise RuntimeError("launch failed after work may have been enqueued")

    def bind(self, args):
        return Command(self, args)


class Command:
    def __init__(self, kernel, args):
        self.kernel, self.args = kernel, args

    def launch(self):
        self.kernel.launch(self.args)

    def close(self):
        self.args = ()


def command(context, tensor, *, fail=False):
    return Prepared(context, Executable((SPEC,), Kernel(fail), context.driver), [tensor])


def test_discarded_output_and_ticket_do_not_reclaim_inflight_storage_or_code():
    driver = Driver()
    context = DeviceContext(driver, 64)
    tensor = context.allocate(SPEC)
    prepared = command(context, tensor)
    kernel = weakref.ref(prepared.kernel)
    ticket = context.submit([prepared])
    tensor.close()
    del ticket, prepared
    gc.collect()
    assert context.allocated_bytes == 64
    assert not driver.buffers[0].released
    assert kernel() is not None
    driver.events[0].complete = True
    context.reap()
    assert context.allocated_bytes == 0
    assert driver.buffers[0].released
    assert kernel() is None


def test_views_charge_shared_backing_once_and_close_independently():
    context = DeviceContext(Driver(), 64)
    tensor = context.allocate(SPEC)
    view = tensor.view(TensorSpec((8,), DType.F32), offset=32)
    tensor.close()
    assert context.allocated_bytes == 64
    view.close()
    view.close()
    assert context.allocated_bytes == 0


def test_native_padding_is_charged_and_failed_admission_releases_allocation():
    class PaddedDriver(Driver):
        def allocate(self, size):
            return super().allocate(size + 32)

    driver = PaddedDriver()
    context = DeviceContext(driver, 96)
    tensor = context.allocate(SPEC)
    assert context.allocated_bytes == 96
    with pytest.raises(ValueError, match="exceeds parent"):
        tensor.view(TensorSpec((24,), DType.F32))
    tensor.close()
    assert context.allocated_bytes == 0
    too_small = DeviceContext(driver, 64)
    with pytest.raises(CapacityError) as failure:
        too_small.allocate(SPEC)
    assert failure.value.required == 96
    assert driver.buffers[-1].released
    assert too_small.allocated_bytes == 0


def test_bounded_upload_and_failure_do_not_publish_partial_storage():
    class Source:
        size = 80

        def __init__(self, fail=False):
            self.reads = []
            self.fail = fail

        def read(self, offset, length):
            self.reads.append((offset, length))
            if self.fail and len(self.reads) == 3:
                raise OSError("source failed")
            return bytes(range(offset, offset + length))

    driver = Driver()
    context = DeviceContext(driver, 64)
    source = Source()
    tensor = context.upload_source(SPEC, source, 16, chunk_bytes=17)
    assert source.reads == [(16, 17), (33, 17), (50, 17), (67, 13)]
    assert driver.buffers[-1].content == bytes(range(16, 80))
    tensor.close()
    with pytest.raises(OSError, match="source failed"):
        context.upload_source(SPEC, Source(fail=True), 16, chunk_bytes=17)
    assert driver.buffers[-1].released
    assert context.allocated_bytes == 0


def test_prepare_failure_unwinds_claims_without_launching_or_mutating():
    context = DeviceContext(Driver(), 64)
    first = context.allocate(SPEC)
    second = first.view(SPEC)
    second.close()
    kernel = Kernel()
    with pytest.raises(RuntimeError, match="closed"):
        Prepared(context, Executable((SPEC, SPEC), kernel, context.driver), [first, second])
    first.close()
    assert context.allocated_bytes == 0
    assert kernel.calls == 0


def test_partial_submission_keeps_all_claims_and_cannot_be_retried():
    context = DeviceContext(Driver(), 64)
    tensor = context.allocate(SPEC)
    first, second = command(context, tensor), command(context, tensor, fail=True)
    with pytest.raises(SubmissionError) as failure:
        context.submit([first, second])
    tensor.close()
    assert context.allocated_bytes == 64
    with pytest.raises(RuntimeError, match="unavailable"):
        context.submit([first])
    with pytest.raises(SubmissionError):
        failure.value.ticket.wait()
    assert context.allocated_bytes == 0
    assert first.kernel.calls == second.kernel.calls == 1


def test_failed_completion_quarantines_storage_and_disables_context():
    driver = Driver()
    context = DeviceContext(driver, 64)
    tensor = context.allocate(SPEC)
    ticket = context.submit([command(context, tensor)])
    tensor.close()
    driver.events[0].fail = True
    with pytest.raises(RuntimeError, match="device lost"):
        ticket.wait()
    assert context.allocated_bytes == 64
    assert not driver.buffers[0].released
    with pytest.raises(RuntimeError, match="unavailable"):
        context.allocate(SPEC)


def test_capacity_reclaims_only_after_last_consumer():
    context = DeviceContext(Driver(), 64)
    tensor = context.allocate(SPEC)
    first = context.submit([command(context, tensor)])
    second = context.submit([command(context, tensor)], after=[first])
    tensor.close()
    first.wait()
    with pytest.raises(CapacityError):
        context.allocate(SPEC)
    second.wait()
    replacement = context.allocate(SPEC)
    replacement.close()
    assert context.allocated_bytes == 0


def test_prepared_cancellation_and_duplicate_submission():
    context = DeviceContext(Driver(), 64)
    tensor = context.allocate(SPEC)
    prepared = command(context, tensor)
    with pytest.raises(ValueError, match="twice"):
        context.submit([prepared, prepared])
    tensor.close()
    prepared.close()
    assert context.allocated_bytes == 0
    with pytest.raises(ValueError, match="consumed"):
        context.submit([prepared])


def test_cross_context_operands_are_rejected_before_launch():
    left, right = DeviceContext(Driver(), 64), DeviceContext(Driver(), 64)
    tensor = left.allocate(SPEC)
    with pytest.raises(ValueError, match="another device"):
        command(right, tensor)
    tensor.close()
    assert left.allocated_bytes == right.allocated_bytes == 0
