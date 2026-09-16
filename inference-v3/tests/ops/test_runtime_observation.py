"""Runtime boundary accounting; these checks need no native compiler or GPU."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from ops.binding import MemorySource, SourceSpan
from ops.runtime.memory import Limit, ReservationLedger
from ops.runtime.observation import Activity, KernelActivity, ObservationStatus
from ops.runtime.resources import Completion, DeviceRuntime, NativeSubmissionError, NativeUpload
from ops.tensor.types import DType, TensorSpec


class Allocation:
    def __init__(self, content):
        self.content = content
        self.allocated_bytes = len(content)
        self.closed = False

    def view(self, spec, offset=0):
        assert not self.closed
        return self.content[offset:offset + spec.storage_nbytes]

    def close(self):
        self.closed = True


class NativeCompletion:
    def __init__(self):
        self.done = False

    def ready(self):
        return self.done

    def wait(self):
        self.done = True


class Entrypoint:
    def submit(self, arguments):
        return NativeCompletion()


class Runtime:
    def capture_kernels(self, limit):
        return None

    def allocate(self, size, alignment):
        return Allocation(bytes((size + alignment - 1) // alignment * alignment))

    def upload(self, spec, content):
        return Allocation(content)

    def download(self, value):
        return value

    def join(self, completions):
        # Simulate an adapter that proves completion with one aggregate event,
        # without calling the individual completion wrappers.
        return NativeCompletion()

    def close(self):
        pass


class UploadRuntime(Runtime):
    def __init__(self, *, extra_bytes=0):
        self.pending = NativeCompletion()
        self.staging = Allocation(b"staging")
        self.extra_bytes = extra_bytes
        self.uploads = 0

    def upload_async(self, spec, content):
        self.uploads += 1
        self.destination = Allocation(content + bytes(self.extra_bytes))
        return NativeUpload(self.destination, self.pending, self.staging)


def test_async_upload_pins_source_and_destination_after_output_is_abandoned():
    runtime = UploadRuntime()
    with DeviceRuntime(runtime, budget_bytes=32) as device:
        execution = device.upload_async(TensorSpec((8,), DType.U8), b"abcdefgh")
        execution.outputs[0].close()
        assert device.allocated_bytes == 16
        assert not execution.completion.done
        assert not runtime.destination.closed and not runtime.staging.closed
        device.drain()
        assert runtime.destination.closed and runtime.staging.closed
        assert device.allocated_bytes == 0
        assert not device._submissions and not device._completions


def test_async_upload_observation_requires_transfer_completion():
    runtime = UploadRuntime()
    with DeviceRuntime(runtime, budget_bytes=32) as device:
        with pytest.raises(RuntimeError, match="before execution completed"):
            with device.observe() as capture:
                execution = device.upload_async(TensorSpec((8,), DType.U8), b"abcdefgh")
        assert capture.result.status == ObservationStatus.INCOMPLETE
        assert capture.result.completed_bytes(Activity.UPLOAD) == 0
        assert not runtime.staging.closed
        execution.completion.wait()
        assert device.read(execution.outputs[0]) == b"abcdefgh"
        assert device.allocated_bytes == 8
        execution.outputs[0].close()


def test_async_upload_charges_host_capacity_before_native_submission():
    from ops.runtime.memory import CapacityError

    runtime = UploadRuntime()
    with DeviceRuntime(runtime, budget_bytes=12) as device:
        with pytest.raises(CapacityError):
            device.upload_async(TensorSpec((8,), DType.U8), b"abcdefgh")
        assert runtime.uploads == 0
        assert device.allocated_bytes == 0


def test_joined_transfer_and_consumer_retire_every_completion_lease():
    runtime = UploadRuntime()
    with DeviceRuntime(runtime, budget_bytes=32) as device:
        with device.observe() as capture:
            transfer = device.upload_async(TensorSpec((8,), DType.U8), b"abcdefgh")
            source = transfer.outputs[0]
            submitted = device.submit_native(Entrypoint(), (source.native,))
            consumer = Completion(device, submitted, (source.fork(),))
            joined = Completion.join((transfer.completion, consumer))
            source.close()
            assert device.allocated_bytes == 16
            assert not runtime.staging.closed and not runtime.destination.closed
            joined.wait()
            assert runtime.staging.closed and runtime.destination.closed
            assert not device._submissions and not device._completions
        assert capture.result.status == ObservationStatus.COMPLETE
        assert capture.result.completed_bytes(Activity.UPLOAD) == 8
        assert capture.result.memory.end_bytes == 0


def test_invalid_async_upload_retains_ownership_when_drain_fails():
    class FailedWait(NativeCompletion):
        fail = True

        def wait(self):
            if self.fail:
                raise RuntimeError("transfer drain failed")
            super().wait()

    runtime = UploadRuntime(extra_bytes=1)
    runtime.pending = FailedWait()
    with DeviceRuntime(runtime, budget_bytes=32) as device:
        with pytest.raises(RuntimeError, match="transfer drain failed"):
            device.upload_async(TensorSpec((8,), DType.U8), b"abcdefgh")
        assert not runtime.staging.closed and not runtime.destination.closed
        assert device.allocated_bytes == 16
        runtime.pending.fail = False
        device.drain()
        assert runtime.staging.closed and runtime.destination.closed
        assert device.allocated_bytes == 0


class NativeCapture:
    clock = "test-native-clock"

    def __init__(self, *, start_error=None, finish_error=None, close_error=None):
        self.events = []
        self.start_error, self.finish_error, self.close_error = start_error, finish_error, close_error

    def start(self):
        self.events.append("start")
        if self.start_error:
            raise self.start_error

    def finish(self):
        self.events.append("finish")
        if self.finish_error:
            raise self.finish_error
        return (KernelActivity("first", 100), KernelActivity("second", 200))

    def close(self):
        self.events.append("close")
        if self.close_error:
            raise self.close_error


class TimedRuntime(Runtime):
    def __init__(self, capture):
        self.capture = capture

    def capture_kernels(self, limit):
        assert limit == 2
        return self.capture


def test_native_timing_is_part_of_the_same_observation():
    native = NativeCapture()
    with DeviceRuntime(TimedRuntime(native), budget_bytes=4096) as device:
        with device.observe(kernel_limit=2) as capture:
            value = device.allocate(TensorSpec((8,), DType.U8))
            value.close()
        assert capture.result.status == ObservationStatus.COMPLETE
        assert capture.result.kernels.clock == native.clock
        assert capture.result.kernels.elapsed_ns == 300
        assert capture.result.memory.reserved_bytes == 8
        assert native.events == ["start", "finish", "close"]
        with device.observe() as host_only:
            pass
        assert host_only.result.kernels is None
        assert native.events == ["start", "finish", "close"]


def test_unsupported_native_timing_remains_explicitly_absent():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with device.observe(kernel_limit=2) as capture:
            pass
        assert capture.result.kernels is None
        assert capture.result.status == ObservationStatus.COMPLETE


@pytest.mark.parametrize("phase", ("start", "finish", "close"))
def test_native_capture_failure_closes_and_does_not_poison_next_observation(phase):
    native = NativeCapture(**{f"{phase}_error": RuntimeError(f"failed {phase}")})
    with DeviceRuntime(TimedRuntime(native), budget_bytes=4096) as device:
        capture = device.observe(kernel_limit=2)
        with pytest.raises(RuntimeError, match=f"failed {phase}"):
            with capture:
                pass
        assert native.events[-1] == "close"
        if phase != "start":
            assert capture.result.status == ObservationStatus.FAILED
        with device.observe() as following:
            pass
        assert following.result.status == ObservationStatus.COMPLETE


def test_native_cleanup_preserves_original_execution_error():
    native = NativeCapture(close_error=RuntimeError("cleanup"))
    with DeviceRuntime(TimedRuntime(native), budget_bytes=4096) as device:
        with pytest.raises(ValueError, match="execution") as raised:
            with device.observe(kernel_limit=2) as capture:
                raise ValueError("execution")
        assert native.events == ["start", "close"]
        assert capture.result.status == ObservationStatus.FAILED
        assert capture.result.error == "ValueError"
        assert "cleanup" in raised.value.__notes__[0]


def test_incomplete_observation_aborts_native_capture_without_resolving_or_waiting():
    native = NativeCapture()
    with DeviceRuntime(TimedRuntime(native), budget_bytes=4096) as device:
        with pytest.raises(RuntimeError, match="before execution completed"):
            with device.observe(kernel_limit=2) as capture:
                pending = device.submit_native(Entrypoint(), ())
        assert not pending.ready()
        assert capture.result.kernels is None
        assert capture.result.status == ObservationStatus.INCOMPLETE
        assert native.events == ["start", "close"]
        pending.wait()


def test_partial_submission_remains_tracked_and_pins_streamed_tiles_until_drain():
    from ops.runtime.streaming import _submit_stage

    class FailedWait(NativeCompletion):
        fail = True

        def wait(self):
            if self.fail:
                raise RuntimeError("drain failed")
            super().wait()

    pending = FailedWait()

    class PartialEntrypoint:
        def submit(self, arguments):
            raise NativeSubmissionError(ValueError("second dispatch failed"), pending)

    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        resource = device.allocate(TensorSpec((8,), DType.U8))
        with pytest.raises(RuntimeError, match="drain failed"):
            _submit_stage(device, PartialEntrypoint(), (resource,))
        resource.close()
        assert device.allocated_bytes == 8
        assert device._submissions and device._completions
        pending.fail = False
        device.drain()
        assert device.allocated_bytes == 0
        assert not device._submissions and not device._completions


def test_native_bound_failure_returns_completion_ownership_without_implicit_wait():
    from ops.runtime.tilelang import _BoundEntrypoint

    pending = NativeCompletion()

    def launch(*arguments):
        raise ValueError("second kernel failed")

    bound = _BoundEntrypoint(launch, lambda: pending)
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with pytest.raises(NativeSubmissionError, match="second kernel failed") as raised:
            device.submit_native(bound, ())
        assert not pending.done
        assert raised.value.__cause__.args == ("second kernel failed",)
        assert device._submissions
        device.drain()
        assert pending.done
        assert not device._submissions


def test_interval_peaks_ignore_old_high_water_and_count_shared_backing_once():
    ledger = ReservationLedger((
        Limit("physical", frozenset({"host", "gpu"}), 1024),
        Limit("execution", frozenset({"gpu"}), 512),
    ))
    old = ledger.reserve(800, frozenset({"host"}))
    old.close()
    baseline = ledger.reserve(64, frozenset({"host", "gpu"}))
    window = ledger.observe()
    first = ledger.reserve(128, frozenset({"gpu"}))
    first.close()
    second = ledger.reserve(32, frozenset({"host"}))
    measured = window.close()
    assert measured.baseline_bytes == 64
    assert measured.peak_bytes == 192
    assert measured.end_bytes == 96
    assert measured.reserved_bytes == 160
    assert measured.released_bytes == 128
    assert measured.constraints[0].peak_bytes == 192
    assert measured.constraints[1].peak_bytes == 192
    assert window.close() is measured
    second.close()
    baseline.close()


def test_complete_boundary_records_io_transfers_aliases_and_completion():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        spec = TensorSpec((8,), DType.U8)
        with device.observe() as capture:
            content = device.read_source(SourceSpan(MemorySource(b"abcdefgh"), 0, 8), value_identity="weights")
            value = device.upload(spec, content)
            alias = value.view(spec)
            native = device.submit_native(Entrypoint(), (alias.native,))
            completion = Completion(device, native, (alias,))
            assert device.read(value, after=completion) == content
            value.close()
        result = capture.result
        assert result.status == ObservationStatus.COMPLETE
        assert result.completed_bytes(Activity.SOURCE_READ) == 8
        assert result.completed_bytes(Activity.UPLOAD) == 8
        assert result.completed_bytes(Activity.DOWNLOAD) == 8
        assert result.completed_bytes(Activity.RELEASE) == 8
        assert result.memory.baseline_bytes == result.memory.end_bytes == 0
        # Destination plus API staging; creating the view reserves nothing.
        assert result.memory.peak_bytes == 16
        assert sum(item.kind == Activity.SUBMIT for item in result.activities) == 1
        assert sum(item.kind == Activity.WAIT for item in result.activities) == 1
        assert all(item.elapsed_ns >= 0 for item in result.activities)
        assert all(item.started_ns + item.elapsed_ns <= result.elapsed_ns for item in result.activities)


def test_unfinished_submit_is_not_a_successful_latency_and_is_not_implicitly_waited():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with pytest.raises(RuntimeError, match="before execution completed"):
            with device.observe() as capture:
                pending = device.submit_native(Entrypoint(), ())
        assert capture.result.status == ObservationStatus.INCOMPLETE
        assert not pending.ready()
        with pytest.raises(RuntimeError, match="earlier invocations"):
            with device.observe():
                pass
        pending.wait()
        with device.observe() as next_capture:
            pass
        assert next_capture.result.status == ObservationStatus.COMPLETE


def test_short_source_read_preserves_observed_bytes_and_failure():
    class ShortSource(MemorySource):
        def read_into(self, offset, destination):
            return super().read_into(offset, destination[:2])

    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with pytest.raises(IOError, match="short bounded read"):
            with device.observe() as capture:
                device.read_source(SourceSpan(ShortSource(b"abcdefgh"), 0, 8), value_identity="weights")
        assert capture.result.status == ObservationStatus.FAILED
        read, = capture.result.activities
        assert read.bytes_requested == 8
        assert read.bytes_completed == 2
        assert read.error == "OSError"


def test_external_wait_does_not_mutate_owner_observation_or_release_resources():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with device.observe() as capture:
            value = device.allocate(TensorSpec((8,), DType.U8))
            native = device.submit_native(Entrypoint(), (value.native,))
            completion = Completion(device, native, (value,))
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(completion.completion_waiter()).result()
            assert device.allocated_bytes == 8
            completion.wait()
            assert device.allocated_bytes == 0
        assert capture.result.status == ObservationStatus.COMPLETE


def test_failed_wait_keeps_resources_pinned_until_successful_drain():
    class FailingCompletion(NativeCompletion):
        fail = True

        def wait(self):
            if self.fail:
                raise RuntimeError("temporary wait failure")
            super().wait()

    class FailingEntrypoint:
        def submit(self, arguments):
            return native

    native = FailingCompletion()
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with pytest.raises(RuntimeError, match="temporary wait failure"):
            with device.observe() as capture:
                value = device.allocate(TensorSpec((8,), DType.U8))
                tracked = device.submit_native(FailingEntrypoint(), (value.native,))
                Completion(device, tracked, (value,)).wait()
        assert capture.result.status == ObservationStatus.FAILED
        assert device.allocated_bytes == 8
        native.fail = False
        device.drain()
        assert device.allocated_bytes == 0


def test_nested_capture_rejection_does_not_break_enclosing_capture():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with device.observe() as capture:
            with pytest.raises(RuntimeError, match="one physical boundary"):
                with device.observe():
                    pass
        assert capture.result.status == ObservationStatus.COMPLETE


def test_aggregate_completion_retires_children_without_individual_wait_callbacks():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        with device.observe() as capture:
            first = device.submit_native(Entrypoint(), ())
            second = device.submit_native(Entrypoint(), ())
            Completion(device, device.join((first, second)), ()).wait()
        assert capture.result.status == ObservationStatus.COMPLETE
        assert not device._submissions


def test_rejected_subview_does_not_leave_an_allocation_claim():
    with DeviceRuntime(Runtime(), budget_bytes=4096) as device:
        allocation = device.allocate(TensorSpec((8,), DType.U8))
        with pytest.raises(ValueError, match="misaligns"):
            allocation.view(TensorSpec((1,), DType.I32), offset=1)
        allocation.close()
        assert device.allocated_bytes == 0
