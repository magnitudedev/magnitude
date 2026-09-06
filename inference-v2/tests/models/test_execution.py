from concurrent.futures import ThreadPoolExecutor

import pytest

from magnitude_engine.models.execution import ExecutionOwner


class Backend:
    def __init__(self, events, *, fail_complete=False, fail_drain=False):
        self.events = events
        self.fail_complete, self.fail_drain = fail_complete, fail_drain

    def submit(self, arrays):
        self.events.append("submit")

    def complete(self, arrays):
        self.events.append("complete")
        if self.fail_complete:
            raise RuntimeError("GPU execution failed")

    def drain(self):
        self.events.append("drain")
        if self.fail_drain:
            raise RuntimeError("device could not drain")


class Lease:
    def __init__(self, events, name, *, fail=False):
        self.events, self.name, self.fail = events, name, fail

    def close(self):
        self.events.append(self.name)
        if self.fail:
            raise RuntimeError("IO failed while draining")


def test_submission_keeps_leases_until_completion_and_retires_in_reverse_order():
    events = []
    owner = ExecutionOwner(Backend(events))
    with owner.scope() as scope:
        scope.acquire(lambda: Lease(events, "expert"))
        scope.acquire(lambda: Lease(events, "embedding"))
        pending = scope.seal()
    pending.submit()
    assert events == ["submit"]
    pending.complete()
    pending.complete()
    owner.close()
    assert events == ["submit", "complete", "embedding", "expert"]


def test_model_build_failure_drains_before_releasing_operation_resources():
    events = []
    owner = ExecutionOwner(Backend(events))
    with pytest.raises(ValueError, match="model failure"):
        with owner.scope() as scope:
            scope.acquire(lambda: Lease(events, "release"))
            raise ValueError("model failure")
    assert events == ["drain", "release"]
    owner.close()


def test_failed_completion_drains_and_failed_lease_does_not_leak_other_leases():
    events = []
    owner = ExecutionOwner(Backend(events, fail_complete=True))
    with owner.scope() as scope:
        scope.acquire(lambda: Lease(events, "first"))
        scope.acquire(lambda: Lease(events, "second", fail=True))
        pending = scope.seal()
    with pytest.raises(BaseExceptionGroup, match="resource retirement"):
        pending.complete()
    assert events == ["complete", "drain", "second", "first"]
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.scope()


def test_failed_device_drain_retains_storage_and_poisoned_owner_rejects_more_work():
    events = []
    owner = ExecutionOwner(Backend(events, fail_complete=True, fail_drain=True))
    with owner.scope() as scope:
        scope.acquire(lambda: Lease(events, "must stay held"))
        pending = scope.seal()
    with pytest.raises(BaseExceptionGroup, match="worker disposal"):
        pending.complete()
    assert events == ["complete", "drain"]
    assert not pending.done
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.scope()


def test_execution_owner_rejects_gpu_work_on_an_io_thread():
    owner = ExecutionOwner(Backend([]))
    with owner.scope():
        pass
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(owner.scope).result()
    owner.close()


def test_scratch_retirement_uses_lease_identity_not_value_equality():
    class EqualLease(Lease):
        def __eq__(self, other):
            raise AssertionError("resource ownership cannot invoke value equality")

    events = []
    owner = ExecutionOwner(Backend(events))
    with owner.scope() as scope:
        first = scope.acquire(lambda: EqualLease(events, "first"))
        second = scope.acquire(lambda: EqualLease(events, "second"))
        scope.retire(second)
        with pytest.raises(ValueError, match="belong"):
            scope.retire(EqualLease(events, "foreign"))
        assert first in (first,)
    assert events == ["complete", "second", "complete", "first"]


def test_span_discards_submitted_graphs_but_holds_all_leases_until_its_fence():
    events = []
    owner = ExecutionOwner(Backend(events))
    with owner.span() as span:
        with owner.scope() as scope:
            scope.acquire(lambda: Lease(events, "first"))
            first = scope.seal()
        span.submit(first)
        first.retain(Lease(events, "transaction"))
        with owner.scope() as scope:
            scope.acquire(lambda: Lease(events, "second"))
            second = scope.seal()
        span.submit(second)
        assert not first.roots and not second.roots
        assert not first.done and not second.done
        assert events == ["submit", "submit"]
        first.complete()  # One member cannot retire ahead of the span's fence.
        assert second.done
        assert events == ["submit", "submit", "drain", "transaction", "first", "second"]
    owner.close()
    assert not owner._pending


def test_failed_span_fence_keeps_all_leases_for_worker_disposal():
    events = []
    owner = ExecutionOwner(Backend(events, fail_drain=True))
    with pytest.raises(RuntimeError, match="could not drain"):
        with owner.span() as span:
            with owner.scope() as scope:
                scope.acquire(lambda: Lease(events, "must stay held"))
                pending = scope.seal()
            span.submit(pending)
    assert events == ["submit", "drain"]
    assert not pending.done
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.scope()


def test_span_exception_drains_and_members_cannot_cross_owners_or_spans():
    events = []
    owner = ExecutionOwner(Backend(events))
    other = ExecutionOwner(Backend([]))
    with pytest.raises(ValueError, match="injected"):
        with owner.span() as span:
            with owner.scope() as scope:
                scope.acquire(lambda: Lease(events, "release"))
                pending = scope.seal()
            with other.span() as foreign:
                with pytest.raises(ValueError, match="owner"):
                    foreign.submit(pending)
            span.submit(pending)
            with pytest.raises(ValueError, match="span"):
                span.submit(pending)
            raise ValueError("injected")
    assert events == ["submit", "drain", "release"]
    assert not owner._pending
    owner.close()
    other.close()



def test_span_submits_downstream_consumers_together_with_model_roots():
    roots = []

    class Recording(Backend):
        def submit(self, arrays):
            roots.append(arrays)
            super().submit(arrays)

    events = []
    owner = ExecutionOwner(Recording(events))
    # Opaque roots suffice: the backend contract, not a particular neural kernel,
    # defines which submitted consumers keep the external lease alive.
    model_root, sampled_token = object(), object()
    with owner.span() as span:
        with owner.scope() as scope:
            scope.acquire(lambda: Lease(events, "release"))
            pending = scope.seal(model_root)
        span.submit(pending, sampled_token)
        assert roots == [(model_root, sampled_token)]
        assert events == ["submit"]
    assert events == ["submit", "drain", "release"]
    owner.close()
