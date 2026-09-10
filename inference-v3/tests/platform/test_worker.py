"""Control saturation, completion ownership and fatal wakeup without polling."""

from concurrent.futures import Future
from contextlib import contextmanager
from threading import Event, Thread, get_ident
from typing import cast

import pytest

from magnitude_engine.platform.execution import Ticket
from magnitude_engine.platform.worker import Worker, WorkerUnavailable


class Completion:
    def __init__(self):
        self.owner = get_ident()
        self.started, self.completed = Event(), Event()
        self.reconciled = False

    def completion_waiter(self):
        assert get_ident() == self.owner

        def wait():
            assert get_ident() != self.owner
            self.started.set()
            assert self.completed.wait(5)

        return wait

    def wait(self):
        assert get_ident() == self.owner
        assert self.completed.is_set()
        self.reconciled = True


class Owner:
    def __init__(self):
        self.ticket: Completion | None = None
        self.error: Exception | None = None
        self.crash = False

    def advance(self):
        if self.crash:
            raise ValueError("deliberate owner failure")
        if self.ticket is not None and not self.ticket.reconciled:
            return cast(Ticket, self.ticket)
        return None

    def failed(self, error):
        self.error = error


def test_control_capacity_cannot_block_completion_or_shutdown():
    closed = Event()

    @contextmanager
    def open_owner():
        owner = Owner()
        try:
            yield owner
        finally:
            closed.set()

    worker = Worker(open_owner, control_capacity=2)
    entered, release = Event(), Event()
    try:
        worker.ready.result(5)

        def submit(owner):
            owner.ticket = Completion()
            return owner.ticket

        ticket = worker.call(submit).result(5)
        assert ticket.started.wait(5)
        # The owner remains responsive while its native wait is blocked.
        assert worker.call(lambda owner: 42).result(5) == 42

        def hold(owner):
            entered.set()
            assert release.wait(5)

        blocking = worker.call(hold)
        assert entered.wait(5)
        cancelled = worker.call(lambda owner: pytest.fail("cancelled call executed"))
        assert cancelled.cancel()
        pending = worker.call(lambda owner: 7)
        with pytest.raises(WorkerUnavailable, match="full"):
            worker.call(lambda owner: 8).result(5)
        closing = Thread(target=worker.close)
        closing.start()
        ticket.completed.set()
        release.set()
        blocking.result(5)
        assert pending.result(5) == 7
        closing.join(5)
        assert not closing.is_alive() and closed.is_set()
        with pytest.raises(WorkerUnavailable):
            worker.call(lambda owner: None).result(5)
    finally:
        release.set()
        if "ticket" in locals():
            ticket.completed.set()
        worker.close()


def test_fatal_owner_failure_rejects_already_queued_work_with_cause():
    entered, release, closed = Event(), Event(), Event()
    owner_record: Future[Owner] = Future()

    @contextmanager
    def open_owner():
        owner = Owner()
        owner_record.set_result(owner)
        try:
            yield owner
        finally:
            closed.set()

    worker = Worker(open_owner)
    try:

        def crash(owner):
            entered.set()
            assert release.wait(5)
            owner.crash = True

        first = worker.call(crash)
        assert entered.wait(5)
        queued = worker.call(lambda owner: pytest.fail("work executed after fatal failure"))
        release.set()
        first.result(5)
        with pytest.raises(WorkerUnavailable) as error:
            queued.result(5)
        assert isinstance(error.value.__cause__, ValueError)
        assert str(owner_record.result().error) == "deliberate owner failure"
        assert closed.wait(5)
    finally:
        release.set()
        worker.close()
