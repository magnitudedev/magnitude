"""Serialized execution, awakened by control jobs or native completion.

Only the owner reconciles tickets and releases device leases. Completion wakes
have reserved delivery: saturation of the bounded control queue cannot prevent
completion or deadlock shutdown. Idle service has no polling timer.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from typing import Protocol

from magnitude_engine.platform.execution import Ticket


class Driven(Protocol):
    def advance(self) -> Ticket | None: ...
    def failed(self, error: Exception) -> None: ...


class WorkerUnavailable(RuntimeError):
    pass


class _Job[Owner](ABC):
    @abstractmethod
    def run(self, owner: Owner) -> None: ...
    @abstractmethod
    def reject(self, error: Exception) -> None: ...


@dataclass
class _Call[Owner, Result](_Job[Owner]):
    action: Callable[[Owner], Result]
    future: Future[Result]

    def run(self, owner: Owner) -> None:
        if self.future.set_running_or_notify_cancel():
            try:
                self.future.set_result(self.action(owner))
            except Exception as error:
                self.future.set_exception(error)

    def reject(self, error: Exception) -> None:
        if self.future.set_running_or_notify_cancel():
            self.future.set_exception(error)


@dataclass(frozen=True)
class _Completed:
    ticket: Ticket


@dataclass(frozen=True)
class _Stop:
    pass


class Worker[Owner: Driven]:
    def __init__(
        self, factory: Callable[[], AbstractContextManager[Owner]], *, control_capacity: int = 256
    ):
        if control_capacity <= 0:
            raise ValueError("control capacity must be positive")
        self.ready: Future[None] = Future()
        self._queue: SimpleQueue[_Job[Owner] | _Completed | _Stop] = SimpleQueue()
        self._lock = Lock()
        self._queued = 0
        self._capacity = control_capacity
        self._closed = False
        self._failure: Exception | None = None
        self._thread = Thread(target=self._run, args=(factory,), name="magnitude-execution")
        self._thread.start()

    def _unavailable(self) -> WorkerUnavailable:
        error = WorkerUnavailable("execution worker stopped")
        error.__cause__ = self._failure
        return error

    def call[Result](self, action: Callable[[Owner], Result]) -> Future[Result]:
        future: Future[Result] = Future()
        with self._lock:
            if self._closed:
                error = self._unavailable()
            elif self._queued >= self._capacity:
                error = WorkerUnavailable("execution control queue is full")
            else:
                self._queued += 1
                self._queue.put(_Call(action, future))
                return future
        # Future callbacks may call back into the worker. Never run under its lock.
        future.set_exception(error)
        return future

    def _run(self, factory: Callable[[], AbstractContextManager[Owner]]) -> None:
        try:
            with factory() as owner, ThreadPoolExecutor(max_workers=1) as waiter:
                waiting: Ticket | None = None
                self.ready.set_result(None)
                try:
                    while True:
                        event = self._queue.get()
                        if isinstance(event, _Stop):
                            break
                        if isinstance(event, _Completed):
                            if event.ticket is not waiting:
                                raise RuntimeError("completion does not match pending execution")
                            try:
                                event.ticket.wait()
                            except Exception as error:
                                owner.failed(error)
                            waiting = None
                        else:
                            with self._lock:
                                self._queued -= 1
                            event.run(owner)
                        completion = owner.advance()
                        if completion is not None and completion is not waiting:
                            if waiting is not None:
                                raise RuntimeError("execution advanced before pending completion")
                            waiting = completion
                            native = waiter.submit(completion.completion_waiter())

                            def finished(future: Future[None], ticket: Ticket = completion):
                                # Native failures are reconciled on the execution owner.
                                future.exception()
                                self._queue.put(_Completed(ticket))

                            native.add_done_callback(finished)
                except Exception as error:
                    owner.failed(error)
                    raise
        except Exception as error:
            with self._lock:
                self._failure = error
            if not self.ready.done():
                self.ready.set_exception(error)
        finally:
            with self._lock:
                self._closed = True
            while True:
                try:
                    event = self._queue.get_nowait()
                except Empty:
                    break
                if isinstance(event, _Job):
                    event.reject(self._unavailable())

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(_Stop())
        self._thread.join()
