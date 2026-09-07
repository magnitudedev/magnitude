"""Explicit ownership of lazy work and the resources consumed by that work."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from threading import get_ident
from typing import Protocol

import mlx.core as mx

from magnitude_engine.components import component


class ResourceLease(Protocol):
    """close must drain resource-owned IO before returning its storage to a pool."""

    def close(self) -> None: ...


class CompletionBackend(Protocol):
    """Submission owns device buffers until execution finishes, independently of
    Python graph handles. drain completes all submitted work on this owner.
    """

    def submit(self, arrays: tuple[mx.array, ...]) -> tuple[mx.array, ...]: ...
    def complete(self, arrays: tuple[mx.array, ...]) -> None: ...
    def drain(self) -> None: ...


class MLXCompletion:
    def submit(self, arrays: tuple[mx.array, ...]) -> tuple[mx.array, ...]:
        if not arrays:
            return ()
        # Depends carries completion of every state write without retaining large
        # Python output handles or requiring a fence on unrelated later work.
        ticket = mx.depends(mx.array(0, mx.int32), arrays)
        mx.async_eval(ticket)
        return (ticket,)

    def complete(self, arrays: tuple[mx.array, ...]) -> None:
        if arrays:
            mx.eval(*arrays)

    def drain(self) -> None:
        mx.synchronize()


@component("EXECUTION:DEVICE:MAG:ASYNC")
class ExecutionOwner:
    """Target and drafter share one host execution owner, not one thread per model.

    The first execution binds the owner to its worker thread. Other threads may
    prepare requests or perform bounded IO, but cannot build or retire model work.
    A failed drain poisons the worker: leases remain held until process disposal.
    """

    def __init__(self, backend: CompletionBackend | None = None):
        self.backend = backend or MLXCompletion()
        self._thread: int | None = None
        self._pending: set[PendingExecution] = set()
        self._failed = False
        self._closed = False

    def check(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("execution owner is unavailable")
        current = get_ident()
        if self._thread is None:
            self._thread = current
        elif self._thread != current:
            raise RuntimeError("model work must run on the execution owner thread")

    def scope(self) -> ExecutionScope:
        self.check()
        return ExecutionScope(self)

    def complete(self) -> bool:
        """Complete owned work before a shared layout change; report whether any existed."""
        self.check()
        pending = tuple(self._pending)
        for execution in pending:
            execution.complete()
        return bool(pending)

    @property
    def requires_disposal(self) -> bool:
        """The owner has failed and must be disposed instead of reused."""
        return self._failed

    def drain_after_failure(self, error: BaseException) -> None:
        """Prove cleanup safety, or preserve ownership until worker disposal.

        State reconciliation may evaluate device work after the original forward
        completes. It has the same failure boundary as program execution.
        """
        if self._failed:
            raise error
        try:
            self.backend.drain()
        except BaseException as drain_error:
            self._failed = True
            raise BaseExceptionGroup(
                "execution could not drain; worker disposal required", [error, drain_error]
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        self.check()
        failures = []
        for execution in tuple(self._pending):
            try:
                execution.complete()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("execution shutdown failed", failures)
        self._closed = True


class PendingExecution:
    """A completion obligation independent of the model result's logical contents."""

    def __init__(
        self, owner: ExecutionOwner, roots: tuple[mx.array, ...], leases: tuple[ResourceLease, ...]
    ):
        self.owner = owner
        self.roots = roots
        self._leases = leases
        self.done = False
        self._submitted = False
        owner._pending.add(self)

    def retain(self, lease: ResourceLease) -> None:
        """Keep committed-but-in-flight allocation obligations until completion."""
        self.owner.check()
        if self.done:
            raise RuntimeError("cannot retain resources after execution completes")
        self._leases = (*self._leases, lease)

    def submit(self, *consumers: mx.array) -> None:
        self.owner.check()
        if self.done:
            self.owner.backend.submit(tuple(consumers))
            return
        if self._submitted and not consumers:
            return
        try:
            self.roots = self.owner.backend.submit((*self.roots, *consumers))
            self._submitted = True
        except BaseException as error:
            self._fail(error)

    def complete(self) -> None:
        self.owner.check()
        if self.done:
            return
        try:
            self.owner.backend.complete(self.roots)
        except BaseException as error:
            self._fail(error)
        self._release()

    def _fail(self, error: BaseException) -> None:
        self.owner.drain_after_failure(error)
        try:
            self._release()
        except BaseException as release_error:
            raise BaseExceptionGroup(
                "execution and resource retirement failed", [error, release_error]
            ) from error
        raise error

    def _release(self) -> None:
        failures = []
        for lease in reversed(self._leases):
            try:
                lease.close()
            except BaseException as error:
                failures.append(error)
        self.done = True
        self.roots = ()
        self._leases = ()
        self.owner._pending.remove(self)
        if failures:
            self.owner._failed = True
            raise BaseExceptionGroup("resource retirement failed", failures)


class ExecutionScope:
    """The program explicitly reports outputs, state writes and operation consumers.

    Operators acquire leases through this scope before building consumers. State
    submission at a producing layer releases its activation predecessors early;
    submission alone never retires a lease or permits a bank to be overwritten.
    """

    def __init__(self, owner: ExecutionOwner):
        self.owner = owner
        self._roots: list[mx.array] = []
        self._leases: list[ResourceLease] = []
        self._sealed = False
        self._pending: PendingExecution | None = None

    def _check(self) -> None:
        self.owner.check()
        if self._sealed:
            raise RuntimeError("execution scope is sealed")

    def acquire[L: ResourceLease](self, factory: Callable[[], L]) -> L:
        self._check()
        lease = factory()
        self._leases.append(lease)
        return lease

    def enter[T](self, context: AbstractContextManager[T]) -> T:
        stack = self.acquire(ExitStack)
        return stack.enter_context(context)

    def depend(self, *arrays: mx.array) -> None:
        self._check()
        self._roots.extend(arrays)

    def submit_state(self, *arrays: mx.array) -> None:
        self.depend(*arrays)
        self.owner.backend.submit(tuple(arrays))

    def retire(self, lease: ResourceLease, *consumers: mx.array) -> None:
        """Retire shared scratch at its own consumer boundary, without evaluating logits."""
        self._check()
        index = next((i for i, owned in enumerate(self._leases) if owned is lease), None)
        if index is None:
            raise ValueError("resource lease does not belong to this execution")
        self.owner.backend.complete(tuple(consumers))
        lease.close()
        del self._leases[index]

    def seal(self, *outputs: mx.array) -> PendingExecution:
        self.depend(*outputs)
        self._sealed = True
        pending = PendingExecution(self.owner, tuple(self._roots), tuple(self._leases))
        self._pending = pending
        self._roots.clear()
        self._leases.clear()
        return pending

    def __enter__(self) -> ExecutionScope:
        self._check()
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _traceback: object) -> None:
        if self._sealed:
            if error is not None and self._pending is not None and not self._pending.done:
                self._pending._fail(error)
            return
        pending = self.seal()
        if error is not None:
            pending._fail(error)
        pending.complete()
