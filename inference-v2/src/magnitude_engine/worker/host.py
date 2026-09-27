"""Host-owned subprocess and pull-based request delivery; no MLX imports."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
from collections import deque
from dataclasses import asdict
from threading import Condition, Event, Lock, Thread
from time import monotonic
from typing import BinaryIO, cast
from uuid import uuid4

from magnitude_engine.composition import Blueprint, digest, dumps
from magnitude_engine.engine.contracts import EngineInstance
from magnitude_engine.engine.delivery import Finished, PrefillProgress, Tokens
from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.preparation import PreparedMedia

from .framing import Frame, read_frame, write_frame


class WorkerUnavailable(RuntimeError):
    pass


class RemoteRequest:
    def __init__(self, host: Worker, identity: str):
        self.host, self.identity = host, identity
        self._condition = Condition()
        self._consumer = Lock()
        self._event: Tokens | Finished | PrefillProgress | None = None
        self._failure: BaseException | None = None
        self._pending = False
        self._accepted = False
        self._closed = False
        self._terminal_error = False
        self._cancelling = False
        self._cancelled = Event()
        self._cancel_ack = False
        self._terminal: Finished | None = None

    def next(self, timeout: float | None = None) -> Tokens | Finished | PrefillProgress:
        deadline = None if timeout is None else monotonic() + timeout

        def remaining() -> float | None:
            return None if deadline is None else max(0, deadline - monotonic())

        if not self._consumer.acquire(blocking=False):
            raise RuntimeError("request already has an output consumer")
        try:
            with self._condition:
                if self._closed:
                    raise StopIteration
                if self._failure is not None:
                    raise self._failure
                if not self._condition.wait_for(
                    lambda: self._accepted or self._failure is not None, remaining()
                ):
                    raise TimeoutError("worker request acceptance is still pending")
                if self._failure is not None:
                    raise self._failure
                if not self._pending:
                    self.host._send({"type": "read", "request_id": self.identity})
                    self._pending = True
                if not self._condition.wait_for(
                    lambda: self._event is not None or self._failure is not None, remaining()
                ):
                    # Keep the outstanding read. A later next() waits for this
                    # same response; an observation timeout never duplicates work.
                    raise TimeoutError("worker output is still pending")
                if self._failure is not None:
                    raise self._failure
                event, self._event = self._event, None
                self._pending = False
                assert event is not None
                if isinstance(event, Finished):
                    self._terminal = event
                    self._closed = True
                    self.host._forget(self.identity)
                return event
        finally:
            self._consumer.release()

    def cancel(self, timeout: float = 5) -> Finished | None:
        with self._condition:
            if self._closed or self._terminal_error:
                return self._terminal
            if not self._cancelling:
                self.host._send({"type": "cancel", "request_id": self.identity})
                self._cancelling = True
                self._failure = WorkerUnavailable("request cancelled")
                self._condition.notify_all()
        if not self._cancelled.wait(timeout):
            raise TimeoutError("worker cancellation has not completed")
        if not self._cancel_ack:
            raise WorkerUnavailable("worker failed before cancellation was acknowledged")
        with self._condition:
            self._closed = True
            self._event = None
        self.host._forget(self.identity)
        return self._terminal

    def _receive(self, event: Tokens | Finished | PrefillProgress) -> None:
        with self._condition:
            if self._cancelling:
                return
            if self._event is not None or not self._pending:
                raise ValueError("worker sent output without an outstanding read")
            self._event = event
            self._condition.notify_all()

    def _fail(self, error: BaseException, *, terminal: bool = False) -> None:
        with self._condition:
            self._failure = error
            self._terminal_error |= terminal
            self._event = None
            self._cancelled.set()
            self._condition.notify_all()


class Worker:
    """One residency generation, one retained Popen handle, bounded command writes.

    This is the explicit development launcher. Installed ICN launch authority is
    a separate integration boundary, not inferred from a checkout or environment.
    """

    def __init__(self, engine: Blueprint[EngineInstance], *, startup_timeout: float = 120):
        payload = dumps(engine)
        composition_digest = digest(engine)
        self.generation = uuid4().hex
        self.blueprint = engine
        self.composition_digest = composition_digest
        self._lock = Lock()
        self._close_lock = Lock()
        self._requests: dict[str, RemoteRequest] = {}
        self._commands: queue.Queue[Frame | None] = queue.Queue(1)
        self._ready = Event()
        self._started = Event()
        self._closing = False
        self._disposed = False
        self._failure: BaseException | None = None
        self.properties: dict = {}
        self._stderr: deque[bytes] = deque(maxlen=64)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "magnitude_engine.worker",
                "--development-runtime",
                "--parent-pid",
                str(os.getpid()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            start_new_session=True,
        )
        self._threads = [
            Thread(target=target, daemon=True)
            for target in (self._write, self._read, self._read_stderr)
        ]
        for thread in self._threads:
            thread.start()
        self._started.set()
        try:
            self._send({"type": "load", "blueprint": payload, "digest": composition_digest})
            if not self._ready.wait(startup_timeout):
                raise TimeoutError("model worker has not completed startup")
            self._check()
        except BaseException:
            self.close(grace=0)
            raise

    @classmethod
    def start(cls, *, engine: Blueprint[EngineInstance], startup_timeout: float = 120) -> Worker:
        return cls(engine, startup_timeout=startup_timeout)

    @property
    def stderr(self) -> str:
        return b"".join(tuple(self._stderr)).decode(errors="replace")

    def _check(self) -> None:
        if self._failure is not None:
            raise WorkerUnavailable(str(self._failure)) from self._failure
        if self._closing:
            raise WorkerUnavailable("model worker is closing")

    def _send(self, message: dict, buffers: tuple[bytes, ...] = ()) -> None:
        self._check()
        try:
            self._commands.put(Frame(self.generation, message, buffers), timeout=1)
        except queue.Full as error:
            raise WorkerUnavailable("worker command writer is backpressured") from error

    def _forget(self, identity: str) -> None:
        with self._lock:
            self._requests.pop(identity, None)

    @property
    def available(self) -> bool:
        with self._lock:
            return not self._closing and self._failure is None and self.process.poll() is None

    def submit(
        self,
        prompt: tuple[int, ...],
        sampling: SamplingPolicy,
        max_tokens: int,
        stop_tokens: tuple[int, ...] = (),
        *,
        constraint: ConstraintSpec | None = None,
        progress: bool = False,
        media: PreparedMedia | None = None,
    ) -> RemoteRequest:
        with self._lock:
            self._check()
            if len(self._requests) >= 1024:
                raise OverflowError("host request delivery capacity is full")
            request = RemoteRequest(self, uuid4().hex)
            self._requests[request.identity] = request
        try:
            self._send(
                {
                    "type": "infer",
                    "request_id": request.identity,
                    "prompt": prompt,
                    "sampling": asdict(sampling),
                    "max_tokens": max_tokens,
                    "stop_tokens": stop_tokens,
                    "constraint": None if constraint is None else asdict(constraint),
                    "progress": progress,
                    "media": None if media is None else media.encode(),
                },
                () if media is None else media.buffers,
            )
        except BaseException:
            self._forget(request.identity)
            raise
        return request

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            if self._failure is not None:
                return
            self._failure = error
            requests = tuple(self._requests.values())
        # Do not hold the registry lock while taking request conditions.
        for request in requests:
            request._fail(error)
        self._ready.set()
        if not self._closing:

            def retire() -> None:
                self._started.wait()
                try:
                    self.close(grace=0)
                except BaseException as disposal:
                    self._failure = WorkerUnavailable(
                        f"{error}; worker disposal failed: {disposal}"
                    )

            Thread(target=retire, daemon=True).start()

    def _write(self) -> None:
        assert self.process.stdin is not None
        try:
            while True:
                message = self._commands.get()
                if message is None:
                    return
                try:
                    write_frame(cast(BinaryIO, self.process.stdin), message)
                finally:
                    message.close()
        except BaseException as error:
            self._fail(WorkerUnavailable(f"worker command transport failed: {error}"))

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            while True:
                message = read_frame(cast(BinaryIO, self.process.stdout), self.generation).message
                kind = message.get("type")
                if kind == "ready":
                    if self._ready.is_set() or message.get("pid") != self.process.pid:
                        raise ValueError("worker readiness does not match the launched process")
                    if message.get("composition_digest") != self.composition_digest:
                        raise ValueError("worker built a different composition")
                    self.properties = message
                    self._ready.set()
                    continue
                if kind == "fatal":
                    raise WorkerUnavailable(message.get("message", "worker failed"))
                identity = message.get("request_id")
                if not isinstance(identity, str):
                    raise ValueError("worker response has no request identity")
                with self._lock:
                    request = self._requests.get(identity)
                if request is None:
                    if kind == "cancelled":
                        continue
                    raise ValueError("worker response names an unknown request")
                if kind == "accepted":
                    with request._condition:
                        request._accepted = True
                        request._condition.notify_all()
                elif kind == "tokens":
                    request._receive(Tokens(tuple(message["event"]["values"])))
                elif kind == "progress":
                    request._receive(PrefillProgress(**message["event"]))
                elif kind == "finished":
                    request._receive(Finished(**message["event"]))
                elif kind == "cancelled":
                    with request._condition:
                        request._terminal = (
                            None if message["event"] is None else Finished(**message["event"])
                        )
                        request._cancel_ack = True
                        request._cancelled.set()
                elif kind == "error":
                    request._fail(WorkerUnavailable(message["message"]), terminal=True)
                    self._forget(identity)
                else:
                    raise ValueError("unknown worker response")
        except BaseException as error:
            self._fail(WorkerUnavailable(f"worker response transport ended: {error}"))

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while data := self.process.stderr.read(1024):
            self._stderr.append(data)

    def close(self, *, grace: float = 2, force: float = 2) -> None:
        with self._close_lock:
            if self._disposed:
                return
            self._closing = True
            try:
                self._commands.put_nowait(Frame(self.generation, {"type": "shutdown"}))
            except queue.Full:
                pass
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                for terminate in (self.process.terminate, self.process.kill):
                    try:
                        terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        self.process.wait(timeout=force)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                else:
                    raise WorkerUnavailable("worker exit was not observed after forced disposal")
            self._fail(WorkerUnavailable("model worker exited"))
            try:
                self._commands.put_nowait(None)
            except queue.Full:
                pass
            for thread in self._threads:
                thread.join(timeout=1)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()
            self._disposed = True

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
