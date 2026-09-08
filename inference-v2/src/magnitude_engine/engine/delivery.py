"""Bounded request output with a terminal lane independent of token capacity."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition


@dataclass(frozen=True)
class Tokens:
    values: tuple[int, ...]


@dataclass(frozen=True)
class PrefillProgress:
    """Completed bulk prompt state, including reused tokens; final anchor is separate."""

    completed_tokens: int
    total_tokens: int
    cached_tokens: int
    elapsed_ns: int


@dataclass(frozen=True)
class Finished:
    reason: str
    prompt_tokens: int
    generated_tokens: int
    cached_tokens: int
    proposed_tokens: int
    accepted_tokens: int
    queued_ns: int
    first_token_ns: int | None
    finished_ns: int
    message: str = ""
    forced_tokens: int = 0
    prefill_ns: int = 0
    decode_ns: int = 0
    first_decode_ns: int = 0
    preparation_ns: int = 0
    media_tokens: int = 0
    cached_input_features: int = 0


class Delivery:
    """One execution-thread producer; consumers return credit when taking tokens.

    Generation is allowed at most the available token credit. Cancellation and
    errors can always publish a terminal record, even with an unresponsive reader.
    """

    def __init__(self, capacity: int, wake: Callable[[], None], *, progress: bool = False):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("delivery capacity must be a positive token count")
        if type(progress) is not bool:
            raise ValueError("progress subscription must be boolean")
        self.capacity = capacity
        self._wake = wake
        self._condition = Condition()
        self._chunks: deque[Tokens] = deque()
        self._buffered = 0
        self._finish: Finished | None = None
        self._terminal_taken = False
        self.progress_enabled = progress
        self._progress: PrefillProgress | None = None

    def report(self, progress: PrefillProgress) -> None:
        """Latest completed state only; progress never consumes output credit."""
        if self.progress_enabled:
            with self._condition:
                if self._finish is None:
                    self._progress = progress
                    self._condition.notify_all()

    @property
    def credit(self) -> int:
        with self._condition:
            return 0 if self._finish is not None else self.capacity - self._buffered

    @property
    def finish(self) -> Finished | None:
        with self._condition:
            return self._finish

    def publish(self, tokens: tuple[int, ...]) -> None:
        with self._condition:
            if self._finish is not None or len(tokens) > self.capacity - self._buffered:
                raise RuntimeError("token publication exceeds live delivery credit")
            if tokens:
                self._chunks.append(Tokens(tokens))
                self._buffered += len(tokens)
                self._condition.notify_all()

    def terminate(self, finish: Finished) -> None:
        with self._condition:
            if self._finish is not None:
                raise RuntimeError("request already has a terminal record")
            self._finish = finish
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> Tokens | Finished | PrefillProgress:
        with self._condition:
            if not self._condition.wait_for(
                lambda: (
                    self._progress is not None or bool(self._chunks) or self._finish is not None
                ),
                timeout,
            ):
                raise TimeoutError("request has no available output")
            if self._progress is not None:
                progress, self._progress = self._progress, None
                return progress
            if self._chunks:
                event = self._chunks.popleft()
                self._buffered -= len(event.values)
            elif not self._terminal_taken:
                self._terminal_taken = True
                assert self._finish is not None
                return self._finish
            else:
                raise StopIteration
        self._wake()
        return event
