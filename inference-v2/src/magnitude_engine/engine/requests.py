"""Request admission data and thread-safe delivery/cancellation handles."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event

from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.generation.sampling_policy import SamplingPolicy

from .delivery import Delivery


@dataclass(frozen=True)
class GenerationRequest:
    prompt: tuple[int, ...]
    sampling: SamplingPolicy
    max_tokens: int
    stop_tokens: tuple[int, ...] = ()
    constraint: ConstraintSpec | None = None

    def __post_init__(self) -> None:
        if (
            not self.prompt
            or any(
                type(t) is not int or not 0 <= t < 2**31 for t in (*self.prompt, *self.stop_tokens)
            )
            or type(self.max_tokens) is not int
            or self.max_tokens < 0
        ):
            raise ValueError("request has invalid tokens or generation allowance")


class RequestHandle:
    def __init__(
        self,
        identity: str,
        request: GenerationRequest,
        delivery: Delivery,
        created_ns: int,
        wake: Callable[[], None],
    ):
        self.identity, self.request, self.delivery = identity, request, delivery
        self.created_ns = created_ns
        self.cancelled = Event()
        self._wake = wake

    def cancel(self) -> None:
        self.cancelled.set()
        self._wake()
