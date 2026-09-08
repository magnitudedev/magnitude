"""Request admission data and thread-safe delivery/cancellation handles."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING

from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.prompt import Prompt

if TYPE_CHECKING:
    from magnitude_engine.models.context import InputSource

from .delivery import Delivery


@dataclass(frozen=True, init=False)
class GenerationRequest:
    prompt: Prompt
    sampling: SamplingPolicy
    max_tokens: int
    stop_tokens: tuple[int, ...] = ()
    constraint: ConstraintSpec | None = None
    inputs: "InputSource | None" = None

    def __init__(
        self,
        prompt: Prompt | tuple[int, ...],
        sampling: SamplingPolicy,
        max_tokens: int,
        stop_tokens: tuple[int, ...] = (),
        constraint: ConstraintSpec | None = None,
        *,
        inputs: "InputSource | None" = None,
    ) -> None:
        prompt = Prompt(prompt) if isinstance(prompt, tuple) else prompt
        if (
            not prompt.tokens
            or any(type(t) is not int or not 0 <= t < 2**31 for t in stop_tokens)
            or type(max_tokens) is not int
            or max_tokens < 0
        ):
            raise ValueError("request has invalid tokens or generation allowance")
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "stop_tokens", stop_tokens)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "inputs", inputs)


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
