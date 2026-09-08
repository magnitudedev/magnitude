"""Model-owned input interpretation and complete continuation checkpoints.

Input state owns architecture semantics and temporary preparation resources.
Storage owns decoder history. A model checkpoint retains both, at one legal
boundary; neither generation nor the allocator interprets the input state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from typing import TYPE_CHECKING, Protocol

from magnitude_engine.resources.retention import RetainedStorage

if TYPE_CHECKING:
    from magnitude_engine.composition import Blueprint

    from .computation import Computation
    from .execution import ResourceLease
    from .features import FeatureLease, FeatureSet
    from .inputs import ModelInputs
    from .operations import Task
    from .preparation import ImagePreparation, PreparedMedia
    from .prompt import InputSpan, Prompt


class StateCheckpoint(Protocol):
    length: int
    closed: bool

    @property
    def reclaimable(self) -> bool: ...
    def retained_storage(self) -> tuple[RetainedStorage, ...]: ...
    def close(self) -> None: ...


class InputCheckpoint(Protocol):
    """Only input semantics required after the committed boundary are retained."""

    @property
    def reclaimable(self) -> bool: ...
    def retained_storage(self) -> tuple[RetainedStorage, ...]: ...
    def restore(self) -> InputState: ...
    def close(self) -> None: ...


class InputState(Protocol):
    @property
    def cache_hits(self) -> int: ...

    def acquire(self, position: int, count: int) -> ResourceLease:
        """Pin prepared allocations until their numerical consumer completes."""
        ...

    def boundary(self, position: int) -> bool: ...
    def prepare(self, position: int, count: int) -> Task[None]:
        """Expose bounded prerequisite work through the ordinary execution owner."""
        ...

    def assemble(self, inputs: ModelInputs, position: int) -> ModelInputs:
        """Describe numerical operands; computation and resource leases stay in the program."""
        ...

    def batch_key(self, position: int, count: int) -> object:
        """Equal keys permit one program invocation, with row-local semantics."""
        ...

    def checkpoint(self, position: int) -> InputCheckpoint: ...
    def close(self) -> None: ...


class InputContinuation(ABC):
    """Input semantics after preparation resources have been released."""

    @property
    def cache_hits(self) -> int:
        return 0

    def boundary(self, position: int) -> bool:
        return type(position) is int and position >= 0

    def acquire(self, position: int, count: int) -> ResourceLease:
        return ExitStack()

    def prepare(self, position: int, count: int) -> Task[None]:
        yield from ()

    @abstractmethod
    def assemble(self, inputs: ModelInputs, position: int) -> ModelInputs: ...

    def batch_key(self, position: int, count: int) -> object:
        return None

    @abstractmethod
    def checkpoint(self, position: int) -> InputCheckpoint: ...

    def close(self) -> None:
        return None


class SpannedInput(Protocol):
    @property
    def span(self) -> InputSpan: ...


class SpanContext[I: SpannedInput](InputContinuation):
    """Prepare and pin only the features intersecting a decoder span.

    Families supply the computation and numerical assembly. Span legality and
    feature lifetime are shared regardless of the input's architecture.
    """

    def __init__(
        self,
        prompt: Prompt,
        items: tuple[I, ...],
        features: FeatureSet,
        bytes_per_token: int,
        encode: Callable[[I, FeatureLease], Computation],
    ):
        self.prompt, self.items, self.features = prompt, items, features
        self.bytes_per_token, self.encode = bytes_per_token, encode

    def boundary(self, position: int) -> bool:
        return (
            type(position) is int and position >= len(self.prompt.tokens)
        ) or self.prompt.boundary(position)

    @property
    def cache_hits(self) -> int:
        return self.features.cache_hits

    def prepare(self, position: int, count: int) -> Task[None]:
        self.features.keep({item.span.identity for item in self.items if item.span.end > position})
        for item in self.items:
            if item.span.start < position + count and item.span.end > position:
                yield from self.features.prepare(
                    item.span.identity,
                    (item.span.end - item.span.start) * self.bytes_per_token,
                    partial(self.encode, item),
                )

    def acquire(self, position: int, count: int) -> ResourceLease:
        return self.features.pin(
            item.span.identity
            for item in self.items
            if item.span.start < position + count and item.span.end > position
        )

    def close(self) -> None:
        self.features.close()


class InputSource(Protocol):
    def bind(self, checkpoint: InputCheckpoint | None) -> InputState:
        """Bind prepared input to a new or restored semantic continuation."""
        ...


class InputFactory(Protocol):
    @property
    def processor(self) -> Blueprint[ImagePreparation]: ...

    def prepare(self, tokens: tuple[int, ...], media: PreparedMedia) -> tuple[Prompt, InputSource]:
        """Validate prepared host tensors and bind their model semantics before admission."""
        ...


class ModelCheckpoint[C: StateCheckpoint]:
    def __init__(self, storage: C, inputs: InputCheckpoint | None, domain: object):
        self.storage = storage
        self.inputs = inputs
        self.domain = domain
        self.length = storage.length
        self.closed = False

    @property
    def reclaimable(self) -> bool:
        return self.storage.reclaimable and (self.inputs is None or self.inputs.reclaimable)

    def retained_storage(self) -> tuple[RetainedStorage, ...]:
        return self.storage.retained_storage() + (
            () if self.inputs is None else self.inputs.retained_storage()
        )

    def close(self) -> None:
        if self.closed:
            return
        errors = []
        for resource in (self.inputs, self.storage):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    errors.append(error)
        self.closed = True
        if errors:
            raise BaseExceptionGroup("model checkpoint release failed", errors)
