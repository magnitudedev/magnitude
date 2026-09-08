"""Model-owned input interpretation and complete continuation checkpoints.

Input state owns architecture semantics and temporary preparation resources.
Storage owns decoder history. A model checkpoint retains both, at one legal
boundary; neither generation nor the allocator interprets the input state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from magnitude_engine.resources.retention import RetainedStorage

if TYPE_CHECKING:
    from magnitude_engine.composition import Blueprint

    from .execution import ResourceLease
    from .inputs import ModelInputs
    from .operations import Task
    from .preparation import ImagePreparation, PreparedMedia
    from .prompt import Prompt


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
