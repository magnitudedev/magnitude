"""The model contract consumed by generation and service policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import ops
from engine.data import TokenId
from engine.inputs.layout import InputLayout


class LogitsSelection(StrEnum):
    NONE = "none"
    LAST = "last"
    ALL = "all"


class ModelAdvance(Protocol):
    @property
    def logits(self) -> ops.Resource | None: ...
    def commit(self) -> None: ...
    def read_sample(self) -> tuple[int, int] | None: ...
    def read_logits(self) -> tuple[tuple[float, ...], ...]: ...
    def close(self) -> None: ...


class TokenMask(Protocol):
    """Owner-thread symbolic work independent of the numerical forward."""

    def mask(self) -> bytes: ...


@dataclass(frozen=True)
class ModelRequest:
    """Advance tokens and optionally read raw logits.

    Host rows follow the selected input positions; columns follow ``vocabulary``
    order, or token-ID order when omitted. A packed batch shares one vocabulary.
    Selected vocabulary readout is unsampled and does not normalize logits.
    """
    sequence: ModelSequence
    tokens: tuple[TokenId, ...]
    selection: LogitsSelection = LogitsSelection.LAST
    draw_words: tuple[int, int, int, int, int, int] | None = None
    allowed_tokens: bytes | TokenMask | None = None
    vocabulary: tuple[TokenId, ...] | None = None


class ModelBatch(Protocol):
    @property
    def logits(self) -> ops.Resource | None:
        """Requested logit rows packed in request order."""
        ...

    @property
    def completion(self) -> ops.Completion: ...
    @property
    def advances(self) -> tuple[ModelAdvance, ...]: ...
    def close(self) -> None: ...


class ModelExecutor(ABC):
    context: ops.DeviceRuntime

    def text_input(self, tokens: tuple[TokenId, ...]) -> ModelInput:
        """Construct text conditioning when supported by this architecture."""
        raise NotImplementedError("this model does not support plain-text input")

    @abstractmethod
    def prime(self, rows: int, horizon: int) -> None:
        """Construct canonical execution forms for a prefill quantum and horizon."""
        ...

    @abstractmethod
    def reclaim(self) -> int:
        """Retire unborrowed scratch/cache backing; preserve all live work."""
        ...

    @abstractmethod
    def reclaimable(self, sequences: tuple[ModelSequence, ...]) -> int:
        """Exclusive numerical backing freed by closing this set then reclaiming."""
        ...

    @abstractmethod
    def prepare(self, requests: tuple[ModelRequest, ...]) -> ModelBatch: ...


class ModelCheckpoint(Protocol):
    @property
    def position(self) -> int: ...
    def fork(self) -> ModelSequence: ...
    def close(self) -> None: ...


class ModelSequence(Protocol):
    @property
    def context(self) -> ops.DeviceRuntime: ...
    @property
    def position(self) -> int: ...
    @property
    def layout(self) -> InputLayout: ...
    @property
    def context_limit(self) -> int: ...
    @property
    def model(self) -> ModelExecutor: ...
    def checkpoint(self) -> ModelCheckpoint: ...
    def close(self) -> None: ...


class InputPreparation(Protocol):
    """One asynchronous, indivisible piece of original conditioning work."""

    @property
    def completion(self) -> ops.Completion: ...
    def finish(self) -> None: ...
    def close(self) -> None: ...


class ModelInput(Protocol):
    """Owned original conditioning capable of constructing fresh numerical state."""

    @property
    def model(self) -> ModelExecutor: ...
    @property
    def prompt(self) -> tuple[TokenId, ...]: ...
    def prepare(self) -> InputPreparation | None:
        """Submit the next conditioning operation, or report ready for open."""
        ...

    def open(self) -> ModelSequence: ...
    def close(self) -> None: ...
