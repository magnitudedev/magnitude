"""The model contract consumed by generation and service policy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import magnitensor as mt
from magnitude_engine.data import TokenId
from magnitude_engine.inputs.layout import InputLayout


class LogitsSelection(StrEnum):
    NONE = "none"
    LAST = "last"
    ALL = "all"


class ModelAdvance(Protocol):
    @property
    def logits(self) -> mt.Resource | None: ...
    def commit(self) -> None: ...
    def read_sample(self) -> tuple[int, int] | None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class ModelRequest:
    sequence: ModelSequence
    tokens: tuple[TokenId, ...]
    selection: LogitsSelection = LogitsSelection.LAST
    draw_words: tuple[int, int, int, int, int, int] | None = None


class ModelBatch(Protocol):
    @property
    def logits(self) -> mt.Resource | None:
        """Requested logit rows packed in request order."""
        ...

    @property
    def completion(self) -> mt.Completion: ...
    @property
    def advances(self) -> tuple[ModelAdvance, ...]: ...
    def close(self) -> None: ...


class ModelExecutor(ABC):
    context: mt.Device

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
    def context(self) -> mt.Device: ...
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


class ModelInput(Protocol):
    """Owned original conditioning capable of constructing fresh numerical state."""

    @property
    def model(self) -> ModelExecutor: ...
    @property
    def prompt(self) -> tuple[TokenId, ...]: ...
    def open(self) -> ModelSequence: ...
    def close(self) -> None: ...
