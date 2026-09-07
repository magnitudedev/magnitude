"""One model allocation scope, injected by engine construction and shared by drafting."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, TypeVar, cast

import mlx.core as mx

from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.lifetime import Closable

from .definition import ModelDefinition
from .execution import ExecutionOwner
from .ownership import OwnedProgram, VocabularyLoan
from .runtime import ModelRuntime
from .state.arena import LayerGeometry
from .state.recurrent import RecurrentLayout

T = TypeVar("T")


@dataclass(frozen=True)
class ModelDescriptor:
    path: str
    context_tokens: int
    vocab_size: int
    tokenizer_identity: str
    implementation: str
    definition: ModelDefinition


@dataclass(frozen=True)
class PagedRequirements:
    attention: tuple[LayerGeometry, ...]
    dtype: mx.Dtype


@dataclass(frozen=True)
class HybridRequirements(PagedRequirements):
    recurrence: tuple[RecurrentLayout, ...]


@dataclass(frozen=True)
class NativeRequirements:
    make_cache: Callable[[], list]
    # (end position, query tokens): zero query reserves retained history; a
    # positive query also covers the cache extension needed by that forward.
    capacity: Callable[[int, int], int]
    source: object


@dataclass(frozen=True)
class DraftRequirements:
    target: OwnedProgram
    target_feature: str
    capacity: int
    vocabulary: VocabularyLoan


@dataclass(frozen=True)
class BoundProgram:
    program: OwnedProgram
    descriptor: ModelDescriptor
    state: PagedRequirements | NativeRequirements
    drafting: DraftRequirements | None = None


@dataclass(frozen=True)
class BoundExecutor:
    model: ModelRuntime
    program: BoundProgram
    selection: str = "candidate"


class ModelResources:
    def __init__(self, *, budget: MemoryBudget, context_tokens: int, max_active: int):
        self.budget = budget
        self.context_tokens, self.max_active = context_tokens, max_active
        self.lifetime = ExitStack()
        self.owner = ExecutionOwner()
        self._instances: dict[int, tuple[object, Any]] = {}

    def once(self, source: object, construct: Callable[[], T]) -> T:
        """Sharing is by declaration instance, within this residency only."""
        key = id(source)
        if key not in self._instances:
            self._instances[key] = (source, construct())
        return cast(T, self._instances[key][1])

    def own[R: Closable](self, value: R) -> R:
        self.lifetime.callback(value.close)
        return value

    def close(self) -> None:
        # Complete device consumers before releasing any resource they may borrow.
        self.owner.close()
        self._instances.clear()
        self.lifetime.close()
