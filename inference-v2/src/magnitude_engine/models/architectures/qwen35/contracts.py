"""Qwen block binding contracts; generic model composition has no family dependency."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from magnitude_engine.models.experts.contracts import ExpertFactory

if TYPE_CHECKING:
    from magnitude_engine.models.experts.contracts import ExpertOperator
    from magnitude_engine.models.projections import ParallelProjections

    from .attention.operation import GatedAttention
    from .feedforward.operation import (
        DenseFeedForward,
        RoutedFeedForward,
    )
    from .recurrence.operation import RecurrentMixer


class AttentionFactory(ABC):
    @abstractmethod
    def bind(self, layer: Any, slot: int, inputs: ParallelProjections) -> GatedAttention: ...


class RecurrentFactory(ABC):
    @abstractmethod
    def bind(self, layer: Any, slot: int, inputs: ParallelProjections) -> RecurrentMixer: ...


class FeedForwardFactory(ABC):
    experts: ExpertFactory

    @abstractmethod
    def bind(
        self, layer: Any, expert: ExpertOperator | None, routing: ParallelProjections | None
    ) -> DenseFeedForward | RoutedFeedForward: ...
