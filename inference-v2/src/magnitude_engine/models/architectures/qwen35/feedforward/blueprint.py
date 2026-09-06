from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.experts.blueprint import Resident
from magnitude_engine.models.experts.contracts import ExpertFactory

from ..contracts import (
    FeedForwardFactory,
)


@component
class MoE(Blueprint[FeedForwardFactory]):
    experts: Blueprint[ExpertFactory] = field(default_factory=Resident)

    @staticmethod
    def implementation() -> type[FeedForwardFactory]:
        from .binding import MoE

        return MoE
