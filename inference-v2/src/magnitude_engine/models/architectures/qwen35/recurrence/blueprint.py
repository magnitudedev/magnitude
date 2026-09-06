from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.recurrence.blueprint import Delta
from magnitude_engine.models.recurrence.contracts import DeltaRecurrence

from ..contracts import (
    RecurrentFactory,
)


@component
class Mixer(Blueprint[RecurrentFactory]):
    update: Blueprint[DeltaRecurrence] = field(default_factory=Delta)

    @staticmethod
    def implementation() -> type[RecurrentFactory]:
        from .binding import Mixer

        return Mixer
