from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.generation.contracts import GenerationFactory

from .contracts import EngineInstance
from .memory.blueprint import Budgeted
from .memory.contracts import MemoryPolicy as Budget
from .prefixes.blueprint import Radix
from .prefixes.contracts import PrefixIndex as Prefixes
from .scheduler.blueprint import TimeShared
from .scheduler.contracts import Scheduler


@component
class Engine(Blueprint[EngineInstance]):
    generation: Blueprint[GenerationFactory]
    scheduler: Blueprint[Scheduler] = field(default_factory=TimeShared)
    memory: Blueprint[Budget] = field(default_factory=Budgeted)
    prefixes: Blueprint[Prefixes] = field(default_factory=Radix)
    context_tokens: int = 32768
    output_capacity: int = 64

    def __post_init__(self) -> None:
        if self.context_tokens < 1 or not 1 <= self.output_capacity <= 4096:
            raise ValueError("invalid engine context or output capacity")

    @staticmethod
    def implementation() -> type[EngineInstance]:
        from .binding import EngineResidency

        return EngineResidency
