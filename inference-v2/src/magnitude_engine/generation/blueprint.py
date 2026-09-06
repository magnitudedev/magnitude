from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.executor.contracts import ExecutorFactory

from .contracts import GenerationFactory, MethodFactory
from .methods.plain.blueprint import Plain


@component
class Generation(Blueprint[GenerationFactory]):
    target: Blueprint[ExecutorFactory]
    method: Blueprint[MethodFactory] = field(default_factory=Plain)

    @staticmethod
    def implementation() -> type[GenerationFactory]:
        from .binding import Generation

        return Generation
