import magnitensor as mt
from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.platform.backend import Backend

__all__ = ["Device"]


@blueprint
class Device(Blueprint[mt.Device]):
    backend: Backend
    budget_bytes: int
    ordinal: int = 0

    @staticmethod
    def implementation():
        def build(backend: Backend, budget_bytes: int, ordinal: int) -> mt.Device:
            return mt.device(backend.value, budget_bytes=budget_bytes, ordinal=ordinal)

        return build
