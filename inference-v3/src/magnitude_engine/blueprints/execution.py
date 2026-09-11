from magnitude_engine.composition import Blueprint, blueprint
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DeviceContext

__all__ = ["Device"]


@blueprint
class Device(Blueprint[DeviceContext]):
    backend: Backend
    budget_bytes: int
    ordinal: int = 0

    @staticmethod
    def implementation():
        from magnitude_engine.platform.host.machine import open_context

        return open_context
