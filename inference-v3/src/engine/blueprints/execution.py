import ops
from engine.composition import Blueprint, blueprint
from engine.devices import DevicePlan

__all__ = ["DeviceRuntime"]


@blueprint
class DeviceRuntime(Blueprint[ops.DeviceRuntime]):
    plan: DevicePlan

    @staticmethod
    def implementation():
        def realize(plan: DevicePlan) -> ops.DeviceRuntime:
            return ops.DeviceRuntime.open(plan)

        return realize
