from pydantic import Field

import ops
from engine.composition import Blueprint, blueprint
from engine.data import Record
from engine.devices import DevicePlan

__all__ = ["DeviceRuntime", "ScheduleProfile"]


class ScheduleProfile(Record):
    """An explicitly calibrated selection store and its validation-suite identity."""

    directory: str = Field(min_length=1)
    validation: str = Field(min_length=1)


@blueprint
class DeviceRuntime(Blueprint[ops.DeviceRuntime]):
    plan: DevicePlan
    schedules: ScheduleProfile | None = None

    @staticmethod
    def implementation():
        def realize(
            plan: DevicePlan, schedules: ScheduleProfile | None = None
        ) -> ops.DeviceRuntime:
            selected = (
                None
                if schedules is None
                else ops.QualifiedSchedules.for_device(
                    plan, schedules.directory, schedules.validation
                )
            )
            return ops.DeviceRuntime.open(plan, schedules=selected)

        return realize
