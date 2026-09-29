"""Prepared production contexts for the shared API and interactive client."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..compiler.compilation import CompileOptions
from ..runtime.resources import DeviceRuntime
from .evidence import ExecutionContext
from .fixtures import Fixture
from .records import MeasurementProtocol
from .worker import Lab


@dataclass(frozen=True, slots=True)
class Configuration:
    label: str
    fixture: Fixture
    device: Callable[[], DeviceRuntime]
    options: CompileOptions
    store: Path
    protocol: MeasurementProtocol = MeasurementProtocol()
    prepared_limit: int = 8
    reference_bytes: int = 256 << 20
    context: ExecutionContext | None = None

    def __post_init__(self):
        if not self.label:
            raise ValueError("a prepared configuration needs a display label")
        if any(type(value) is not int or value < 1 for value in (self.prepared_limit, self.reference_bytes)):
            raise ValueError("preparation count and reference byte limits must be positive integers")

    def open(self) -> Lab:
        return Lab(fixture=self.fixture, device=self.device, options=self.options,
                   store=self.store, protocol=self.protocol, label=self.label,
                   prepared_limit=self.prepared_limit, reference_bytes=self.reference_bytes,
                   context=self.context)


def show(configurations: Sequence[Configuration]) -> None:
    """Choose typed prepared contexts without constructing another benchmark tree.

    Only the selected context opens a device. Changing configuration drains and
    closes its owner before another starts. History remains in the shared store.
    """
    from .archive import StoredConfiguration
    from .store import ObservationStore
    from .tui import ConfigurationApp, PerformanceApp, RecordedApp

    configurations = tuple(configurations)
    if not configurations or any(not isinstance(item, Configuration) for item in configurations):
        raise ValueError("show requires prepared Lab configurations")
    while True:
        recorded = []
        for path in dict.fromkeys(configuration.store.resolve() for configuration in configurations):
            if path.is_file():
                with ObservationStore(path) as store:
                    recorded.extend(StoredConfiguration(path, item) for item in store.configurations())
        selected = ConfigurationApp((*configurations, *recorded)).run()
        if selected is None:
            return
        if isinstance(selected, StoredConfiguration):
            with ObservationStore(selected.store) as store:
                result = RecordedApp(store, selected.record).run()
            if result != "choose":
                return
            continue
        lab = selected.open()
        try:
            result = PerformanceApp(lab, configuration_label=selected.label, can_choose=True).run()
        finally:
            lab.close()
        if result != "choose":
            return
