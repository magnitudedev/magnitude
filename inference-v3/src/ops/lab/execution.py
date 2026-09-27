"""Headless use of the same production configurations and retained formula fixtures."""

from __future__ import annotations

import importlib
import json
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

from pydantic import Field, JsonValue

from .configuration import Configuration
from .evidence import ExecutionContext
from .records import MeasurementProtocol, Outcome, Record


class MeasurementRequest(Record):
    context: ExecutionContext
    # Existing configuration factories own model/benchmark construction. Arguments
    # travel verbatim; the measurement layer does not render another prompt.
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    occurrences: tuple[int, ...] = Field(min_length=1)
    semantics: dict[int, str]
    protocol: MeasurementProtocol | None = None
    characterize: bool = False
    repeats: int = Field(default=1, ge=1, le=100)
    retained_boundaries: dict[int, str] = Field(default_factory=dict)
    save_boundaries: str | None = None


@contextmanager
def open_configuration(factory: str, arguments: dict):
    module, separator, name = factory.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("factory must name a Python module:callable")
    with ExitStack() as resources:
        configuration = getattr(importlib.import_module(module), name)(**arguments)
        if not isinstance(configuration, Configuration):
            configuration = resources.enter_context(configuration)
        if not isinstance(configuration, Configuration):
            raise TypeError("production factory must return or yield a measurement Configuration")
        yield configuration


def describe_configuration(factory: str, arguments: dict):
    from ..formula import FormulaTree

    with open_configuration(factory, arguments) as configuration:
        return {
            "label": configuration.label,
            "graph": configuration.fixture.root.fingerprint,
            "protocol": configuration.protocol.model_dump(mode="json"),
            "scopes": [
                {
                    "occurrence": t.call.occurrence,
                    "formula": t.definition.id,
                    "semantics": t.semantic_identity if t.call.complete else None,
                    "parent": t.call.parent,
                    "complete": t.call.complete,
                }
                for t in FormulaTree(configuration.fixture.root)
            ],
        }


def execute_request(factory: str, request: MeasurementRequest, store: Path) -> None:
    with open_configuration(factory, request.arguments) as original:
        configuration = replace(
            original,
            store=store,
            context=request.context,
            protocol=request.protocol or original.protocol,
        )
        _execute(configuration, request)


def _execute(configuration: Configuration, request: MeasurementRequest):
    with configuration.open() as lab:
        tree = lab.ready.result()
        if request.characterize:
            lab.characterize().result()
        by_occurrence = {target.call.occurrence: target for target in tree}
        targets = []
        for occurrence in request.occurrences:
            target = by_occurrence[occurrence]
            if request.semantics.get(occurrence) != target.semantic_identity:
                raise ValueError("requested scope does not match this production trace")
            targets.append(target)
            if occurrence in request.retained_boundaries:
                lab.boundary(
                    target, Path(request.retained_boundaries[occurrence]), restore=True
                ).result()
            if request.save_boundaries:
                lab.boundary(target, Path(request.save_boundaries) / f"{occurrence}.zip").result()
        # One live device owner retains fixtures and compiled programs across the
        # whole request. Exit ends residency; persisted evidence never claims otherwise.
        for repeat in range(request.repeats):
            for target in targets:
                print(
                    json.dumps(
                        {"phase": "measure", "repeat": repeat, "occurrence": target.call.occurrence}
                    ),
                    flush=True,
                )
                result = lab.measure(target).result.result()
                print(result.model_dump_json(), flush=True)
                if result.outcome != Outcome.COMPLETE:
                    if (
                        configuration.protocol.measure_invalid
                        and result.measurement is not None
                        and (result.error or "").startswith("NumericalMismatch:")
                    ):
                        from .store import ObservationStore

                        with ObservationStore(configuration.store, read_only=True) as store:
                            if store.measurement(result.measurement).observed_seconds is not None:
                                continue
                    raise RuntimeError(result.error or result.outcome.value)
