"""Persist the actual formula composition for device-free historical browsing."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import Literal

from pydantic import Field, model_validator

from ..formula import FormulaRef, FormulaTree
from .records import Record


class RecordedFormula(Record):
    occurrence: int = Field(ge=0)
    definition: FormulaRef
    semantics: str | None = Field(default=None, min_length=1)
    parent: int | None = None
    dependencies: tuple[int, ...] = ()
    complete: bool


class RecordedConfiguration(Record):
    schema_version: Literal[1] = 1
    identity: str = Field(min_length=1)
    label: str = Field(min_length=1)
    created: datetime
    device: str = Field(min_length=1)
    formulas: tuple[RecordedFormula, ...]

    @model_validator(mode="after")
    def valid_composition(self):
        if self.created.utcoffset() is None:
            raise ValueError("recorded configuration requires a timezone")
        known = set()
        for formula in self.formulas:
            if formula.complete != (formula.semantics is not None):
                raise ValueError("only complete recorded formulas have comparable semantics")
            if formula.occurrence in known:
                raise ValueError("recorded formula occurrence is duplicated")
            if formula.parent is not None and formula.parent not in known:
                raise ValueError("recorded formula parent must precede its child")
            known.add(formula.occurrence)
        if any(dependency not in known or dependency == formula.occurrence
               for formula in self.formulas for dependency in formula.dependencies):
            raise ValueError("recorded formula dependency is not another occurrence")
        return self

    @classmethod
    def capture(cls, tree: FormulaTree, *, label: str, device: str):
        # Match ports from the existing trace; do not invent another component DAG.
        graph = tree.graph
        calls = tuple(graph.formulas)
        output_owners = {}
        for call in calls:
            for port in call.outputs:
                output_owners.setdefault(port.value, set()).add(call.occurrence)
        records = []
        for target in tree:
            call = target.call
            ancestors = set()
            parent = target.parent
            while parent is not None:
                ancestors.add(parent.call.occurrence)
                parent = parent.parent
            dependencies = set()
            for port in call.inputs:
                dependencies.update(output_owners.get(port.value, ()))
            # A pass-through parent/descendant is composition, not an independent
            # producer. Only earlier completed scopes supply external inputs.
            dependencies = {identity for identity in dependencies
                            if identity < call.occurrence and identity not in ancestors}
            records.append(RecordedFormula(occurrence=call.occurrence, definition=target.definition,
                                           semantics=target.semantic_identity if call.complete else None, parent=call.parent,
                                           dependencies=tuple(sorted(dependencies)), complete=call.complete))
        return cls(identity=str(uuid4()), label=label, created=datetime.now(UTC),
                   device=device, formulas=tuple(records))


@dataclass(frozen=True, slots=True)
class StoredConfiguration:
    store: Path
    record: RecordedConfiguration

    @property
    def label(self):
        return f"Recorded · {self.record.label} · {self.record.created.isoformat()} · {self.record.device}"


def browse(path: Path) -> None:
    """Open recorded configurations without a production fixture or live runtime."""
    from .store import ObservationStore
    from .tui import ConfigurationApp, RecordedApp

    store = ObservationStore(path)
    try:
        while (selected := ConfigurationApp(store.configurations(), recorded=True).run()) is not None:
            result = RecordedApp(store, selected).run()
            if result != "choose":
                return
    finally:
        store.close()
