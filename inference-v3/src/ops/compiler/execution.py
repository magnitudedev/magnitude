"""Physical actions: source I/O, transfers, numerical units and retirement boundaries."""

from dataclasses import dataclass
from enum import StrEnum

from ..binding import Residency
from ..runtime.imports import ImportPlan, plan_import


class ActionKind(StrEnum):
    IMPORT = "import"
    EXECUTE = "execute"
    WAIT = "wait"
    RETIRE = "retire"
    SOURCE_LOOP = "source-loop"


@dataclass(frozen=True, slots=True)
class Action:
    index: int
    kind: ActionKind
    predecessors: tuple[int, ...]
    unit: int | None = None
    value: int | None = None
    import_plan: ImportPlan | None = None
    source_loop: object | None = None


@dataclass(frozen=True, slots=True)
class ExecutionGraph:
    actions: tuple[Action, ...]
    streamed_by_unit: tuple[tuple[int, ...], ...]


def execution_graph(graph, submissions, bindings):
    actions = []
    prior = ()
    def emit(kind, *, unit=None, value=None, import_plan=None, source_loop=None):
        nonlocal prior
        action = Action(len(actions), kind, prior, unit, value, import_plan, source_loop)
        actions.append(action)
        prior = (action.index,)

    for value, binding in bindings.items():
        if binding.residency == Residency.RESIDENT:
            emit(ActionKind.IMPORT, value=value, import_plan=plan_import(binding))
    sources_by_unit = []
    for unit in submissions:
        loop = unit.operations[0].source_loop if len(unit.operations) == 1 else None
        sources = tuple(sorted({value for operation in unit.operations for value in operation.inputs
                                if value in bindings and bindings[value].residency == Residency.STREAMED
                                and (loop is None or value not in loop.source_values)}))
        sources_by_unit.append(sources)
        if sources and unit.index:
            emit(ActionKind.WAIT, unit=unit.index - 1)
        for value in sources:
            emit(ActionKind.IMPORT, unit=unit.index, value=value, import_plan=plan_import(bindings[value]))
        emit(ActionKind.SOURCE_LOOP if loop is not None else ActionKind.EXECUTE,
             unit=unit.index, source_loop=loop)
        if sources:
            emit(ActionKind.WAIT, unit=unit.index)
            for value in sources:
                emit(ActionKind.RETIRE, unit=unit.index, value=value)
    return ExecutionGraph(tuple(actions), tuple(sources_by_unit))
