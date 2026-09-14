"""One physical definition per formula, with ordinary explicit composition."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, replace

from .compiler.dependencies import code_dependencies
from .compiler.lowering import BoundOperation, LoweringContext, order_operations
from .compiler.program import operation_definition
from .formula import Formula, FormulaCall, FormulaRef
from .tensor.graph import Graph
from .tensor.types import TensorSpec


@dataclass(frozen=True, slots=True)
class _KernelBody:
    body: Callable

    def __call__(self, operands):
        self.body(*operands)


@dataclass(frozen=True, slots=True)
class Operation:
    formula: FormulaRef
    body: Callable[[OperationContext], tuple[BoundOperation, ...]]

    def build(self, context: OperationContext) -> tuple[BoundOperation, ...]:
        body = self.body
        # Reload definitions, not live owners. Top-level bodies resolve to their
        # loaded revision; closures remain explicit bound definitions.
        name = getattr(body, "__qualname__", None)
        if name is not None and "<locals>" not in name:
            current = sys.modules.get(body.__module__)
            for part in name.split("."):
                current = getattr(current, part, None)
            if callable(current):
                body = current
        result = body(context)
        if not isinstance(result, tuple) or any(not isinstance(item, BoundOperation) for item in result):
            raise TypeError("an operation body must return its composed physical operations")
        dependencies = code_dependencies(body)
        return tuple(replace(item, dependencies=tuple(dict.fromkeys((*item.dependencies, *dependencies))))
                     for item in result)


_DEFINITIONS: dict[FormulaRef, Operation] = {}


def operation(formula: Formula):
    """Define a formula's operation. Duplicate definitions are an error."""
    if not isinstance(formula, Formula):
        raise TypeError("operation registration requires a Formula object")

    def decorate(body):
        if formula.ref in _DEFINITIONS:
            raise ValueError(f"formula {formula.ref.id!r} already has an operation")
        definition = Operation(formula.ref, body)
        _DEFINITIONS[formula.ref] = definition
        return definition
    return decorate


@dataclass(frozen=True, slots=True)
class OperationContext:
    graph: Graph
    call: FormulaCall | None
    lowering: LoweringContext
    nodes: frozenset[int]

    @property
    def inputs(self):
        """Typed effective inputs in first-use order, derived from the formula."""
        identities = dict.fromkeys(value for node in sorted(self.nodes)
                                   for value in self.graph.node(node).inputs
                                   if self.graph.value(value).producer not in self.nodes)
        return tuple(self.graph.value(value) for value in identities)

    @property
    def outputs(self):
        # Preserve the formula's return-port order, even when it differs from
        # producer order. Additional exposed intermediates follow in graph order.
        declared = tuple(port.value for port in self.call.outputs) if self.call else self.graph.outputs
        identities = dict.fromkeys(value for value in declared if self.graph.value(value).producer in self.nodes)
        for node in sorted(self.nodes):
            for value in self.graph.node(node).outputs:
                if any(user not in self.nodes for user in self.graph.users[value]):
                    identities.setdefault(value, None)
        return tuple(self.graph.value(value) for value in identities)

    def kernel(self, body: Callable, *, workspace: tuple[TensorSpec, ...] = (), kernels: int = 1):
        """Implement this boundary with one authored TileLang body.

        Arguments are effective formula inputs, outputs, then scratch tensors.
        Specs, aliases and mathematical node ownership come from this formula;
        the body declares physical scratch but no duplicated metric equations.
        Device control flow is authored with TileLang's @T.macro, or a
        host callable composing such macros. A body may contain several
        ordered T.Kernel regions; ops does not add another Python tracer.
        """
        inputs, outputs = self.inputs, self.outputs
        resource_inputs = {value.resource_id: value.id for value in inputs if value.resource_id is not None}
        aliases = tuple((value.id, resource_inputs[value.resource_id]) for value in outputs
                        if value.resource_id in resource_inputs)
        name = self.call.formula.id if self.call else self.graph.name
        return (BoundOperation(name, self.nodes, tuple(value.id for value in inputs),
                               tuple(value.id for value in outputs), _KernelBody(body),
                               workspace=workspace, aliases=aliases, kernel_count=kernels,
                               dependencies=code_dependencies(body)),)

    def compose(self, *authored: BoundOperation) -> tuple[BoundOperation, ...]:
        """Compose remaining children around explicit authored work, not alternatives."""
        covered = set()
        for item in authored:
            if not item.nodes <= self.nodes or item.nodes & covered:
                raise ValueError("authored operation overlaps or escapes its formula scope")
            covered.update(item.nodes)
        result = list(authored)
        children = self.graph.formulas.children(self.call.occurrence if self.call else None)
        roots = {}
        for call in children:
            if call.nodes:
                roots.setdefault(min(call.nodes), []).append(call)
        for node in sorted(self.nodes):
            if node in covered:
                continue
            child = next((call for call in roots.get(node, ())
                          if set(call.nodes) <= self.nodes - covered), None)
            if child is not None:
                scope = OperationContext(self.graph, child, self.lowering, frozenset(child.nodes))
                definition = _DEFINITIONS.get(child.formula) if child.complete else None
                built = definition.build(scope) if definition is not None else scope.compose()
            else:
                from .kernels import build_primitive

                built = build_primitive(self.graph, node, self.lowering, remaining=self.nodes - covered)
            if not built:
                raise ValueError(f"operation did not implement primitive {self.graph.node(node).operation}")
            added = set()
            for item in built:
                if not item.nodes <= self.nodes - covered or item.nodes & added:
                    raise ValueError("composed operation overlaps or escapes the mathematical scope")
                added.update(item.nodes)
            if child is not None and added != set(child.nodes):
                raise ValueError("formula operation does not implement its complete mathematical body")
            result.extend(built)
            covered.update(added)
        if covered != self.nodes:
            raise ValueError("incomplete formula operation composition")
        return tuple(result)


def build_operations(graph: Graph, context: LoweringContext) -> tuple[BoundOperation, ...]:
    unsupported = {value.spec.dtype for value in graph.values
                   if value.spec.dtype not in context.capabilities.supported_dtypes}
    if unsupported:
        raise ValueError(f"device cannot represent graph dtypes {sorted(item.value for item in unsupported)}")
    root = OperationContext(graph, None, context, frozenset(range(len(graph.nodes))))
    operations = order_operations(graph, root.compose())
    result = []
    for item in operations:
        if item.workspace_bytes > context.workspace_limit:
            raise ValueError(f"{item.name} workspace exceeds available capacity")
        if item.kernel_count and item.source_loop is None:
            item = replace(item, definition=operation_definition(graph, item))
        result.append(item)
    return tuple(result)
