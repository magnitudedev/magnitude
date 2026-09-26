"""Array dependencies between independently lowered numerical regions."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

import mlx.core as mx

from .graph import Graph
from .kernel import BoundKernel, ConstantInputs


@dataclass(frozen=True)
class Region:
    graph: Graph
    backend: str
    call: Callable[..., tuple[mx.array, ...]]


@dataclass(frozen=True)
class OperandBinding:
    """Preserve a bound implementation while mapping repeated operands/results."""

    call: Callable[..., tuple[mx.array, ...]]
    operands: tuple[int, ...]
    results: tuple[int, ...]

    def __call__(self, *arrays):
        outputs = self.call(*(arrays[i] for i in self.operands))
        return tuple(outputs[i] for i in self.results)


@dataclass(frozen=True)
class ExecutionPlan:
    graph: Graph
    regions: tuple[Region, ...]

    @cached_property
    def compiled(self):
        def run(*inputs):
            values = {v.name: a for v, a in zip(self.graph.inputs, inputs, strict=True)}
            values.update((v.name, a) for v, a in self.graph.constants)
            for region in self.regions:
                outputs = region.call(*(values[v.name] for v in region.graph.inputs))
                values.update(
                    (v.name, a) for v, a in zip(region.graph.outputs, outputs, strict=True)
                )
            return tuple(values[v.name] for v in self.graph.outputs)

        # Iteration constructs an MLX graph once. Warm calls execute MLX's compiled
        # function, not this interpreter, including when nested in another compile.
        return mx.compile(run)

    def explain(self):
        owners = {v: i for i, region in enumerate(self.regions) for v in region.graph.outputs}
        lines = []
        for i, region in enumerate(self.regions):
            inputs = ", ".join(v.name for v in region.graph.inputs)
            outputs = ", ".join(v.name for v in region.graph.outputs)
            lines.append(
                f"{i + 1}. {region.backend} ({inputs}) → ({outputs}): "
                + getattr(region.call, "description", region.graph.describe())
            )
            materialized = [v.name for v in region.graph.inputs if v in owners and owners[v] != i]
            if materialized:
                lines.append(
                    "   Array boundary for "
                    + ", ".join(materialized)
                    + ": retained by the qualified layout/connection and bounded fusion policy."
                )
        return "\n".join(lines)

    def artifact(self):
        """Export the actual compiler plan and sources without executing it."""

        def value(v):
            return {"name": v.name, "shape": v.tensor.shape, "dtype": str(v.tensor.dtype)}

        regions = []
        for region in self.regions:
            call = region.call.kernel if isinstance(region.call, ConstantInputs) else region.call
            mapping = None
            if isinstance(call, OperandBinding):
                mapping = {"operands": call.operands, "results": call.results}
                call = call.call
            record = {
                "backend": region.backend,
                "description": getattr(call, "description", region.graph.describe()),
                "inputs": [value(v) for v in region.graph.inputs],
                "outputs": [value(v) for v in region.graph.outputs],
                "operations": [
                    str(n.operation)
                    if isinstance(n.operation, str)
                    else type(n.operation).__module__ + "." + type(n.operation).__qualname__
                    for n in region.graph.nodes
                ],
            }
            if mapping is not None:
                record["argument_binding"] = mapping
            if isinstance(call, BoundKernel):
                record.update(
                    source=call.source,
                    header=call.header,
                    sources=dict(call.sources),
                    grid=call.launch.grid,
                    threadgroup=call.launch.threadgroup,
                    templates=[(k, str(v)) for k, v in call.template],
                    math_mode="safe",
                    input_layout="explicit row-contiguous MLX operands",
                )
            regions.append(record)
        return {"version": 1, "regions": regions}

    def __call__(self, *arrays):
        return tuple(self.compiled(*arrays))


def select(graph: Graph, nodes) -> Graph:
    """Extract a dependency region, preserving every externally observed result."""
    selected = set(nodes)
    ordered = tuple(n for n in graph.nodes if n in selected)
    internal = {v for n in ordered for v in n.outputs}
    inputs = tuple(dict.fromkeys(v for n in ordered for v in n.inputs if v not in internal))
    used = {v for n in graph.nodes if n not in selected for v in n.inputs} | set(graph.outputs)
    outputs = tuple(v for n in ordered for v in n.outputs if v in used)
    return Graph(inputs, outputs, (), ordered)


def order_regions(graph: Graph, groups):
    """Topologically order a partition; merging across a dependency is rejected."""
    groups = [tuple(g) for g in groups]
    owners = {v: i for i, group in enumerate(groups) for n in group for v in n.outputs}
    dependencies = [
        {owners[v] for n in group for v in n.inputs if v in owners and owners[v] != i}
        for i, group in enumerate(groups)
    ]
    remaining, order = set(range(len(groups))), []
    while remaining:
        ready = [i for i in sorted(remaining) if not dependencies[i] & remaining]
        if not ready:
            raise ValueError("fusion would introduce a dependency cycle")
        order.extend(ready)
        remaining.difference_update(ready)
    return [groups[i] for i in order]
