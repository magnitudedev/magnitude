"""Generate connections between completed Metal tiles and captured scalar graphs."""

from functools import partial

from ._emitter import Binary, Index, Literal, Symbol, emit, expression
from .assembly import merge_sources
from .elementwise import address, scalar_program, supported
from .fragments import type_parameters
from .graph import Graph, signature
from .kernel import BoundKernel
from .metal import (
    Binding,
    Distributed,
    Load,
    ReadOnly,
    Thread,
    TileCall,
    argument,
    dtype_name,
)
from .plan import Launch
from .proposal import Proposal


def compatible(binding: Binding):
    interface = binding.interface
    assert isinstance(interface, TileCall)
    return interface.domain, interface.scope, type(interface.result)


def local_transfer(view, domain, distributed):
    coordinates = tuple(i.value for i in domain.indices)
    if view.coordinates == coordinates:
        return 0
    if distributed and view.coordinates[:-1] == coordinates[:-1]:
        match view.coordinates[-1]:
            case Binary("^", axis, Literal(mask)) if axis == coordinates[-1] and 0 <= mask < 32:
                return mask
    raise ValueError("consumer coordinates require an unsupported local exchange")


def connected(nodes, bindings):
    outputs = {v for n in nodes for v in n.outputs}
    first = next(iter(bindings.values())).interface
    distributed = isinstance(first.result, Distributed)
    for node, binding in bindings.items():
        internal = {v for v in node.inputs if v in outputs}
        if not internal:
            continue
        referenced = set()
        for arg in binding.interface.arguments.values():
            if isinstance(arg, (ReadOnly, Load)) and arg.view.tensor.value in internal:
                if not isinstance(arg, Load):
                    return False
                try:
                    local_transfer(arg.view, first.domain, distributed)
                except ValueError:
                    return False
                referenced.add(arg.view.tensor.value)
        if referenced != internal:
            return False
    return True


def lower_tiles(graph: Graph, bindings: dict) -> BoundKernel:
    first = next(iter(bindings.values())).interface
    domain, scope = first.domain, first.scope
    shape = domain.shape
    distributed = isinstance(first.result, Distributed)
    independent = isinstance(scope, Thread)
    roots = graph.inputs
    sources = dict(merge_sources(b.source for b in bindings.values()))
    lines = [
        "ushort lane = thread_index_in_simdgroup;",
        "uint index = thread_position_in_grid.x;"
        if independent or distributed
        else "uint index = threadgroup_position_in_grid.x;",
        f"if (index >= {domain.size}) return;" if independent else "",
    ]
    stride = domain.size
    for name, extent in domain.axes:
        stride //= extent
        lines.append(f"uint coord_{name} = (index / {stride}u) % {extent}u;")
    values, headers = {}, []
    scalar = [n for n in graph.nodes if n not in bindings]
    needed = {v.name for n in scalar for v in n.inputs} | {v.name for v in graph.outputs}
    values.update(
        (
            v.name,
            Symbol(v.name)
            if not v.tensor.shape
            else Index(Symbol(v.name), address(v, shape, Symbol("index"))),
        )
        for v in roots
        if v.name in needed
    )
    produced = {v.name for n in graph.nodes for v in n.outputs}
    types = {v.tensor.dtype: dtype_name(v.tensor.dtype) for n in graph.nodes for v in n.outputs}
    transfers = 0
    for node in graph.nodes:
        if node not in bindings:
            statements, values = scalar_program((node,), values, types)
            lines.append(emit(tuple(statements)))
            continue
        interface = bindings[node].interface
        args = []
        for arg in interface.arguments.values():
            if isinstance(arg, Load) and arg.view.tensor.value.name in produced:
                mask = local_transfer(arg.view, domain, distributed)
                value = expression(values[arg.view.tensor.value.name])
                if mask:
                    value = f"simd_shuffle_xor({value}, {mask})"
                    transfers += 1
                args.append(value)
            else:
                args.append(argument(arg))

        suffix = f"<{type_parameters(interface.template)}>" if interface.template else ""
        name = f"tile_{node.outputs[0].name}"
        lines.append(
            f"{dtype_name(interface.result.dtype)} {name} = "
            f"{bindings[node].function}{suffix}({', '.join(args)});"
        )
        values[node.outputs[0].name] = Symbol(name)
    lines.append("{" if distributed or independent else "if (lane == 0) {")
    for i, output in enumerate(graph.outputs):
        lines.append(f"    out{i}[index] = {expression(values[output.name])};")
    lines.append("}")
    header = "\n".join((*sources.values(), *headers))
    return BoundKernel(
        roots,
        signature(
            tuple(f"out{i}" for i in range(len(graph.outputs))),
            tuple(v.tensor for v in graph.outputs),
        ),
        "\n".join(lines),
        header,
        tuple(sources.items()),
        Launch(
            (domain.size if distributed or independent else domain.size * scope.size, 1, 1),
            (128 if independent else scope.size, 1, 1),
        ),
        description=f"{len(bindings)} Metal tiles; "
        f"{len(scalar)} scalar operations; {transfers} local exchanges; "
        f"completed {scope.size}-lane results",
    )


def plan_tiles(graph, bindings):
    if not bindings or not all(isinstance(b.interface, TileCall) for b in bindings.values()):
        return None
    domain, scope, result = compatible(next(iter(bindings.values())))
    if not all(compatible(b) == (domain, scope, result) for b in bindings.values()):
        return None
    if not all(
        supported(n) and n.outputs[0].tensor.shape == domain.shape
        for n in graph.nodes
        if n not in bindings
    ):
        return None
    if any(v.tensor.shape != domain.shape for v in graph.outputs) or not connected(
        graph.nodes, bindings
    ):
        return None
    return Proposal(graph, partial(lower_tiles, graph, bindings))
