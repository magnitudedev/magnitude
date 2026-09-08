"""Generate connections between completed Metal tiles and captured scalar graphs."""

from collections import defaultdict

from ._emitter import Binary, Index, Literal, Symbol, emit, expression
from .assembly import source_files
from .elementwise import address, scalar_program
from .graph import Graph, signature
from .kernel import BoundKernel
from .metal import (
    Binding,
    Distributed,
    Load,
    OrderedFold,
    ReadOnly,
    Sample,
    State,
    TileCall,
    argument,
    dtype_name,
)
from .plan import Launch


def compatible(binding: Binding):
    interface = binding.interface
    assert isinstance(interface, (TileCall, OrderedFold))
    return interface.domain, interface.scope, type(interface.result)


def _fold(nodes, bindings, roots, serial):
    first = bindings[nodes[0]].interface
    domain = first.domain
    fields = [f"A{i} {v.name};" for i, v in enumerate(roots)]
    parameters = ", ".join(f"typename A{i}" for i in range(len(roots)))
    argument_types = ", ".join(f"decltype({v.name})" for v in roots)
    fields.extend(f"uint coord_{name};" for name, _ in domain.axes)
    state, initial, updates, finals, results = [], [], [], [], []
    for index, node in enumerate(nodes):
        interface = bindings[node].interface
        carried = interface.state[0]
        name = f"s{index}"
        state.append(f"{dtype_name(carried.dtype)} {name};")
        initial.append(argument(interface.initial[carried]))

        def hook_arg(arg, state_name=name):
            if isinstance(arg, State):
                return f"state.{state_name}"
            if isinstance(arg, Sample):
                return "sample"
            return argument(arg)

        step = interface.step
        finish = interface.finish
        updates.append(
            f"state.{name} = {step.function}({', '.join(map(hook_arg, step.arguments))});"
        )
        finals.append(
            f"{dtype_name(interface.result.dtype)}({finish.function}"
            f"({', '.join(map(hook_arg, finish.arguments))}))"
        )
        results.append(f"{dtype_name(interface.result.dtype)} r{index};")
    typename = f"FoldBody{serial}"
    header = f"""template<{parameters}>
struct {typename} {{
    {" ".join(fields)}
    struct State {{ {" ".join(state)} }};
    struct Result {{ {" ".join(results)} }};
    State initial() const {{ return {{{", ".join(initial)}}}; }}
    void step(thread State& state, uint {first.iteration.axis},
              {dtype_name(first.shared_inputs[0].view.tensor.dtype)} sample) const {{
        {" ".join(updates)}
    }}
    Result finish(thread State& state) const {{ return {{{", ".join(finals)}}}; }}
}};"""
    captured = [v.name for v in roots] + [f"coord_{name}" for name, _ in domain.axes]
    lines = [
        f"{typename}<{argument_types}> body{serial}{{{', '.join(captured)}}};",
        f"auto fold{serial} = {bindings[nodes[0]].function}"
        f"({', '.join(map(argument, first.arguments.values()))}, body{serial});",
    ]
    values = {}
    for index, node in enumerate(nodes):
        name = f"tile_{node.outputs[0].name}"
        lines.append(f"auto {name} = fold{serial}.r{index};")
        values[node.outputs[0].name] = Symbol(name)
    return header, lines, values


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
        if isinstance(binding.interface, OrderedFold):
            return False
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
    if any(compatible(b) != (domain, scope, type(first.result)) for b in bindings.values()):
        raise ValueError("tile participation or invocation domains disagree")
    if any(
        len(n.outputs) != 1
        or n.outputs[0].tensor.shape != shape
        or n.outputs[0].tensor.dtype != b.interface.result.dtype
        for n, b in bindings.items()
    ):
        raise ValueError("tile result does not cover its operation output")
    if any(v.tensor.shape != shape for v in graph.outputs):
        raise ValueError("tile region output needs an unsupported mapping")
    roots = graph.inputs
    if not connected(graph.nodes, bindings):
        raise ValueError("tile inputs require an array boundary or unsupported exchange")
    sources = {}
    for binding in bindings.values():
        for path, content in source_files(binding.source):
            if path in sources and sources[path] != content:
                raise ValueError(f"conflicting source content: {path}")
            sources[path] = content
    lines = [
        "ushort lane = thread_index_in_simdgroup;",
        "uint index = threadgroup_position_in_grid.x * 32 + lane;"
        if distributed
        else "uint index = threadgroup_position_in_grid.x;",
    ]
    stride = domain.size
    for name, extent in domain.axes:
        stride //= extent
        lines.append(f"uint coord_{name} = (index / {stride}u) % {extent}u;")
    values, headers = {}, []
    folds = defaultdict(list)
    for node, binding in bindings.items():
        interface = binding.interface
        if isinstance(interface, OrderedFold):
            key = (
                binding.source,
                binding.function,
                tuple(interface.arguments.items()),
                interface.iteration,
                interface.shared_inputs,
            )
            folds[key].append(node)
    # Fold bodies are independent of internal values (checked above); their results
    # may be shared by any downstream tile or scalar node in topological order.
    for serial, nodes in enumerate(folds.values()):
        header, calls, outputs = _fold(nodes, bindings, roots, serial)
        headers.append(header)
        lines.extend(calls)
        values.update(outputs)
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
        if isinstance(interface, OrderedFold):
            continue
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
        from .fragments import type_parameters

        suffix = f"<{type_parameters(interface.template)}>" if interface.template else ""
        name = f"tile_{node.outputs[0].name}"
        lines.append(
            f"{dtype_name(interface.result.dtype)} {name} = "
            f"{bindings[node].function}{suffix}({', '.join(args)});"
        )
        values[node.outputs[0].name] = Symbol(name)
    lines.append("{" if distributed else "if (lane == 0) {")
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
            (domain.size if distributed else domain.size * scope.size, 1, 1), (scope.size, 1, 1)
        ),
        description=f"{len(bindings)} Metal tiles; {len(folds)} shared ordered drivers; "
        f"{len(scalar)} scalar operations; {transfers} local exchanges; "
        f"completed {scope.size}-lane results",
    )
