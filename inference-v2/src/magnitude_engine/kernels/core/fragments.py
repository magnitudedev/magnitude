"""Connect complete blocked fragments without architecture-specific graph matching."""

from ._emitter import Index, Symbol, emit, expression
from .assembly import source_files
from .elementwise import address, scalar_program
from .kernel import BoundKernel
from .metal import (
    ArgumentType,
    ColumnStart,
    FragmentFold,
    GroupPosition,
    RowIndices,
    SIMDIndex,
    argument,
    dtype_name,
)
from .plan import Launch


def fragment_argument(value):
    if isinstance(value, GroupPosition):
        return f"threadgroup_position_in_grid.{value.axis}"
    if isinstance(value, RowIndices):
        return "logical_rows"
    if isinstance(value, ColumnStart):
        return "first"
    if isinstance(value, SIMDIndex):
        return "simdgroup_index_in_threadgroup"
    return argument(value)


def lower_fragments(graph, bindings):
    first = next(iter(bindings.values())).interface
    layout = first.layout
    roots = graph.inputs
    sources, values = {}, {}
    row = f"{layout.permutation.name}[base + r]" if layout.permutation else "base + r"
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        f"uint base = threadgroup_position_in_grid.z * {layout.tile_rows};",
        f"uint first = threadgroup_position_in_grid.y * {layout.channels * layout.groups}"
        f" + simdgroup_index_in_threadgroup * {layout.channels};",
        f"int logical_rows[{layout.tile_rows}];",
        f"for (uint r = 0; r < {layout.tile_rows}; ++r)"
        f" logical_rows[r] = base + r < {layout.rows} ? int({row}) : -1;",
    ]
    folds = {}
    adapters = []
    for i, (node, binding) in enumerate(bindings.items()):
        interface = binding.interface
        if interface.layout != layout or interface.output != first.output:
            raise ValueError("fragment ownership or output geometry disagrees")
        for path, text in source_files(binding.source):
            if path in sources and sources[path] != text:
                raise ValueError(f"conflicting Metal dependency: {path}")
            sources[path] = text
        if isinstance(interface, FragmentFold):
            key = (
                binding.source,
                binding.function,
                tuple(interface.arguments.items()),
                interface.template,
                interface.pack,
            )
            folds.setdefault(key, []).append(i)
        else:
            template = type_parameters(interface.template)
            arguments = ", ".join(fragment_argument(a) for a in interface.arguments.values())
            lines.append(f"auto tile{i} = {binding.function}<{template}>({arguments});")
        values[node.outputs[0].name] = Symbol(f"local{i}")
    for serial, indices in enumerate(folds.values()):
        header, calls = compose_fold(list(bindings.items()), indices, serial)
        adapters.append(header)
        lines.extend(calls)
    lines += [
        f"for (uint r = 0; r < {layout.tile_rows}; ++r) {{",
        "int row = logical_rows[r];",
        f"for (uint c = 0; c < {layout.channels}; ++c) {{",
    ]
    for i, (_node, _) in enumerate(bindings.items()):
        lines.append(f"auto local{i} = tile{i}.values[r * {layout.channels} + c];")
    lines.append(
        f"if (lane == 0 && row >= 0 && row < {layout.rows} && first + c < {layout.columns}) {{"
    )
    lines.append(f"size_t index = size_t(row) * {layout.columns} + first + c;")
    scalar = [n for n in graph.nodes if n not in bindings]
    needed = {v.name for n in scalar for v in n.inputs} | {v.name for v in graph.outputs}
    values.update(
        (
            v.name,
            Symbol(v.name)
            if not v.tensor.shape
            else Index(Symbol(v.name), address(v, first.output.shape, Symbol("index"))),
        )
        for v in roots
        if v.name in needed
    )
    types = {v.tensor.dtype: dtype_name(v.tensor.dtype) for n in graph.nodes for v in n.outputs}
    statements, values = scalar_program(scalar, values, types)
    lines.append(emit(tuple(statements)))
    for i, output in enumerate(graph.outputs):
        lines.append(f"out{i}[index] = {expression(values[output.name])};")
    lines.append("} } }")
    from .graph import signature

    return BoundKernel(
        roots,
        signature(
            tuple(f"out{i}" for i in range(len(graph.outputs))),
            tuple(v.tensor for v in graph.outputs),
        ),
        "\n".join(lines),
        "\n".join((*sources.values(), *adapters)),
        tuple(sources.items()),
        Launch(
            (
                64,
                (layout.columns + 7) // 8,
                (layout.rows + layout.tile_rows - 1) // layout.tile_rows,
            ),
            (64, 1, 1),
        ),
        description=f"{len(bindings)} completed blocked tiles; scalar finalization; "
        f"{layout.tile_rows} independent rows per tile; {len(folds)} shared pack drivers",
    )


def type_parameters(template):
    return ", ".join(
        f"decltype({fragment_argument(v.argument)})"
        if isinstance(v, ArgumentType)
        else str(v).lower()
        if isinstance(v, (str, int))
        else dtype_name(v)
        for v in template
    )


def compose_fold(bindings, indices, serial):
    """Compose independent handwritten step types; the numerical loop stays Metal."""
    interfaces = [bindings[i][1].interface for i in indices]
    first = interfaces[0]
    concrete_types = [f"{s.body.name}<{type_parameters(s.body.template)}>" for s in interfaces]
    types = [f"B{i}" for i in range(len(interfaces))]
    members = [f"{t} b{i};" for i, t in enumerate(types)]
    state = [f"typename {t}::State s{i};" for i, t in enumerate(types)]
    result = [f"typename {t}::Result r{i};" for i, t in enumerate(types)]
    name = f"CombinedStep{serial}"
    header = f"""template<{", ".join(f"typename {t}" for t in types)}>
struct {name} {{
    {" ".join(members)}
    struct State {{ {" ".join(state)} }};
    struct Result {{ {" ".join(result)} }};
    void prepare(uint k, uint first) {{
        {" ".join(f"b{i}.prepare(k, first);" for i in range(len(types)))}
    }}
    void step(thread State& state, uint row, const thread float* pack, float sum) const {{
        {" ".join(f"b{i}.step(state.s{i}, row, pack, sum);" for i in range(len(types)))}
    }}
    Result finish(thread State& state, uint lane) const {{
        return {{{", ".join(f"b{i}.finish(state.s{i}, lane)" for i in range(len(types)))}}};
    }}
}};"""
    members = [
        "{" + ", ".join(fragment_argument(a) for a in s.body.arguments) + "}" for s in interfaces
    ]
    lines = [
        f"{name}<{', '.join(concrete_types)}> body{serial}{{{', '.join(members)}}};",
        f"auto folded{serial} = {bindings[indices[0]][1].function}"
        f"<{type_parameters(first.template)}>("
        f"{', '.join(fragment_argument(a) for a in first.arguments.values())}, body{serial});",
    ]
    lines.extend(f"auto tile{index} = folded{serial}.r{i};" for i, index in enumerate(indices))
    return header, lines


def lower_ordered(graph, producer, production, reduction, fold):
    interface = production.interface
    if isinstance(interface, FragmentFold) or interface.layout.tile_rows != 1:
        raise ValueError("ordered fragment connection requires a complete single-row producer")
    if fold.interface.input != producer.outputs[0]:
        raise ValueError("ordered fold must consume its connected producer")
    shape = fold.interface.output.shape
    slots = fold.interface.input.tensor.shape[-2]
    columns, rows = shape[-1], fold.interface.output.size // shape[-1]
    roots = graph.inputs
    fields = [f"A{i} {v.name};" for i, v in enumerate(roots)]
    parameters = ", ".join(f"typename A{i}" for i in range(len(roots)))
    argument_types = ", ".join(f"decltype({v.name})" for v in roots)
    dtype = dtype_name(fold.interface.output.dtype)
    arguments = ", ".join(fragment_argument(a) for a in interface.arguments.values())
    call = f"{production.function}<{type_parameters(interface.template)}>({arguments})"
    header = f"""template<{parameters}>
struct OrderedProducer {{
    {" ".join(fields)}
    MagnitudeFragment<{dtype}, 4> operator()(uint row, uint first, uint lane) const {{
        int logical_rows[1] = {{int(row)}};
        return {call};
    }}
}};"""
    step = fold.interface.body
    captures = ", ".join(v.name for v in roots)
    step_args = ", ".join(fragment_argument(a) for a in step.arguments)
    source = f"""OrderedProducer<{argument_types}> producer{{{captures}}};
{step.name}<{type_parameters(step.template)}> reducer{{{step_args}}};
uint row = threadgroup_position_in_grid.z;
uint first = threadgroup_position_in_grid.y * 8 + simdgroup_index_in_threadgroup * 4;
uint lane = thread_index_in_simdgroup;
auto result = {fold.function}<{slots}>(row, first, lane, producer, reducer);
if (lane == 0) {{
    for (uint c = 0; c < 4; ++c)
        if (first + c < {columns}) out0[size_t(row) * {columns} + first + c] = result.values[c];
}}"""
    sources = {}
    for s in (production.source, fold.source):
        for path, text in source_files(s):
            if path in sources and sources[path] != text:
                raise ValueError(f"conflicting Metal dependency: {path}")
            sources[path] = text
    from .graph import signature

    return BoundKernel(
        roots,
        signature(("out0",), (fold.interface.output,)),
        source,
        "\n".join((*sources.values(), header)),
        tuple(sources.items()),
        Launch((64, (columns + 7) // 8, rows), (64, 1, 1)),
        description="completed producer tiles → ordered slot reduction; no intermediate array",
    )
