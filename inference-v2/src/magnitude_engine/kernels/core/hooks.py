"""Generate typed input/output functors for cooperative Metal row drivers."""

from functools import partial

from ._emitter import Index, Symbol, emit, expression
from .assembly import Captures, source_files
from .elementwise import address, scalar_program, supported
from .fragments import fragment_argument, type_parameters
from .graph import signature
from .kernel import BoundKernel
from .metal import ReadOnly, RowTransform, Scratch, ThreadPosition, dtype_name
from .plan import Launch
from .proposal import Proposal


def row_transform(graph, node, binding):
    interface = binding.interface
    shape, dtype = interface.output.shape, interface.output.dtype
    width, threads = shape[-1], interface.scope.size
    roots = graph.inputs
    types = {v.tensor.dtype: dtype_name(v.tensor.dtype) for n in graph.nodes for v in n.outputs}
    types.update((v.tensor.dtype, dtype_name(v.tensor.dtype)) for v in roots)
    native = dtype_name(dtype)
    index = Symbol("index")
    root_values = {
        v.name: Symbol(v.name)
        if not v.tensor.shape
        else Index(Symbol(v.name), address(v, shape, index))
        for v in roots
    }
    scalar = [n for n in graph.nodes if n != node]
    needed = {interface.input.name}
    preparation = []
    for n in reversed(scalar):
        if any(v.name in needed for v in n.outputs):
            preparation.append(n)
            needed.update(v.name for v in n.inputs)
    preparation.reverse()
    prep, prepared = scalar_program(preparation, dict(root_values), types)
    values = dict(root_values)
    values[node.outputs[0].name] = Symbol("result")
    values[interface.input.name] = Symbol("original")
    final, values = scalar_program(
        [n for n in scalar if n.outputs[0].name != interface.input.name], values, types
    )
    captured_names = [v.name for v in roots] + [f"out{i}" for i in range(len(graph.outputs))]
    captured = Captures(tuple(captured_names))
    fields = list(captured.fields)
    row_members = ""
    if interface.row_api:
        fields.extend(["uint row_index;", "uint thread_index;", "uint lane;", "uint simd_group;"])
        row_members = f"""
    using Value = {native};
    static constexpr constant uint width = {width};
    static constexpr constant uint threads = {threads};
    {native} load(uint column) const {{ return input(size_t(row_index) * width + column); }}
    void store(uint column, {native} result, {native} original) const {{
        output(size_t(row_index) * width + column, result, original);
    }}
"""
    parameters, argument_types = captured.parameters, captured.arguments
    stores = "\n".join(
        f"out{i}[index] = {expression(values[v.name])};" for i, v in enumerate(graph.outputs)
    )
    header = f"""template<{parameters}>
struct RowBody {{
    {" ".join(fields)}
    {row_members}
    {native} input(size_t index) const {{
        {emit(tuple(prep))}
        return {expression(prepared[interface.input.name])};
    }}
    void output(size_t index, {native} result, {native} original) const {{
        {emit(tuple(final))}
        {stores}
    }}
}};"""
    captures = [v.name for v in roots] + [f"out{i}" for i in range(len(graph.outputs))]
    if interface.row_api:
        captures.extend(
            [
                "threadgroup_position_in_grid.y",
                "thread_position_in_threadgroup.x",
                "thread_index_in_simdgroup",
                "simdgroup_index_in_threadgroup",
            ]
        )
    scratch, arguments = [], ["body"]
    for name, value in interface.arguments.items():
        if isinstance(value, Scratch):
            symbol = f"scratch_{name}"
            scratch.append(f"threadgroup {dtype_name(value.dtype)} {symbol}[{value.count}];")
            arguments.append(symbol)
        elif isinstance(value, ThreadPosition):
            arguments.append(f"thread_position_in_threadgroup.{value.axis}")
        else:
            arguments.append(fragment_argument(value))
    # Lane is a typed argument rather than an implicit authored global.
    body = "uint lane = thread_index_in_simdgroup;\n"
    body += f"RowBody<{argument_types}> body{{{', '.join(captures)}}};\n"
    body += "\n".join(scratch) + "\n"
    template = f"<{type_parameters(interface.template)}>" if interface.template else ""
    body += f"{binding.function}{template}({', '.join(arguments)});"
    sources = source_files(binding.source)

    return BoundKernel(
        roots,
        signature(
            tuple(f"out{i}" for i in range(len(graph.outputs))),
            tuple(v.tensor for v in graph.outputs),
        ),
        body,
        "\n".join(text for _, text in sources) + "\n" + header,
        sources,
        Launch((threads, interface.output.size // width, 1), (threads, 1, 1)),
        description="cooperative row driver with graph-derived input/output hooks",
    )


def plan_rows(graph, bindings):
    if len(bindings) != 1:
        return None
    node, binding = next(iter(bindings.items()))
    interface = binding.interface
    if not isinstance(interface, RowTransform):
        return None
    produced = {v for n in graph.nodes for v in n.outputs}
    if any(
        isinstance(a, ReadOnly) and a.view.tensor.value in produced
        for a in interface.arguments.values()
    ):
        return None
    if not all(
        supported(n) and n.outputs[0].tensor.shape == interface.output.shape
        for n in graph.nodes
        if n != node
    ):
        return None
    if any(v.tensor.shape != interface.output.shape for v in graph.outputs):
        return None
    return Proposal(graph, partial(row_transform, graph, node, binding))
