"""Source-tiled operation realizations. Logical ports retain their full shapes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..binding import Binding, Residency
from ..kernels.schedules import AffineSchedule, select_affine_tile
from ..runtime.imports import plan_import
from ..tensor.types import DType, TensorSpec
from .dependencies import code_dependencies
from .program import KernelDefinition, KernelPort, PortRole, define_kernel


@dataclass(frozen=True, slots=True)
class GatherTileEmitter:
    table: TensorSpec
    output: TensorSpec
    step: int
    threads: int

    def __call__(self, operands):
        import tilelang.language as T

        from ..kernels.indexing import _streamed_embedding

        table, destinations, output = operands
        flat = T.view(output, shape=(self.output.elements // self.table.shape[1], self.table.shape[1]),
                      dtype=self.output.dtype.value)
        _streamed_embedding(table, destinations, flat, self.table, self.step, self.threads)


@dataclass(frozen=True, slots=True)
class GatherLoop:
    """Bounded token chunks with source rows gathered in encoding-aligned groups."""
    weight: int
    hidden: int
    output: int
    tokens: int
    step: int
    alignment: int
    width: int
    source: Binding
    prototype: Binding
    templates: tuple[SourceTemplate, ...]
    peak_bytes: int

    @property
    def source_values(self):
        return (self.weight,)

    def gather(self, rows: tuple[int, ...], *, source: Binding | None = None):
        from ..binding import SegmentedSource, SourcePlane, SourceSpan, ZeroSource

        source = self.source if source is None else source
        if source.spec != self.source.spec or source.recipe != self.source.recipe:
            raise ValueError("gather source changes its numerical/storage geometry")
        if not rows or len(rows) > self.step:
            raise ValueError("gather requires one bounded nonempty token chunk")
        if any(row < 0 or row >= source.spec.shape[0] for row in rows):
            raise ValueError("embedding index is outside the vocabulary")
        groups = tuple(sorted({row // self.alignment for row in rows}))
        locations = {group: index * self.alignment for index, group in enumerate(groups)}
        # Coalesce adjacent groups before touching the underlying byte sources.
        runs = []
        for group in groups:
            if runs and runs[-1][0] + runs[-1][1] == group:
                first, count = runs[-1]
                runs[-1] = first, count + 1
            else:
                runs.append((group, 1))
        planes = []
        for plane, prototype in zip(source.planes, self.prototype.planes, strict=True):
            spans = tuple(plane.region(first * self.alignment * self.width,
                                       count * self.alignment * self.width).span for first, count in runs)
            padding = prototype.span.length - sum(span.length for span in spans)
            if padding:
                spans += (SourceSpan(ZeroSource(padding), 0, padding),)
            segments = SegmentedSource(spans)
            planes.append(SourcePlane(SourceSpan(segments, 0, segments.size),
                                      plane.group_elements, plane.group_bytes))
        binding = Binding(self.prototype.spec, source.value_identity + "/gather/" +
                          ",".join(map(str, groups)), Residency.STREAMED, tuple(planes), source.recipe,
                          root_identity=source.root_identity)
        return binding, tuple(locations[row // self.alignment] + row % self.alignment for row in rows)


def gather_loop(graph, root, context, binding):
    from ..kernels.matrix import _dense
    from ..kernels.packed import packet_format
    from .lowering import BoundOperation

    node = graph.node(root)
    indices, table = (graph.value(value).spec for value in node.inputs)
    output = graph.value(node.outputs[0]).spec
    if node.operation != "embedding" or binding.residency != Residency.STREAMED:
        raise ValueError("source gather requires a streamed embedding binding")
    if not indices.static or not table.static or table.rank != 2 or not indices.dtype.integer:
        raise ValueError("source gather requires concrete integer indices and a matrix table")
    columns, width = table.shape
    if not columns or not width or not indices.elements:
        raise ValueError("source gather requires nonempty token and table geometry")
    alignment = binding.region_alignment // math.gcd(binding.region_alignment, width)
    packet = packet_format(table)
    if columns % alignment or (packet is not None and width % packet.matrix_packet):
        raise ValueError("source gather rows violate encoding alignment")
    if packet is None and not _dense(table):
        raise ValueError("source gather has no decoder for the table representation")
    bytes_per_group = math.ceil(max(binding.source_bytes, table.storage_nbytes) / columns) * alignment
    step = min(indices.elements, columns // alignment, max(1, (8 << 20) // bytes_per_group))
    budget = max(0, context.workspace_limit - output.storage_nbytes)
    while True:
        prototype = binding.region(0, step * alignment * width, shape=(step * alignment, width))
        imported = plan_import(prototype)
        # Index readback, host map, transfer staging and device map are all live
        # operation storage. No full-table allocation occurs.
        peak = (imported.host_peak_bytes + imported.execution_peak_bytes +
                2 * indices.storage_nbytes + 3 * step * 16)
        if peak <= budget:
            break
        if step == 1:
            raise ValueError("one gathered source group exceeds available capacity")
        step = max(1, step // 2)
    emitter = GatherTileEmitter(prototype.spec, output, step, min(128, context.compiler_target.threads_per_group))
    ports = (KernelPort(prototype.spec, PortRole.READ),
             KernelPort(TensorSpec((step, 2), DType.I64), PortRole.READ),
             KernelPort(output, PortRole.WRITE))
    template = SourceTemplate(step, define_kernel(emitter, ports))
    loop = GatherLoop(node.inputs[1], node.inputs[0], node.outputs[0], indices.elements,
                      step, alignment, width, binding, prototype, (template,), peak)
    return BoundOperation(f"embedding.source-gather@{root}", frozenset({root}), node.inputs,
                          node.outputs, emitter, source_loop=loop,
                          kernel_count=math.ceil(indices.elements / step),
                          dependencies=code_dependencies(gather_loop))


@dataclass(frozen=True, slots=True)
class ProjectionTileEmitter:
    hidden: TensorSpec
    weight: TensorSpec
    output: TensorSpec
    strategy: str
    threads: int
    tile: tuple[int, int, int]
    contraction_schedule: AffineSchedule | None
    bias: bool
    outputs_per_subgroup: int = 1

    def __call__(self, operands):
        from ..kernels.matrix import _dense_matrix, _dense_vector, _packed_matrix, _packed_vector

        hidden, weight, extent, output = operands[:4]
        bias = operands[4] if self.bias else hidden
        m, k = self.hidden.shape
        n = self.weight.shape[0]
        if self.strategy == "packet-vector":
            _packed_vector(hidden, weight, bias, output, self.weight, m, n, k, self.output.dtype.value,
                           self.bias, self.threads, self.outputs_per_subgroup, extent)
        elif self.strategy == "dense-vector":
            _dense_vector(hidden, weight, bias, output, m, n, k, self.output.dtype.value, self.bias,
                          extent)
        elif self.strategy == "packet-matrix":
            _packed_matrix(hidden, weight, bias, output, self.weight, m, n, k, self.contraction_schedule,
                           self.output.dtype.value, self.threads, *self.tile, self.bias, extent)
        else:
            _dense_matrix(hidden, weight, bias, output, "linear", m, n, k, self.hidden.dtype.value,
                          self.output.dtype.value, self.threads, *self.tile, self.bias, extent)


@dataclass(frozen=True, slots=True)
class SourceTemplate:
    extent: int
    definition: KernelDefinition


def _expert_program(hidden, *, gate, up, down, activation):
    """Physical decomposition of the routed primitive's publication contract."""
    from ..tensor import ops

    gated = ops.cast(ops.linear(hidden, gate), DType.F32)
    expanded = ops.cast(ops.linear(hidden, up), DType.F32)
    activated = ops.silu(gated) if activation == "silu" else ops.tanh(gated)
    product = ops.cast(activated * expanded, hidden.dtype)
    return ops.linear(product, down)


@dataclass(frozen=True, slots=True)
class ExpertLoop:
    hidden: int
    routes: int
    scores: int
    output: int
    source_values: tuple[int, ...]
    sources: tuple[Binding, ...]
    experts: int
    step: int
    selected: int
    gathered: TensorSpec
    contributions: TensorSpec
    route_map: TensorSpec
    inner_plan: object
    templates: tuple[SourceTemplate, ...]
    peak_bytes: int

    def region(self, source: Binding, expert: int):
        if not 0 <= expert < self.experts:
            raise ValueError("expert route is outside the source bank")
        shape = source.spec.shape[1:]
        elements = math.prod(shape)
        return source.region(expert * elements, elements, shape=shape)


def expert_loop(graph, root, context, bindings):
    from ..kernels.source_experts import CombineExpertRows, GatherExpertRows, ScatterExpertRows
    from ..tensor.graph import ValueKind
    from ..tensor.tracing import Argument, Signature, trace
    from .compilation import CompileOptions, analyze_graph
    from .lowering import BoundOperation

    node = graph.node(root)
    hidden, routes, scores = (graph.value(value).spec for value in node.inputs[:3])
    output = graph.value(node.outputs[0]).spec
    if len(bindings) != 3 or any(binding.residency != Residency.STREAMED for binding in bindings):
        raise ValueError("bounded expert execution requires source bindings for all three expert banks")
    experts = bindings[0].spec.shape[0]
    if any(math.prod(binding.spec.shape[1:]) % binding.region_alignment for binding in bindings):
        raise ValueError("expert regions must contain complete source encoding groups")
    selected = routes.shape[1]
    step = 1 if context.mode == "decode" else min(64, routes.elements)
    if step < 1 or experts < 1:
        raise ValueError("streamed experts require a nonempty route/bank geometry")
    gathered = TensorSpec((step, hidden.shape[1]), hidden.dtype)
    contributions = TensorSpec((routes.elements, hidden.shape[1]), hidden.dtype)
    route_map = TensorSpec((step,), DType.I32)
    if routes.elements > (1 << 31) - 1:
        raise ValueError("expert route map exceeds its declared index width")
    prototypes = {name: binding.region(0, math.prod(binding.spec.shape[1:]), shape=binding.spec.shape[1:])
                  for name, binding in zip(("gate", "up", "down"), bindings, strict=True)}
    signature = Signature((Argument(gathered, "hidden"),),
                          {name: Argument(binding.spec, name, ValueKind.CONSTANT) for name, binding in prototypes.items()},
                          {"activation": node.attributes["activation"]})
    inner_graph = trace(_expert_program, signature)
    outer_storage = (gathered.storage_nbytes + contributions.storage_nbytes +
                     3 * route_map.storage_nbytes + 2 * routes.storage_nbytes)
    inner_budget = context.workspace_limit - output.storage_nbytes - outer_storage
    if inner_budget <= 0:
        raise ValueError("expert row staging exceeds physical capacity")
    inner = analyze_graph(inner_graph, compiler_target=context.compiler_target, compiler_identity=context.compiler_identity,
                          available_bytes=inner_budget, device_identity=context.device_identity,
                          options=CompileOptions(mode=context.mode, precision=context.precision,
                                                 schedules=context.schedules), constants=prototypes)
    from .memory import StorageClass

    inner_storage = inner.memory.temporary_bytes + sum(placement.spec.storage_nbytes for placement in inner.memory.values.values()
                                                       if placement.storage == StorageClass.OUTPUT)
    inner_storage += max((operation.source_loop.peak_bytes for operation in inner.operations
                          if operation.source_loop is not None), default=0)
    threads = min(128, context.compiler_target.threads_per_group)
    definitions = (
        define_kernel(GatherExpertRows(step, hidden.shape[1], selected, threads), (
            KernelPort(hidden, PortRole.READ), KernelPort(route_map, PortRole.READ), KernelPort(gathered, PortRole.WRITE))),
        define_kernel(ScatterExpertRows(step, hidden.shape[1], threads), (
            KernelPort(gathered, PortRole.READ), KernelPort(route_map, PortRole.READ), KernelPort(contributions, PortRole.WRITE))),
        define_kernel(CombineExpertRows(hidden.shape[0], selected, hidden.shape[1], threads), (
            KernelPort(contributions, PortRole.READ), KernelPort(scores, PortRole.READ), KernelPort(output, PortRole.WRITE))),
    )
    loop = ExpertLoop(node.inputs[0], node.inputs[1], node.inputs[2], node.outputs[0], node.inputs[3:6],
                      tuple(bindings), experts, step, selected, gathered, contributions, route_map, inner,
                      tuple(SourceTemplate(index, definition) for index, definition in enumerate(definitions)),
                      outer_storage + inner_storage)
    # Route-dependent dispatch count is an upper bound, never a predicted cost.
    dispatches = routes.elements * (2 + sum(operation.kernel_count for operation in inner.operations)) + 1
    return BoundOperation(f"routed_experts.source@{root}", frozenset({root}), node.inputs, node.outputs,
                          lambda operands: None, source_loop=loop, kernel_count=dispatches,
                          dependencies=code_dependencies(expert_loop))


@dataclass(frozen=True, slots=True)
class ProjectionLoop:
    weight: int
    hidden: int
    output: int
    bias: int | None
    columns: int
    step: int
    width: int
    source: Binding
    templates: tuple[SourceTemplate, ...]
    peak_bytes: int

    @property
    def source_values(self):
        return (self.weight,)

    def regions(self):
        for first in range(0, self.columns, self.step):
            count = min(self.step, self.columns - first)
            yield first, self.source.region(first * self.width, count * self.width,
                                           shape=(count, self.width))

    def template(self, extent):
        return next(template for template in self.templates if template.extent == extent)

def projection_loop(graph, root, context, binding):
    """One bounded source-projection operation, with capacity-constrained regions."""
    from ..kernels.matrix import (
        _dense,
        _packed_vector_geometry,
        _packet_reduction_width,
        matrix_geometry,
    )
    from ..kernels.packed import packet_format
    from .lowering import BoundOperation

    node = graph.node(root)
    if node.operation != "linear" or binding.residency != Residency.STREAMED:
        raise ValueError("source projection requires a streamed linear binding")
    hidden, weight = (graph.value(value).spec for value in node.inputs[:2])
    output = graph.value(node.outputs[0]).spec
    if hidden.rank != 2 or weight.rank != 2 or context.compiler_target.subgroup_width != 32:
        raise ValueError("streamed projection requires a legal rank-two subgroup geometry")
    columns, width = weight.shape
    row_alignment = binding.region_alignment // math.gcd(binding.region_alignment, width)
    if columns % row_alignment:
        raise ValueError("source projection rows violate encoded-region alignment")
    packed = packet_format(weight)
    vector = None
    if packed is not None and width % packed.tile == 0:
        vector = _packed_vector_geometry(weight, context)
    elif _dense(weight):
        if context.compiler_target.threads_per_group >= 128:
            vector = (128, 1)
    else:
        raise ValueError("streamed weight representation has no projection realization")
    if vector is not None and hidden.shape[0] < 8:
        strategy = "packet-vector" if packed else "dense-vector"
        threads, outputs_per_subgroup = vector
        tile, arithmetic = (1, 1, 1), None
    else:
        bk = _packet_reduction_width(weight) if packed else 16
        bm, bn, threads = matrix_geometry(context, hidden.dtype, hidden.shape[0], columns, bk,
                                          packed_specs=(weight,) if packed else (), storage_dtype=hidden.dtype)
        strategy = "packet-matrix" if packed else "dense-matrix"
        tile, arithmetic, outputs_per_subgroup = (bm, bn, bk), None, 1
        if packed:
            schedule = select_affine_tile(
                context, hidden, (weight,), (bm, bn, bk, threads),
                template=ProjectionTileEmitter, name="linear.streamed-affine",
                workload=(output, len(node.inputs) == 3, context.workspace_limit),
            )
            tile = (schedule.rows, schedule.columns, schedule.reduction)
            threads, arithmetic = schedule.threads, schedule.operands
    bytes_per_row = max(1, math.ceil(binding.spec.storage_nbytes / columns))
    step = min(columns, max(row_alignment, ((8 << 20) // bytes_per_row // row_alignment) * row_alignment))
    budget = max(0, context.workspace_limit - output.storage_nbytes)
    # Shrink only to satisfy physical capacity, never to score estimated speed.
    while True:
        extents = sorted({step, columns % step} - {0}, reverse=True)
        regions = tuple(binding.region(0, extent * width, shape=(extent, width)) for extent in extents)
        imports = tuple(plan_import(region) for region in regions)
        peak = max(item.host_peak_bytes + item.execution_peak_bytes for item in imports) + 4
        if peak <= budget:
            break
        if step == row_alignment:
            raise ValueError("one aligned source region exceeds available physical capacity")
        step = max(row_alignment, (step // 2 // row_alignment) * row_alignment)
    templates = []
    for extent, region in zip(extents, regions, strict=True):
        emitter = ProjectionTileEmitter(hidden, region.spec, output, strategy, threads,
                                        tile, arithmetic, len(node.inputs) == 3, outputs_per_subgroup)
        ports = (KernelPort(hidden, PortRole.READ), KernelPort(region.spec, PortRole.READ),
                 KernelPort(TensorSpec((1,), DType.I32), PortRole.READ), KernelPort(output, PortRole.WRITE))
        if len(node.inputs) == 3:
            ports += (KernelPort(graph.value(node.inputs[2]).spec, PortRole.READ),)
        templates.append(SourceTemplate(extent, define_kernel(emitter, ports)))
    loop = ProjectionLoop(node.inputs[1], node.inputs[0], node.outputs[0],
                          node.inputs[2] if len(node.inputs) == 3 else None,
                          columns, step, width, binding, tuple(templates), peak)
    return BoundOperation(f"linear.source-{strategy}@{root}", frozenset({root}),
                          node.inputs, node.outputs, emitter, source_loop=loop,
                          kernel_count=math.ceil(columns / step), dependencies=code_dependencies(projection_loop))
