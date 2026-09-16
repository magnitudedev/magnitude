"""One component owns prepared-input attention and its persistent state sink."""
import tilelang.language as T

from ..compiler.lowering import BoundOperation
from ..compiler.schedules import schedule_boundary
from ..kv import KVRepresentation
from .attention import attention_projection_tile
from .attention_fusion import _PrepareAppendEmitter
from .kv_packed import validate_vector_codec
from .matrix import _packed_matrix, _packed_vector, _packed_vector_geometry
from .normalization import _reduction_threads
from .packed import packet_format
from .persistent_attention import PersistentAttentionRule


class _PersistentComponentEmitter:
    def __init__(self, preparation, attention, value_shape, gated):
        self.preparation, self.attention = preparation, attention
        self.value_shape, self.gated = value_shape, gated

    def __call__(self, operands):
        qg, raw_key, raw_value, qnorm, knorm, coordinates, history, destinations, visible = operands[:9]
        if self.gated:
            attended, next_history, query, key, gate, *scratch = operands[9:]
        else:
            attended, gate, next_history, query, key, *scratch = operands[9:]
        values = T.view(raw_value, shape=self.value_shape, dtype=raw_value.dtype)
        # Reserved destinations are disjoint from committed read visibility.
        # The sink shares the preparation's registers; completion covers both
        # the appended state and attention's dense-current/packed-history result.
        self.preparation((qg, raw_key, raw_value, qnorm, knorm, coordinates,
                          history, destinations, query, gate, next_history, key))
        trailing = (gate, attended, *scratch) if self.gated else (attended, *scratch)
        self.attention((query, history, key, values, visible, *trailing))


class PersistentAttentionComponentRule:
    def __init__(self, *, gated=False):
        self.gated = gated

    def build(self, graph, root, context):
        if root + 3 >= len(graph.nodes):
            return ()
        prepare, reshape, attention, append = graph.nodes[root:root + 4]
        if (prepare.operation != 'attention_prepare' or reshape.operation != 'reshape'
                or attention.operation != 'persistent_attention' or append.operation != 'kv_append'
                or not append.attributes.get('reserved', False)):
            return ()
        query, key, gate = prepare.outputs
        history, appended_key, appended_value, destinations = append.inputs
        values = reshape.outputs[0]
        if ((appended_key, appended_value) != (key, values)
                or attention.inputs[:4] != (query, history, key, values)):
            return ()
        history_spec = graph.value(history).spec
        if not isinstance(history_spec.representation, KVRepresentation):
            return ()
        validate_vector_codec(history_spec, context.compiler_target.subgroup_width)
        threads = _reduction_threads(prepare.attributes['width'], context)
        if threads is None:
            return ()
        nodes = frozenset((prepare.id, reshape.id, attention.id, append.id))
        outputs = (attention.outputs[0], gate, append.outputs[0])
        if self.gated:
            if root + 6 >= len(graph.nodes):
                return ()
            sigmoid, multiply, flatten = graph.nodes[root + 4:root + 7]
            if (sigmoid.operation != 'sigmoid' or sigmoid.inputs != (gate,)
                    or multiply.operation != 'multiply'
                    or set(multiply.inputs) != {attention.outputs[0], sigmoid.outputs[0]}
                    or flatten.operation != 'reshape' or flatten.inputs != multiply.outputs):
                return ()
            nodes |= frozenset((sigmoid.id, multiply.id, flatten.id))
            outputs = (flatten.outputs[0], append.outputs[0])
        boundary_specs = tuple(graph.value(value).spec for node in graph.nodes[root:root + (7 if self.gated else 4)]
                               for value in (*node.inputs, *node.outputs))
        context = schedule_boundary(context, _PersistentComponentEmitter,
                                    (boundary_specs, prepare.attributes, self.gated))
        persistent = PersistentAttentionRule().build(graph, attention.id, context)[0]
        emitter = persistent.emitter
        workspace = (graph.value(query).spec, graph.value(key).spec,
                     *((graph.value(gate).spec,) if self.gated else ()), *persistent.workspace)
        if self.gated:
            emitter = type(emitter)(emitter.specs, emitter.scale, emitter.history_schedule,
                                    emitter.current_schedule, emitter.matrix, emitter.threads,
                                    emitter.subgroup_width, fuse_gate=True)
        if sum(spec.storage_nbytes for spec in workspace) > context.workspace_limit:
            return ()
        qg, raw_key, qnorm, knorm, coordinates = prepare.inputs
        raw_value = reshape.inputs[0]
        inputs = (qg, raw_key, raw_value, qnorm, knorm, coordinates,
                  history, destinations, attention.inputs[4])
        preparation = _PrepareAppendEmitter(
            prepare.attributes, graph.value(query).spec.shape[0], context.compiler_target.subgroup_width,
            min(threads // context.compiler_target.subgroup_width,
                prepare.attributes['query_heads'] + prepare.attributes['kv_heads']),
            graph.value(query).spec.dtype.value, history_spec)
        return (BoundOperation(
            f'attention.persistent-component{"-gated" if self.gated else ""}@{root}', nodes, inputs, outputs,
            _PersistentComponentEmitter(preparation, emitter, graph.value(values).spec.shape, self.gated),
            workspace=workspace, aliases=((append.outputs[0], history),),
            kernel_count=persistent.kernel_count + 1,
            workspace_values={query: 0, key: 1, **({gate: 2} if self.gated else {})}),)


class _PersistentMixerEmitter:
    def __init__(self, component, activation, weight, output, tile, vector):
        self.component, self.activation, self.weight, self.output = component, activation, weight, output
        self.tile, self.vector = tile, vector

    def __call__(self, operands):
        weight, output, next_history, activation, *scratch = operands[9:]
        self.component((*operands[:9], activation, next_history, *scratch))
        rows, width = self.activation.shape
        columns = self.weight.shape[0]
        if self.tile is not None:
            threads, bm, bn, bk, contraction_schedule = self.tile
            _packed_matrix(activation, weight, activation, output, self.weight,
                           rows, columns, width, contraction_schedule, self.output.dtype.value,
                           threads, bm, bn, bk, False)
        else:
            threads, outputs_per_subgroup = self.vector
            _packed_vector(activation, weight, activation, output, self.weight,
                           rows, columns, width, self.output.dtype.value,
                           False, threads, outputs_per_subgroup)


class PersistentAttentionMixerRule:
    """Keep state, attention, gate and output contraction in the same owner."""

    def build(self, graph, root, context):
        if root + 7 >= len(graph.nodes):
            return ()
        projection = graph.nodes[root + 7]
        if projection.operation != 'linear' or len(projection.inputs) != 2:
            return ()
        context = schedule_boundary(context, _PersistentMixerEmitter,
                                    tuple(graph.value(value).spec for value in (*projection.inputs, *projection.outputs)))
        built = PersistentAttentionComponentRule(gated=True).build(graph, root, context)
        if not built:
            return ()
        component = built[0]
        following = max(component.nodes) + 1
        if following >= len(graph.nodes):
            return ()
        projection = graph.nodes[following]
        attended, next_history = component.outputs
        if (projection.operation != 'linear' or len(projection.inputs) != 2
                or projection.inputs[0] != attended):
            return ()
        activation = graph.value(attended).spec
        weight = graph.value(projection.inputs[1]).spec
        output = graph.value(projection.outputs[0]).spec
        packet = packet_format(weight)
        if (packet is None or not activation.static or not weight.static
                or activation.shape[1] % packet.tile):
            return ()
        tile = vector = None
        if context.mode == 'prefill':
            tile = attention_projection_tile(context, activation, weight,
                                             template=_PersistentMixerEmitter,
                                             workload=(component.emitter, activation, weight, output))
            if tile is None:
                return ()
        else:
            vector = _packed_vector_geometry(weight, context)
            if vector is None:
                return ()
        workspace = (activation, *component.workspace)
        if sum(spec.storage_nbytes for spec in workspace) > context.workspace_limit:
            return ()
        return (BoundOperation(
            f'attention.persistent-mixer@{root}', component.nodes | frozenset((projection.id,)),
            (*component.inputs, projection.inputs[1]), (projection.outputs[0], next_history),
            _PersistentMixerEmitter(component.emitter, activation, weight, output, tile, vector),
            workspace=workspace, aliases=component.aliases, kernel_count=component.kernel_count + 1,
            workspace_values={attended: 0, **{value: index + 1
                for value, index in component.workspace_values.items()}}),)
