
from tests.ops.target_fixture import matrix_query
import pytest

import ops
from ops.operation import build_operations
from ops.compiler.lowering import LoweringContext, plan_submissions
from ops.compiler.memory import plan_memory
from ops.compiler.unit import build_unit
from ops.runtime.tilelang import _build_reusable_module

CAPABILITIES = ops.CompilerTarget(
    32,
    256,
    32 * 1024,
    matrix_query=matrix_query((ops.MatrixTile(8, 8, 8, ops.DType.F16, ops.DType.F32),)),


    identity="construction-test",
)


def _construct(function, signature):
    graph = ops.trace(function, signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 26)
    cover = build_operations(graph, context)
    memory = plan_memory(graph, cover)
    submissions = plan_submissions(graph, cover)
    return tuple(_build_reusable_module(build_unit(graph, memory, unit)) for unit in submissions)


def test_dense_and_encoded_projection_construct_real_prim_funcs():
    hidden = ops.TensorSpec((2, 512), ops.DType.F16)
    dense = ops.TensorSpec((8, 512), ops.DType.F16)
    encoded = dense.with_representation(
        ops.Affine(
            ops.Code(4),
            64,
            ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16),
        )
    )
    dense_functions = _construct(
        lambda value, weight: ops.linear(value, weight),
        ops.Signature(
            (
                ops.Argument(hidden, "value"),
                ops.Argument(dense, "weight", ops.ValueKind.CONSTANT),
            )
        ),
    )
    encoded_functions = _construct(
        lambda value, weight: ops.linear(value, weight),
        ops.Signature(
            (
                ops.Argument(hidden, "value"),
                ops.Argument(encoded, "weight", ops.ValueKind.CONSTANT),
            )
        ),
    )
    assert len(dense_functions) == len(encoded_functions) == 1


def test_attention_and_recurrence_construct_real_prim_funcs():
    queries = ops.TensorSpec((2, 4, 256), ops.DType.F16)
    history = ops.TensorSpec((2, 1024, 2, 256), ops.DType.F16)
    visible = ops.TensorSpec((2, 2), ops.DType.I32)
    attention = _construct(
        lambda q, h, lengths: ops.causal_attention(q, h, lengths),
        ops.Signature(
            (
                ops.Argument(queries, "query"),
                ops.Argument(history, "history", ops.ValueKind.RESOURCE),
                ops.Argument(visible, "visible"),
            )
        ),
    )
    query = ops.TensorSpec((2, 1, 4), ops.DType.F32)
    value = ops.TensorSpec((2, 2, 3), ops.DType.F32)
    parameter = ops.TensorSpec((2, 2), ops.DType.F32)
    state = ops.TensorSpec((1, 2, 3, 4), ops.DType.F32)
    offsets = ops.TensorSpec((2,), ops.DType.I32)
    recurrence = _construct(
        lambda q, k, v, decay, beta, recurrent, rows: ops.gated_delta_recurrence(
            q, k, v, decay, beta, recurrent, rows, mapping="tiled"
        )[0],
        ops.Signature(
            (
                ops.Argument(query, "queries"),
                ops.Argument(query, "keys"),
                ops.Argument(value, "values"),
                ops.Argument(parameter, "decay"),
                ops.Argument(parameter, "beta"),
                ops.Argument(state, "state", ops.ValueKind.RESOURCE),
                ops.Argument(offsets, "offsets"),
            )
        ),
    )
    assert len(attention) == len(recurrence) == 1


def test_quantized_residency_import_constructs_real_prim_func():
    from engine.weights.formats.gguf import Encoding, quantization

    _, codec = quantization(Encoding.Q8_0)
    source = ops.TensorSpec((codec.block_bytes,), ops.DType.U8)
    target = ops.TensorSpec((32,), ops.DType.F16).with_representation(
        ops.Affine(
            ops.Code(8, interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),
            32,
            ops.DirectCoefficients(ops.DType.F16),
        )
    )
    extent = ops.TensorSpec((2,), ops.DType.I32)
    functions = _construct(
        lambda raw, limits, *, resident: ops.quantized_import(
            raw,
            resident,
            limits,
            codec=codec,
            staged_tiles=1,
        ),
        ops.Signature(
            (ops.Argument(source, "source"), ops.Argument(extent, "extent")),
            {"resident": ops.Argument(target, "resident", ops.ValueKind.RESOURCE)},
        ),
    )
    assert len(functions) == 1


def test_sampling_constructs_inside_a_tensor_program():
    functions = _construct(
        lambda logits, draws: ops.sample(logits, draws),
        ops.Signature(
            (
                ops.Argument(ops.TensorSpec((2, 32), ops.DType.F32), "logits"),
                ops.Argument(ops.TensorSpec((2, 6), ops.DType.U32), "draws"),
            )
        ),
    )
    assert len(functions) == 1


def test_routing_and_experts_compose_into_one_prim_func():
    hidden = ops.TensorSpec((2, 256), ops.DType.F16)
    router = ops.TensorSpec((2, 4), ops.DType.F32)
    representation = ops.Affine(
        ops.Code(8, interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),
        32,
        ops.DirectCoefficients(ops.DType.F16),
    )
    expert = ops.TensorSpec((4, 256, 256), ops.DType.F16).with_representation(representation)
    down = ops.TensorSpec((4, 256, 256), ops.DType.F16).with_representation(representation)

    def mixture(value, logits, gate, up, down_weight):
        routes, scores = ops.route_topk(logits, 2)
        return ops.routed_experts(value, routes, scores, gate, up, down_weight)

    functions = _construct(
        mixture,
        ops.Signature(
            (
                ops.Argument(hidden, "hidden"),
                ops.Argument(router, "router"),
                ops.Argument(expert, "gate", ops.ValueKind.CONSTANT),
                ops.Argument(expert, "up", ops.ValueKind.CONSTANT),
                ops.Argument(down, "down", ops.ValueKind.CONSTANT),
            )
        ),
    )
    assert len(functions) == 1


def test_fused_pointwise_region_constructs_real_prim_func():
    values = ops.TensorSpec((2, 8), ops.DType.F16)
    functions = _construct(
        lambda x: ops.tanh(ops.silu(x) + x),
        ops.Signature((ops.Argument(values, "values"),)),
    )
    assert len(functions) == 1


def test_concatenation_uses_one_kernel_for_many_inputs():
    spec = ops.TensorSpec((1, 4), ops.DType.F16)
    signature = ops.Signature(tuple(ops.Argument(spec, f"x{i}") for i in range(8)))
    graph = ops.trace(lambda *values: ops.concatenate(values, axis=0), signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = build_operations(graph, context)
    submissions = plan_submissions(graph, cover)
    assert len(submissions) == 1 and submissions[0].kernel_count == 1
    memory = plan_memory(graph, cover)
    assert _build_reusable_module(build_unit(graph, memory, submissions[0])).functions


def test_shape_embedding_rotary_and_state_schedules_construct():
    matrix = ops.TensorSpec((2, 4), ops.DType.F16)
    _construct(
        lambda left, right: ops.concatenate((left, right), axis=0),
        ops.Signature(
            (
                ops.Argument(matrix, "left"),
                ops.Argument(matrix, "right"),
            )
        ),
    )

    indices = ops.TensorSpec((3,), ops.DType.I32)
    table = ops.TensorSpec((16, 8), ops.DType.F16)
    assert (
        len(
            _construct(
                lambda token, weight: ops.embedding(token, weight),
                ops.Signature(
                    (
                        ops.Argument(indices, "indices"),
                        ops.Argument(table, "table", ops.ValueKind.CONSTANT),
                    )
                ),
            )
        )
        == 1
    )

    heads = ops.TensorSpec((2, 4, 8), ops.DType.F16)
    positions = ops.TensorSpec((2,), ops.DType.I32)
    assert (
        len(
            _construct(
                lambda q, k, p: ops.rotary(q, k, p),
                ops.Signature(
                    (
                        ops.Argument(heads, "queries"),
                        ops.Argument(heads, "keys"),
                        ops.Argument(positions, "positions"),
                    )
                ),
            )
        )
        == 1
    )

    history = ops.TensorSpec((2, 16, 4, 8), ops.DType.F16)
    appended = ops.TensorSpec((2, 4, 8), ops.DType.F16)
    graph = ops.trace(
        lambda cache, keys, values, write: ops.kv_append(cache, keys, values, write),
        ops.Signature(
            (
                ops.Argument(history, "history", ops.ValueKind.RESOURCE),
                ops.Argument(appended, "keys"),
                ops.Argument(appended, "values"),
                ops.Argument(positions, "destinations"),
            )
        ),
    )
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 26)
    defined = build_operations(graph, context)
    assert len(defined) == 1
    assert defined[0].aliases == ((graph.outputs[0], graph.resources[0]),)
