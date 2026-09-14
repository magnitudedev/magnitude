import pytest

import magnitensor as mt
from magnitensor.compiler.lowering import LoweringContext, plan_submissions, select_cover
from magnitensor.compiler.memory import plan_memory
from magnitensor.compiler.unit import build_unit
from magnitensor.runtime.tilelang import _build_prim_func

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    matrix_instructions=(mt.MatrixInstruction(8, 8, 8, mt.DType.F16, mt.DType.F32),),
    memory_scopes=frozenset({"global", "shared", "local"}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="construction-test",
)


def _construct(function, signature):
    graph = mt.trace(function, signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 26)
    candidates = mt.lowerings.enumerate(graph, context)
    cover = select_cover(graph, candidates)
    memory = plan_memory(graph, cover, CAPABILITIES)
    submissions = plan_submissions(graph, cover, CAPABILITIES)
    return tuple(_build_prim_func(build_unit(graph, memory, unit)) for unit in submissions)


def test_dense_and_encoded_projection_construct_real_prim_funcs():
    hidden = mt.TensorSpec((2, 512), mt.DType.F16)
    dense = mt.TensorSpec((8, 512), mt.DType.F16)
    encoded = dense.with_representation(
        mt.Affine(
            mt.Code(4),
            64,
            mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16),
        )
    )
    dense_functions = _construct(
        lambda value, weight: mt.linear(value, weight),
        mt.Signature(
            (
                mt.Argument(hidden, "value"),
                mt.Argument(dense, "weight", mt.ValueKind.CONSTANT),
            )
        ),
    )
    encoded_functions = _construct(
        lambda value, weight: mt.linear(value, weight),
        mt.Signature(
            (
                mt.Argument(hidden, "value"),
                mt.Argument(encoded, "weight", mt.ValueKind.CONSTANT),
            )
        ),
    )
    assert len(dense_functions) == len(encoded_functions) == 1


def test_attention_and_recurrence_construct_real_prim_funcs():
    queries = mt.TensorSpec((2, 4, 256), mt.DType.F16)
    history = mt.TensorSpec((2, 1024, 2, 256), mt.DType.F16)
    visible = mt.TensorSpec((2, 2), mt.DType.I32)
    attention = _construct(
        lambda q, h, lengths: mt.causal_attention(q, h, lengths),
        mt.Signature(
            (
                mt.Argument(queries, "query"),
                mt.Argument(history, "history", mt.ValueKind.RESOURCE),
                mt.Argument(visible, "visible"),
            )
        ),
    )
    query = mt.TensorSpec((2, 1, 4), mt.DType.F32)
    value = mt.TensorSpec((2, 2, 3), mt.DType.F32)
    parameter = mt.TensorSpec((2, 2), mt.DType.F32)
    state = mt.TensorSpec((1, 2, 3, 4), mt.DType.F32)
    offsets = mt.TensorSpec((2,), mt.DType.I32)
    recurrence = _construct(
        lambda q, k, v, decay, beta, recurrent, rows: mt.gated_delta_recurrence(
            q, k, v, decay, beta, recurrent, rows, mapping="tiled"
        )[0],
        mt.Signature(
            (
                mt.Argument(query, "queries"),
                mt.Argument(query, "keys"),
                mt.Argument(value, "values"),
                mt.Argument(parameter, "decay"),
                mt.Argument(parameter, "beta"),
                mt.Argument(state, "state", mt.ValueKind.RESOURCE),
                mt.Argument(offsets, "offsets"),
            )
        ),
    )
    assert len(attention) == len(recurrence) == 1


def test_quantized_residency_import_constructs_real_prim_func():
    from magnitude_engine.weights.formats.gguf import Encoding, quantization

    _, codec = quantization(Encoding.Q8_0)
    source = mt.TensorSpec((codec.block_bytes,), mt.DType.U8)
    target = mt.TensorSpec((32,), mt.DType.F16).with_representation(
        mt.Affine(
            mt.Code(8, interpretation=mt.CodeInterpretation.TWOS_COMPLEMENT),
            32,
            mt.DirectCoefficients(mt.DType.F16),
        )
    )
    extent = mt.TensorSpec((2,), mt.DType.I32)
    functions = _construct(
        lambda raw, limits, *, resident: mt.quantized_import(
            raw,
            resident,
            limits,
            codec=codec,
            staged_tiles=1,
        ),
        mt.Signature(
            (mt.Argument(source, "source"), mt.Argument(extent, "extent")),
            {"resident": mt.Argument(target, "resident", mt.ValueKind.RESOURCE)},
        ),
    )
    assert len(functions) == 1


def test_sampling_constructs_inside_a_tensor_program():
    functions = _construct(
        lambda logits, draws: mt.sample(logits, draws),
        mt.Signature(
            (
                mt.Argument(mt.TensorSpec((2, 32), mt.DType.F32), "logits"),
                mt.Argument(mt.TensorSpec((2, 6), mt.DType.U32), "draws"),
            )
        ),
    )
    assert len(functions) == 1


def test_routing_and_experts_compose_into_one_prim_func():
    hidden = mt.TensorSpec((2, 256), mt.DType.F16)
    router = mt.TensorSpec((2, 4), mt.DType.F32)
    representation = mt.Affine(
        mt.Code(8, interpretation=mt.CodeInterpretation.TWOS_COMPLEMENT),
        32,
        mt.DirectCoefficients(mt.DType.F16),
    )
    expert = mt.TensorSpec((4, 256, 256), mt.DType.F16).with_representation(representation)
    down = mt.TensorSpec((4, 256, 256), mt.DType.F16).with_representation(representation)

    def mixture(value, logits, gate, up, down_weight):
        routes, scores = mt.route_topk(logits, 2)
        return mt.routed_experts(value, routes, scores, gate, up, down_weight)

    functions = _construct(
        mixture,
        mt.Signature(
            (
                mt.Argument(hidden, "hidden"),
                mt.Argument(router, "router"),
                mt.Argument(expert, "gate", mt.ValueKind.CONSTANT),
                mt.Argument(expert, "up", mt.ValueKind.CONSTANT),
                mt.Argument(down, "down", mt.ValueKind.CONSTANT),
            )
        ),
    )
    assert len(functions) == 1


def test_fused_pointwise_region_constructs_real_prim_func():
    values = mt.TensorSpec((2, 8), mt.DType.F16)
    functions = _construct(
        lambda x: mt.tanh(mt.silu(x) + x),
        mt.Signature((mt.Argument(values, "values"),)),
    )
    assert len(functions) == 1


def test_concatenation_uses_one_kernel_for_many_inputs():
    spec = mt.TensorSpec((1, 4), mt.DType.F16)
    signature = mt.Signature(tuple(mt.Argument(spec, f"x{i}") for i in range(8)))
    graph = mt.trace(lambda *values: mt.concatenate(values, axis=0), signature)
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 20)
    cover = select_cover(graph, mt.lowerings.enumerate(graph, context))
    submissions = plan_submissions(graph, cover, CAPABILITIES)
    assert len(submissions) == 1 and submissions[0].kernel_count == 1
    memory = plan_memory(graph, cover, CAPABILITIES)
    assert _build_prim_func(build_unit(graph, memory, submissions[0])).attrs is not None


def test_shape_embedding_rotary_and_state_schedules_construct():
    matrix = mt.TensorSpec((2, 4), mt.DType.F16)
    _construct(
        lambda left, right: mt.concatenate((left, right), axis=0),
        mt.Signature(
            (
                mt.Argument(matrix, "left"),
                mt.Argument(matrix, "right"),
            )
        ),
    )

    indices = mt.TensorSpec((3,), mt.DType.I32)
    table = mt.TensorSpec((16, 8), mt.DType.F16)
    assert (
        len(
            _construct(
                lambda token, weight: mt.embedding(token, weight),
                mt.Signature(
                    (
                        mt.Argument(indices, "indices"),
                        mt.Argument(table, "table", mt.ValueKind.CONSTANT),
                    )
                ),
            )
        )
        == 1
    )

    heads = mt.TensorSpec((2, 4, 8), mt.DType.F16)
    positions = mt.TensorSpec((2,), mt.DType.I32)
    assert (
        len(
            _construct(
                lambda q, k, p: mt.rotary(q, k, p),
                mt.Signature(
                    (
                        mt.Argument(heads, "queries"),
                        mt.Argument(heads, "keys"),
                        mt.Argument(positions, "positions"),
                    )
                ),
            )
        )
        == 1
    )

    history = mt.TensorSpec((2, 16, 4, 8), mt.DType.F16)
    appended = mt.TensorSpec((2, 4, 8), mt.DType.F16)
    graph = mt.trace(
        lambda cache, keys, values, write: mt.kv_append(cache, keys, values, write),
        mt.Signature(
            (
                mt.Argument(history, "history", mt.ValueKind.RESOURCE),
                mt.Argument(appended, "keys"),
                mt.Argument(appended, "values"),
                mt.Argument(positions, "destinations"),
            )
        ),
    )
    context = LoweringContext(CAPABILITIES, "decode", "model", "test", 1 << 26)
    with pytest.raises(ValueError, match="no legal lowering"):
        select_cover(graph, mt.lowerings.enumerate(graph, context))
