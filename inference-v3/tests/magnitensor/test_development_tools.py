import pytest

import magnitensor as mt
from magnitude_engine.kernel_bench import definition

CAPABILITIES = mt.Capabilities(
    32,
    256,
    32 * 1024,
    matrix_instructions=(mt.MatrixInstruction(8, 8, 8, mt.DType.F16, mt.DType.F32),),
    memory_scopes=frozenset({"global", "shared", "local"}),
    atomics=frozenset({mt.DType.I32}),
    native_multi_launch=True,
    partial_binding=True,
    fingerprint="development-tool-test",
)


def _encoded(shape):
    return mt.TensorSpec(shape, mt.DType.F16).with_representation(
        mt.Affine(
            mt.Code(4),
            64,
            mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16),
        )
    )


@pytest.mark.parametrize(
    "name,mode,rows,expected",
    (
        ("encoded-linear", "decode", 1, "linear.packet-vector"),
        ("parallel-linear", "decode", 1, "linear.parallel-packet"),
        ("dense-swiglu", "prefill", 8, "dense_swiglu.packet-prefill"),
        ("attention", "decode", 1, "causal_attention.online"),
        ("recurrent-prepare", "prefill", 8, "recurrent_prepare.channel-parallel"),
        ("gated-recurrence", "prefill", 8, "gated_delta.register-state"),
        ("grouped-experts", "prefill", 8, "routed_experts.grouped"),
    ),
)
def test_kernel_benchmark_case_has_a_valid_compile_free_plan(name, mode, rows, expected):
    case = definition(name, rows, 64)
    plan = mt.analyze(
        case.function,
        signature=mt.Signature(tuple(operand.argument for operand in case.operands)),
        capabilities=CAPABILITIES,
        options=mt.CompileOptions(mode=mode),
        available_bytes=1 << 30,
    )

    selected = tuple(candidate.name for candidate in plan.cover.candidates)
    assert any(candidate.startswith(expected + "@") for candidate in selected)


def test_small_parallel_projections_form_one_lowering_region():
    source = mt.TensorSpec((1, 512), mt.DType.F16)
    weight = _encoded((32, 512))

    def project(value, first, second, third):
        return (
            mt.linear(value, first),
            mt.linear(value, second),
            mt.linear(value, third),
        )

    plan = mt.analyze(
        project,
        signature=mt.Signature(
            (
                mt.Argument(source, "source"),
                mt.Argument(weight, "first", mt.ValueKind.CONSTANT),
                mt.Argument(weight, "second", mt.ValueKind.CONSTANT),
                mt.Argument(weight, "third", mt.ValueKind.CONSTANT),
            )
        ),
        capabilities=CAPABILITIES,
        options=mt.CompileOptions(mode="decode"),
        available_bytes=1 << 30,
    )

    assert [candidate.name.split("@", 1)[0] for candidate in plan.cover.candidates] == [
        "linear.parallel-packet"
    ]
    assert plan.diagnostics.dispatches == 1
    assert len(plan.submissions) == 1
