"""Schedule identity and pre-composition selection preserve qualification boundaries."""

from dataclasses import replace

import pytest

from ops.compiler.lowering import CompilerTarget, LoweringContext
from ops.compiler.schedules import select_schedule
from ops.kernels.schedules import AffineSchedule, OperandPreparation


class Capture:
    def select(self, request, default):
        self.request = request
        return request.candidates[-1]


def template_one():
    return 1


def template_two():
    return 2


def test_every_construction_boundary_invalidates_the_request():
    capture = Capture()
    context = LoweringContext(
        CompilerTarget(32, 128, 32768),
        "prefill",
        "model",
        "compiler",
        1 << 20,
        schedules=capture,
        device_identity="physical-and-runtime",
    )
    family = (
        AffineSchedule(OperandPreparation.WHOLE),
        AffineSchedule(OperandPreparation.SLICED, 8),
    )

    def choose(ctx=context, candidates=family, template=template_one, workload=(9, 257, 512)):
        selected = select_schedule(
            ctx, "projection", candidates, candidates[0], template=template, workload=workload
        )
        assert selected == candidates[-1]
        return capture.request

    original = choose()
    assert choose() == original
    assert choose(workload=(10, 257, 512)) != original
    assert choose(candidates=tuple(reversed(family))) != original
    assert choose(template=template_two) != original
    assert choose(ctx=replace(context, precision="different")) != original
    assert choose(ctx=replace(context, schedule_scope=(("fused", (9, 256)),))) != original
    assert choose(ctx=replace(context, device_identity="changed-driver")) != original
    assert choose(ctx=replace(context, compiler_identity="changed-compiler")) != original
    assert (
        choose(
            ctx=replace(
                context, compiler_target=replace(context.compiler_target, shared_memory_bytes=16384)
            )
        )
        != original
    )


def test_resolver_cannot_introduce_an_undeclared_schedule():
    class Invalid:
        def select(self, request, default):
            return AffineSchedule(OperandPreparation.SHARED)

    context = LoweringContext(
        CompilerTarget(32, 128, 32768), "prefill", "model", "compiler", 1 << 20, schedules=Invalid()
    )
    default = AffineSchedule(OperandPreparation.WHOLE)
    with pytest.raises(ValueError, match="outside"):
        select_schedule(
            context, "projection", (default,), default, template=template_one, workload=(1, 2, 3)
        )


def test_fused_attention_cannot_reuse_standalone_qualification():
    import ops

    class Requests:
        def __init__(self):
            self.values = []

        def select(self, request, default):
            self.values.append(request)
            return default

    @ops.formula
    def standalone(q, h, k, v, r, gate):
        return ops.persistent_attention(q, h, k, v, r, sequence_count=1)

    @ops.formula
    def gated(q, h, k, v, r, gate):
        attended = ops.persistent_attention(q, h, k, v, r, sequence_count=1)
        return ops.reshape(attended * ops.sigmoid(gate), (33, 4096))

    ops.operation(gated)(ops.operation_bodies.attention_mixer)

    query = ops.TensorSpec((33, 16, 256), ops.DType.BF16)
    current = ops.TensorSpec((33, 2, 256), ops.DType.BF16)
    history = ops.kv_state_spec(97, 2, ops.DType.BF16, ops.affine_k8_uniform_v4(256, 256))
    signature = ops.Signature(
        tuple(
            ops.Argument(spec, name, ops.ValueKind.RESOURCE if name == "h" else ops.ValueKind.INPUT)
            for spec, name in zip(
                (query, history, current, current, ops.TensorSpec((33, 4), ops.DType.I32), query),
                ("q", "h", "k", "v", "r", "gate"),
                strict=True,
            )
        )
    )
    requests = []
    for formula in (standalone, gated):
        capture = Requests()
        plan = ops.analyze(
            formula,
            signature=signature,
            compiler_target=CompilerTarget(32, 256, 32768),
            compiler_identity="test",
            available_bytes=1 << 28,
            options=ops.CompileOptions(mode="prefill", schedules=capture),
        )
        requests.append(
            next(r for r in capture.values if r.semantic_kernel.startswith("attention.persistent"))
        )
        assert any("persistent-gated" in op.name for op in plan.operations) == (formula is gated)
    assert requests[0].workload != requests[1].workload


def test_composed_selection_qualifies_joint_choices_once():
    from ops.compiler.schedules import CollectSchedules, select_composed_schedule

    calls = []

    class LastBundle:
        def select(self, request, default):
            calls.append(request)
            return request.candidates[-1]

    context = LoweringContext(
        CompilerTarget(32, 256, 32768),
        "prefill",
        "model",
        "compiler",
        1 << 20,
        schedules=LastBundle(),
        device_identity="device",
    )

    def build(ctx):
        first = select_schedule(ctx, "first", (8, 16), 8, template=template_one, workload=(33, 256))
        second = select_schedule(
            ctx, "second", (32, 64), 32, template=template_two, workload=(first, 256)
        )
        return first, second

    collected = CollectSchedules()
    assert build(replace(context, schedules=collected)) == (8, 32)
    selected = select_composed_schedule(
        context, collected, name="composed", template=template_one, workload=(33, 256), build=build
    )
    assert selected == (16, 64)
    assert len(calls) == 1
    assert len(calls[0].candidates) == 4
    assert {
        tuple(choice.value for choice in candidate.choices) for candidate in calls[0].candidates
    } == {(8, 32), (8, 64), (16, 32), (16, 64)}
