"""Each table picks the kernel the engine's former if-chains picked.

These are the inputs the engine actually sees: decode at one row, prefill at 8,
64, 256 and 4096 rows; K-quant and MLX weights; each precision preset;
a 32-lane subgroup endpoint and the host. Nothing is compiled — a candidate's
choice is a fact about the four axes, so it can be read without a device.
"""

import pytest

from magnitude_engine.kernels.attention.select import (
    TABLE as ATTENTION,
)
from magnitude_engine.kernels.attention.select import (
    AttentionShape,
    HistoryShape,
)
from magnitude_engine.kernels.capabilities import HOST, Capability
from magnitude_engine.kernels.embedding.select import TABLE as EMBEDDING
from magnitude_engine.kernels.embedding.select import EmbeddingShape
from magnitude_engine.kernels.norm.select import TABLE as NORM
from magnitude_engine.kernels.norm.select import NormShape
from magnitude_engine.kernels.precision import (
    MIXED_BF16,
    MIXED_BF16_F32_RESIDUAL,
    NATIVE_BF16,
    PRESETS,
    REFERENCE_F32,
)
from magnitude_engine.kernels.projection.select import (
    GATED_TABLE,
    GatedShape,
    ProjectionShape,
)
from magnitude_engine.kernels.projection.select import (
    TABLE as PROJECTION,
)
from magnitude_engine.kernels.recurrence.select import TABLE as RECURRENCE
from magnitude_engine.kernels.recurrence.select import RecurrenceShape
from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.operations.candidates import NoCandidate, Selection, select
from magnitude_engine.platform.execution import DType
from magnitude_engine.weights.representation import (
    Affine,
    Code,
    Codebook,
    CodeInterpretation,
    Dense,
    DirectCoefficients,
    HierarchicalCoefficients,
    WeightLayout,
)

METAL = Capability(
    subgroup_width=32, threads_per_group=1024, shared_memory_bytes=32768, matrix_instructions=True
)
UNBOUNDED = 1 << 60

SCALE_MIN = HierarchicalCoefficients(
    256,
    6,
    CodeInterpretation.UNSIGNED,
    DType.F16,
    local_bias_bits=6,
    super_bias_dtype=DType.F16,
    bias_sign=-1,
)
Q4_K = Affine(Code(4), 32, SCALE_MIN)
Q6_K = Affine(
    Code(4, 2, CodeInterpretation.OFFSET_BINARY, 32),
    16,
    HierarchicalCoefficients(256, 8, CodeInterpretation.TWOS_COMPLEMENT, DType.F16),
)
IQ4_XS = Codebook(
    4,
    tuple(range(16)),
    32,
    HierarchicalCoefficients(
        256,
        6,
        CodeInterpretation.OFFSET_BINARY,
        DType.F16,
        local_scale_zero_point=32,
    ),
)
F32 = Dense(DType.F32)
MLX_Q4 = Affine(Code(4), 64, DirectCoefficients(DType.BF16, DType.BF16))


def chosen(name, table, shape, precision, capability, layout=None, budget=UNBOUNDED):
    candidate, _ = select(
        name,
        table,
        Selection(shape, precision, capability, layout),
        available_bytes=budget,
    )
    return candidate.name


def projection(rows, representation, inputs=2048, widths=(2048,), capability=METAL, **kwargs):
    layout = WeightLayout(representation, sum(widths), inputs)
    return chosen(
        "projection",
        PROJECTION,
        ProjectionShape(rows, widths, inputs, DType.BF16, DType.BF16),
        kwargs.pop("precision", NATIVE_BF16),
        capability,
        layout,
        **kwargs,
    )


@pytest.mark.parametrize("precision", tuple(PRESETS.values()))
@pytest.mark.parametrize(
    "rows,representation,expected",
    [
        # Decode reduces one row across a subgroup.
        (1, Q4_K, "projection.hierarchical"),
        (1, Q6_K, "projection.groupwise"),
        (1, IQ4_XS, "projection.groupwise"),
        (1, MLX_Q4, "projection.direct_affine.vector"),
        (1, F32, "projection.subgroup"),
        # From eight rows the contraction fills matrix tiles.
        (8, Q4_K, "projection.matrix"),
        (8, MLX_Q4, "projection.direct_affine.matrix"),
        (64, Q6_K, "projection.matrix"),
        (256, MLX_Q4, "projection.direct_affine.matrix"),
        (4096, Q4_K, "projection.matrix"),
        (4096, MLX_Q4, "projection.direct_affine.matrix"),
    ],
)
def test_projection_selects_the_same_schedule_as_before(rows, representation, expected, precision):
    assert projection(rows, representation, precision=precision) == expected


def test_projection_falls_back_to_the_host_where_there_are_no_groups():
    assert projection(1, Q4_K, capability=HOST) == "projection.serial"
    assert projection(64, Q4_K, capability=HOST) == "projection.serial"
    assert projection(1, MLX_Q4, capability=HOST) == "projection.serial"


def test_projection_uses_the_canonical_direct_schedule_for_narrow_inputs():
    narrow = Capability(
        subgroup_width=32,
        threads_per_group=1024,
        shared_memory_bytes=32768,
        matrix_instructions=False,
    )
    assert projection(1, MLX_Q4, inputs=768, capability=narrow) == (
        "projection.direct_affine.vector"
    )


def test_projection_uses_a_threadgroup_where_lanes_cannot_exchange():
    no_subgroup = Capability(
        subgroup_width=1,
        threads_per_group=256,
        shared_memory_bytes=16384,
        matrix_instructions=False,
    )
    assert projection(1, Q4_K, capability=no_subgroup) == "projection.threadgroup"


def attention(rows, capacity, capability=METAL, dtype=DType.BF16, width=256, budget=UNBOUNDED):
    shape = AttentionShape(rows, 32, 4, width, dtype, (HistoryShape(capacity, capacity, 1),))
    return chosen("attention", ATTENTION, shape, NATIVE_BF16, capability, budget=budget)


@pytest.mark.parametrize(
    "rows,capacity,expected",
    [
        (1, 4096, "attention.decode_partitioned"),
        (1, 8192, "attention.decode_partitioned"),
        (1, 16384, "attention.decode_online"),
        (1, 65536, "attention.decode_online"),
        (8, 16384, "attention.streaming"),
        (64, 65536, "attention.streaming"),
        (256, 16384, "attention.materialized"),
        (4096, 16384, "attention.materialized"),
    ],
)
def test_attention_selects_the_same_schedule_as_before(rows, capacity, expected):
    assert attention(rows, capacity) == expected


def test_a_plan_that_cannot_fit_its_scratch_is_skipped_before_any_allocation():
    """Materialized scores are the largest declaration; streaming still fits."""
    assert attention(256, 16384) == "attention.materialized"
    assert attention(256, 16384, budget=64 * 1024**2) == "attention.streaming"
    # With room for neither, the remaining schedules need only statistics.
    assert attention(256, 16384, budget=1024**2) == "attention.portable"


def test_attention_falls_back_to_the_host_and_to_portable_matrix_hardware():
    assert attention(1, 65536, capability=HOST) == "attention.serial"
    assert attention(256, 4096, capability=HOST) == "attention.serial"
    no_subgroup = Capability(
        subgroup_width=1,
        threads_per_group=1024,
        shared_memory_bytes=65536,
        matrix_instructions=True,
    )
    assert attention(1, 65536, capability=no_subgroup) == "attention.portable"
    assert attention(256, 65536, capability=no_subgroup) == "attention.portable"


def test_wide_heads_leave_the_tiled_decode_schedules():
    assert attention(1, 65536, width=512) == "attention.portable"


def test_recurrence_owns_a_channel_per_lane_where_lanes_exist():
    shape = RecurrenceShape(1, 1, 4, 8, 128, 128, HeadMapping.TILED, DType.BF16)
    assert chosen("recurrence", RECURRENCE, shape, NATIVE_BF16, METAL) == (
        "recurrence.channel_simd"
    )
    assert chosen("recurrence", RECURRENCE, shape, NATIVE_BF16, HOST) == "recurrence.portable"


@pytest.mark.parametrize(
    "precision,capability,expected",
    [
        (NATIVE_BF16, METAL, "norm.subgroup"),
        (NATIVE_BF16, HOST, "norm.portable"),
        (MIXED_BF16, METAL, "norm.portable"),
        (MIXED_BF16_F32_RESIDUAL, METAL, "norm.portable"),
        (REFERENCE_F32, METAL, "norm.portable"),
    ],
)
def test_norm_uses_the_subgroup_chain_only_for_native_rounding(precision, capability, expected):
    shape = NormShape(1, 2048, 1e-6, DType.BF16, DType.BF16)
    assert chosen("norm", NORM, shape, precision, capability) == expected


@pytest.mark.parametrize(
    "representation,expected",
    [
        (MLX_Q4, "embedding.gather"),
        (Q4_K, "embedding.gather"),
        (F32, "embedding.gather"),
    ],
)
def test_embedding_gathers_from_whatever_the_table_is_resident_in(representation, expected):
    shape = EmbeddingShape(3, 1024, 2048, DType.BF16)
    layout = WeightLayout(representation, shape.vocabulary, shape.width)
    assert chosen("embedding", EMBEDDING, shape, NATIVE_BF16, METAL, layout) == expected


def gated(rows, representation, precision=NATIVE_BF16, capability=METAL):
    shape = GatedShape(rows, 4096, 2048, DType.BF16, DType.BF16)
    layout = WeightLayout(representation, 2 * shape.width, shape.inputs)
    return chosen("gated", GATED_TABLE, shape, precision, capability, layout)


@pytest.mark.parametrize(
    "rows,representation,precision,expected",
    [
        (1, MLX_Q4, NATIVE_BF16, "gated.direct_affine_fused_vector"),
        (1, MLX_Q4, MIXED_BF16, "gated.projected"),
        (8, MLX_Q4, NATIVE_BF16, "gated.projected"),
        (1, Q4_K, NATIVE_BF16, "gated.hierarchical_fused_vector"),
    ],
)
def test_gated_fusion_is_a_candidate_of_the_composite(rows, representation, precision, expected):
    assert gated(rows, representation, precision) == expected


def test_an_operation_with_no_applicable_candidate_says_so():
    shape = RecurrenceShape(1, 1, 4, 8, 128, 128, HeadMapping.TILED, DType.BF16)
    with pytest.raises(NoCandidate):
        select("recurrence", (), Selection(shape, NATIVE_BF16, METAL), available_bytes=UNBOUNDED)
