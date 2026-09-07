"""Operand and use schemas for reusable model operations."""

from magnitude_engine import components as c
from magnitude_engine.models.attention.gathered import GatheredAttention
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding, ResidentEmbedding
from magnitude_engine.models.embeddings.streaming import RowLease, StreamedEmbedding
from magnitude_engine.models.experts.computation import ResidentExperts
from magnitude_engine.models.experts.streaming import StreamedExperts
from magnitude_engine.models.ownership import OwnedProgram, VocabularyLoan
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.recurrence.mlx import MLXDelta
from magnitude_engine.models.recurrence.reference import DeltaReference
from magnitude_engine.models.state.paged import PagedStateStore
from performance.bindings import Fields, Use, alias, neural, schema


@schema(GatheredAttention, context=c.AttentionGeometry)
def gathered(a: GatheredAttention, geometry: c.AttentionGeometry) -> Fields[c.AttentionGeometry]:
    return Fields(geometry)


@schema(MetalPagedAttention, context=c.AttentionGeometry)
def paged(a: MetalPagedAttention, geometry: c.AttentionGeometry) -> Fields[c.AttentionGeometry]:
    return Fields(
        geometry,
        children={"fallback": Use(a.prefill, geometry)},
        configuration=c.Configuration(
            settings={"partition_tokens": a.partition_tokens, "heads_per_group": a.heads_per_group}
        ),
    )


@schema(DeltaReference, context=c.RecurrentGeometry)
def reference(a: DeltaReference, geometry: c.RecurrentGeometry) -> Fields[c.RecurrentGeometry]:
    return Fields(geometry)


@schema(MLXDelta, context=c.RecurrentGeometry)
def mlx(a: MLXDelta, geometry: c.RecurrentGeometry) -> Fields[c.RecurrentGeometry]:
    return Fields(geometry)


@schema(MetalDelta, context=c.RecurrentGeometry)
def delta(a: MetalDelta, geometry: c.RecurrentGeometry) -> Fields[c.RecurrentGeometry]:
    return Fields(
        geometry,
        configuration=c.Configuration(settings={"specialize_prefill": a.specialize_prefill}),
    )


@schema(ResidentEmbedding)
def embedding(a: ResidentEmbedding, _: None) -> Fields[c.NeuralParameters]:
    return neural(operands={"weight": a.weight}, weight_use=c.WeightUse.EMBEDDING)


@schema(ResidentAffineEmbedding)
def affine(a: ResidentAffineEmbedding, _: None) -> Fields[c.NeuralParameters]:
    return neural(
        operands={"weight": a.weight, "scales": a.scales, "biases": a.biases},
        weight_use=c.WeightUse.EMBEDDING,
    )


@schema(ResidentExperts, context=int)
def experts(a: ResidentExperts, top_k: int) -> Fields[c.NeuralParameters]:
    return neural(
        operands={"up": a.weights.up, "gate": a.weights.gate, "down": a.weights.down},
        top_k=top_k,
        weight_use=c.WeightUse.EXPERTS,
        sources=(a.math.apply, a.math.activation),
    )


@schema(StreamedEmbedding)
def streamed_embedding(a: StreamedEmbedding, _: None) -> Fields[c.NeuralParameters]:
    table = a.table
    return Fields(
        c.NeuralParameters(
            arrays={
                "encoded": c.TensorFacts(
                    identity="encoded",
                    shape=(table.rows, table.row_bytes),
                    bytes=table.rows * table.row_bytes,
                    dtype="uint8",
                )
            },
            weight_use=c.WeightUse.EMBEDDING,
            settings={"cache_bytes": a.cache_bytes, "max_pending": a.max_pending},
        ),
        sources=(RowLease, a.reader),
    )


@schema(StreamedExperts, context=int)
def streamed_experts(a: StreamedExperts, top_k: int) -> Fields[c.NeuralParameters]:
    source = a.source
    return Fields(
        c.NeuralParameters(
            arrays={
                "encoded": c.TensorFacts(
                    identity="encoded",
                    shape=(source.experts, source.expert_bytes),
                    bytes=source.experts * source.expert_bytes,
                    dtype="uint8",
                )
            },
            weight_use=c.WeightUse.EXPERTS,
            top_k=top_k,
            settings={"slots": a.bank.slots},
        ),
        sources=(a.math, a.bank, a.reader, a.residency),
    )


alias(OwnedProgram, lambda a: a.bindings())
alias(VocabularyLoan, lambda a: a.bindings())
alias(PagedStateStore, lambda a: a.pages)
