from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.attention.contracts import PagedAttention


@component
class Attention(SubjectBlueprint):
    computation: Blueprint[PagedAttention]
    prefix_tokens: int
    query_tokens: int = 1
    query_heads: int = 16
    kv_heads: int = 2
    width: int = 256
    fragmented: bool = False

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import AttentionTrace

        return AttentionTrace
