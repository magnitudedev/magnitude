from benchmarks.contracts import Subject, SubjectBlueprint
from magnitude_engine.artifacts.source import LocalArtifact
from magnitude_engine.composition import Blueprint, component


@component
class UpstreamDecode(SubjectBlueprint):
    artifact: Blueprint[LocalArtifact]
    prompt_tokens: int
    output_tokens: int
    service_tokens: int
    prefill_tokens: int = 512
    cache_limit_bytes: int = 256 << 20

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import DecodeTrace

        return DecodeTrace


@component
class UpstreamPrefill(SubjectBlueprint):
    artifact: Blueprint[LocalArtifact]
    prefix_tokens: int
    input_tokens: int
    prefill_tokens: int = 512
    cache_limit_bytes: int = 256 << 20

    @staticmethod
    def implementation() -> type[Subject]:
        from .runtime import PrefillTrace

        return PrefillTrace


@component
class UpstreamWaves(SubjectBlueprint):
    artifact: Blueprint[LocalArtifact]
    prompt_text: str
    prompt_tokens: int
    output_tokens: int
    rows: int
    prefill_tokens: int = 512
    cache_limit_bytes: int = 256 << 20

    @staticmethod
    def implementation() -> type[Subject]:
        from .batch import BatchTrace

        return BatchTrace
