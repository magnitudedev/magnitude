"""The upstream call and parameter owner are explicit library bindings."""

from magnitude_engine.models.architectures.gemma4.vision import GemmaVision
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryForward, LibraryProgram
from magnitude_engine.models.architectures.qwen35.vision import QwenVision
from magnitude_engine.models.features import FeatureCache
from performance.bindings import Fields, Use, schema
from performance.facts import Configuration, OpaqueParameters


@schema(LibraryProgram)
def library(a: LibraryProgram, _: None) -> Fields[OpaqueParameters]:
    if not isinstance(a.call, LibraryForward):
        raise TypeError("library capture requires an explicitly bound LibraryForward")
    return Fields(
        OpaqueParameters(),
        operands={"model": a.call.model},
        sources=(a.call,) if a.input_forward is None else (a.call, a.input_forward),
    )


@schema(QwenVision)
def qwen_vision(a: QwenVision, _: None) -> Fields[OpaqueParameters]:
    return Fields(
        OpaqueParameters(),
        operands={"encoder_projector": a.model},
        dependencies={"features": Use(a.cache)},
    )


@schema(GemmaVision)
def gemma_vision(a: GemmaVision, _: None) -> Fields[OpaqueParameters]:
    return Fields(
        OpaqueParameters(),
        operands={"encoder_projector": a.model},
        dependencies={"features": Use(a.cache)},
    )


@schema(FeatureCache)
def input_features(a: FeatureCache, _: None) -> Fields[Configuration]:
    return Fields(Configuration(settings={"max_bytes": a.max_bytes}))
