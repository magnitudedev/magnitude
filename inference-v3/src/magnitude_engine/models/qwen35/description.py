"""Qwen's geometry and the weight roles it asks a container for.

This module names what the architecture needs. Which container supplies it is
``formats/``; how those weights become resident is ``weights/``. Nothing here
imports either.
"""

import math
from enum import StrEnum

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

import magnitensor as mt
from magnitude_engine.data import Record
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.identity import ArtifactIdentity


class MixerKind(StrEnum):
    ATTENTION = "attention"
    RECURRENT = "recurrent"


class HeadMapping(StrEnum):
    GROUPED = "grouped"
    TILED = "tiled"


class ExpertGeometry(Record):
    count: PositiveInt
    selected: PositiveInt
    intermediate: PositiveInt
    shared_intermediate: PositiveInt
    normalize_selected: bool = True

    @model_validator(mode="after")
    def validate_experts(self):
        if self.selected > self.count:
            raise ValueError("selected expert count exceeds the expert collection")
        return self


class Geometry(Record):
    activation_dtype: mt.DType
    hidden: PositiveInt
    intermediate: PositiveInt
    vocabulary: PositiveInt
    context_limit: PositiveInt
    layers: tuple[MixerKind, ...]
    attention_heads: PositiveInt
    kv_heads: PositiveInt
    attention_width: PositiveInt
    rotary_width: PositiveInt
    rotary_base: PositiveFloat
    rotary_sections: tuple[int, int, int, int]
    epsilon: PositiveFloat
    convolution_width: PositiveInt
    recurrent_key_heads: PositiveInt
    recurrent_value_heads: PositiveInt
    recurrent_width: PositiveInt
    recurrent_head_mapping: HeadMapping = HeadMapping.TILED
    experts: ExpertGeometry | None = None

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.activation_dtype not in (mt.DType.F16, mt.DType.BF16):
            raise ValueError("Qwen activations require a qualified 16-bit floating dtype")
        if not math.isfinite(self.epsilon) or not math.isfinite(self.rotary_base):
            raise ValueError("Qwen numerical parameters must be finite")
        if not self.layers or self.attention_heads % self.kv_heads:
            raise ValueError("invalid Qwen layer/head geometry")
        if self.recurrent_value_heads % self.recurrent_key_heads:
            raise ValueError("recurrent value heads must be a multiple of key heads")
        if self.rotary_width % 2 or self.rotary_width > self.attention_width:
            raise ValueError("invalid partial rotary width")
        if (
            any(n < 0 for n in self.rotary_sections)
            or sum(self.rotary_sections) * 2 != self.rotary_width
        ):
            raise ValueError("rotary sections do not cover the rotary width")
        if self.rotary_sections[3] != 0:
            raise ValueError("Qwen's current input contract has three rotary axes")
        if self.convolution_width < 2:
            raise ValueError("Qwen recurrent convolution requires retained history")
        return self

    @property
    def recurrent_channels(self) -> int:
        return (2 * self.recurrent_key_heads + self.recurrent_value_heads) * self.recurrent_width


class AttentionWeights(Record):
    query_gate: WeightDescriptor
    key: WeightDescriptor
    value: WeightDescriptor
    query_norm: WeightDescriptor
    key_norm: WeightDescriptor
    output: WeightDescriptor


class RecurrentWeights(Record):
    query_key_value: WeightDescriptor
    gate: WeightDescriptor
    alpha: WeightDescriptor
    beta: WeightDescriptor
    convolution: WeightDescriptor
    decay: WeightDescriptor
    time_bias: WeightDescriptor
    norm: WeightDescriptor
    output: WeightDescriptor


class DenseFeedForwardWeights(Record):
    gate: WeightDescriptor
    up: WeightDescriptor
    down: WeightDescriptor


class RoutedFeedForwardWeights(Record):
    router: WeightDescriptor
    shared_router: WeightDescriptor
    expert_gate: WeightDescriptor
    expert_up: WeightDescriptor
    expert_down: WeightDescriptor
    shared_gate: WeightDescriptor
    shared_up: WeightDescriptor
    shared_down: WeightDescriptor


class BlockWeights(Record):
    input_norm: WeightDescriptor
    mixer: AttentionWeights | RecurrentWeights
    feedforward_norm: WeightDescriptor
    feedforward: DenseFeedForwardWeights | RoutedFeedForwardWeights


class DenseDescription(Record):
    artifact_identity: ArtifactIdentity = Field(pattern=r"^[0-9a-f]{64}$")
    geometry: Geometry
    embedding: WeightDescriptor
    output_norm: WeightDescriptor
    output: WeightDescriptor
    blocks: tuple[BlockWeights, ...]
