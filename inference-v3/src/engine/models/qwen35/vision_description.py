"""Image-encoder geometry and artifact-independent parameter roles."""

from pydantic import PositiveInt, model_validator

from engine.data import Record
from engine.models.qwen35.preparation import ImageGeometry
from engine.weights.descriptor import WeightDescriptor


class VisionGeometry(Record):
    image: ImageGeometry
    hidden: PositiveInt
    intermediate: PositiveInt
    output: PositiveInt
    heads: PositiveInt
    depth: PositiveInt
    table_side: PositiveInt

    @model_validator(mode="after")
    def validate_heads(self):
        if self.hidden % (self.heads * 4):
            raise ValueError("vision heads must contain complete rotary quarter-pairs")
        return self


class AffineWeights(Record):
    weight: WeightDescriptor
    bias: WeightDescriptor


class VisionBlockWeights(Record):
    norm1: AffineWeights
    qkv: AffineWeights
    projection: AffineWeights
    norm2: AffineWeights
    up: AffineWeights
    down: AffineWeights


class VisionDescription(Record):
    geometry: VisionGeometry
    patch: AffineWeights
    positions: WeightDescriptor
    blocks: tuple[VisionBlockWeights, ...]
    merger_norm: AffineWeights
    merger_up: AffineWeights
    merger_down: AffineWeights

    @model_validator(mode="after")
    def validate_layers(self):
        if len(self.blocks) != self.geometry.depth:
            raise ValueError("vision depth differs from its parameter roles")
        return self
