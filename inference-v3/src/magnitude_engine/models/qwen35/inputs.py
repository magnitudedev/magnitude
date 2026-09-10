"""Qwen input meaning and owned, already projected conditioning features.

Media preparation/encoding constructs this plan. Decoder continuation only
consumes its coordinates and features; it never interprets image markers.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, model_validator

from magnitude_engine.data import Record, TokenId
from magnitude_engine.inputs.layout import BoundaryRule, ConditioningIdentity, InputLayout
from magnitude_engine.platform.execution import DType, Tensor, TensorSpec

type Coordinate = Annotated[int, Field(ge=0, le=0x7FFFFFFF)]
type RotaryCoordinates = tuple[Coordinate, Coordinate, Coordinate]


class InputPlan(Record):
    tokens: tuple[Annotated[TokenId, Field(ge=0, le=0x7FFFFFFF)], ...]
    layout: InputLayout
    coordinates: tuple[RotaryCoordinates, ...] = ()
    continuation: Coordinate

    @model_validator(mode="after")
    def geometry(self):
        if len(self.tokens) != self.layout.count:
            raise ValueError("input layout and token count differ")
        if self.coordinates and len(self.coordinates) != len(self.tokens):
            raise ValueError("rotary coordinates must cover the complete prompt")
        if self.layout.spans and not self.coordinates:
            raise ValueError("conditioned Qwen input requires explicit rotary coordinates")
        if any(span.boundaries != BoundaryRule.CAUSAL for span in self.layout.spans):
            raise ValueError("Qwen decoder input spans are causal")
        if not self.coordinates and self.continuation != len(self.tokens):
            raise ValueError("text continuation must follow its prompt positions")
        return self

    @classmethod
    def text(cls, tokens: tuple[TokenId, ...]):
        return cls(tokens=tokens, layout=InputLayout(count=len(tokens)), continuation=len(tokens))

    def rotary(self, position: int, count: int) -> tuple[RotaryCoordinates, ...]:
        if (
            type(position) is not int
            or type(count) is not int
            or position < 0
            or count < 0
            or position + count > 0x7FFFFFFF
        ):
            raise ValueError("rotary input range must use nonnegative int32 positions")
        values = []
        for index in range(position, position + count):
            if index < len(self.tokens):
                values.append(self.coordinates[index] if self.coordinates else (index,) * 3)
            else:
                value = self.continuation + index - len(self.tokens)
                if value > 0x7FFFFFFF:
                    raise ValueError("rotary continuation exceeds int32 addressability")
                values.append((value,) * 3)
        return tuple(values)


@dataclass(frozen=True)
class Feature:
    identity: ConditioningIdentity
    values: Tensor


@dataclass(frozen=True)
class FeatureSlice:
    values: Tensor
    source: int
    destination: int
    count: int


@dataclass(frozen=True)
class Inputs:
    tokens: tuple[TokenId, ...]
    coordinates: tuple[RotaryCoordinates, ...]
    features: tuple[FeatureSlice, ...] = ()

    @classmethod
    def text(cls, tokens: tuple[TokenId, ...], position: int):
        return cls(
            tokens,
            tuple((index, index, index) for index in range(position, position + len(tokens))),
        )


class InputState:
    """Immutable semantic continuation with independently owned feature views.

    Advancing creates a new owner before commit. Checkpoints retain precisely the
    features still needed after their boundary, including a partially used image.
    """

    def __init__(self, plan: InputPlan, position: int, features: tuple[Feature, ...], width: int):
        if not plan.layout.boundary(position):
            raise ValueError("input continuation is not at a legal boundary")
        required = {span.identity for span in plan.layout.spans if span.end > position}
        supplied = {feature.identity: feature.values for feature in features}
        if len(supplied) != len(features) or set(supplied) != required:
            raise ValueError("conditioning features must exactly cover the unconsumed input")
        for span in plan.layout.spans:
            if span.end > position and supplied[span.identity].spec != TensorSpec(
                (span.end - span.start, width), DType.F32
            ):
                raise ValueError("projected feature geometry differs from its input span")
        with ExitStack() as cleanup:
            retained = []
            for feature in features:
                view = feature.values.view(feature.values.spec)
                cleanup.callback(view.close)
                retained.append(Feature(feature.identity, view))
            self._ownership = cleanup.pop_all()
        self.plan, self.position, self.width = plan, position, width
        self.features, self.closed = tuple(retained), False

    def after(self, position: int) -> InputState:
        self.check()
        if position < self.position:
            raise ValueError("input state cannot recover already released conditioning")
        remaining = {span.identity for span in self.plan.layout.spans if span.end > position}
        return InputState(
            self.plan,
            position,
            tuple(feature for feature in self.features if feature.identity in remaining),
            self.width,
        )

    def assemble(self, tokens: tuple[TokenId, ...]) -> Inputs:
        self.check()
        position, end = self.position, self.position + len(tokens)
        if not tokens or not self.plan.layout.boundary(end):
            raise ValueError("input advance requires a nonempty legal span")
        overlap = max(0, min(end, len(self.plan.tokens)) - position)
        if tokens[:overlap] != self.plan.tokens[position : position + overlap]:
            raise ValueError("input tokens differ from the bound prompt")
        features = {feature.identity: feature.values for feature in self.features}
        slices = tuple(
            FeatureSlice(
                features[span.identity],
                max(position, span.start) - span.start,
                max(position, span.start) - position,
                min(end, span.end) - max(position, span.start),
            )
            for span in self.plan.layout.spans
            if span.start < end and span.end > position
        )
        return Inputs(tokens, self.plan.rotary(position, len(tokens)), slices)

    def check(self) -> None:
        if self.closed:
            raise RuntimeError("input state is closed")

    def close(self) -> None:
        if not self.closed:
            self._ownership.close()
            self.closed = True
