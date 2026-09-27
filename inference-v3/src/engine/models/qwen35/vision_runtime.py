"""Owned asynchronous image encoding; decoder input leases consume its features."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, cast

import numpy as np

import ops
from engine.models.qwen35.inputs import Feature
from engine.models.qwen35.preparation import ImagePatches, PreparedInput, spatial_controls
from engine.models.qwen35.vision_description import VisionDescription
from engine.models.qwen35.vision_program import define, weight_roles
from engine.weights.residency import WeightResidency

if TYPE_CHECKING:
    from engine.models.qwen35.runtime import DenseRuntime


class ImageSource:
    """Original pixels survive decoder eviction; projected features are shared leases."""

    def __init__(self, model: DenseRuntime, encoder: VisionEncoder, prepared: PreparedInput):
        if (
            model.closed
            or encoder.closed
            or model.device is not encoder.device
            or not prepared.plan.tokens
            or len(prepared.plan.tokens) > model.context_capacity
            or any(token >= model.geometry.vocabulary for token in prepared.plan.tokens)
        ):
            raise ValueError("image input differs from the bound execution owner or input limits")
        self.model, self.encoder, self.prepared = model, encoder, prepared
        self.images = tuple({image.identity: image for image in prepared.images}.values())
        self.encoded: list[EncodedImage] = []
        self.pending: _Encoding | None = None
        self.closed = False

    @property
    def prompt(self):
        return self.prepared.plan.tokens

    def prepare(self):
        if self.closed or self.pending is not None:
            raise RuntimeError("image source is closed or already preparing")
        if len(self.encoded) == len(self.images):
            return None
        self.pending = _Encoding(self, self.encoder.submit(self.images[len(self.encoded)]))
        return self.pending

    def open(self):
        if self.closed or self.pending is not None or len(self.encoded) != len(self.images):
            raise RuntimeError("image source has not finished preparation")
        return self.model.create(self.prepared.plan, tuple(item.feature for item in self.encoded))

    def close(self):
        if not self.closed:
            self.closed = True
            if self.pending is not None:
                self.pending.close()
            for item in self.encoded:
                item.close()
            self.encoded.clear()


class _Encoding:
    def __init__(self, source: ImageSource, image: EncodedImage):
        self.source, self.image = source, image
        self.completion = image.completion
        self.committed = self.closed = False

    def finish(self):
        if self.closed or self.source.closed:
            return
        if not self.completion.done:
            raise RuntimeError("cannot publish unfinished image features")
        self.completion.wait()
        self.source.encoded.append(self.image)
        self.source.pending = None
        self.committed = True

    def close(self):
        if not self.closed:
            if not self.committed:
                self.image.close()
            if self.source.pending is self:
                self.source.pending = None
            self.closed = True


class EncodedImage:
    def __init__(self, identity, execution: ops.Execution, owned: ExitStack):
        self.feature = Feature(identity, execution.outputs[0])
        self.completion = execution.completion
        self._owned = owned
        self.closed = False

    def close(self):
        if not self.closed:
            self.feature.values.close()
            self._owned.close()
            self.closed = True


class VisionEncoder:
    def __init__(
        self, description: VisionDescription, device: ops.DeviceRuntime, weights: WeightResidency
    ):
        self.description, self.device = description, device
        self.bindings = {
            role.name: weights.bind(role, ops.DType.BF16) for role in weight_roles(description)
        }
        self._programs: dict[int, ops.CompiledFunction] = {}
        self.closed = False

    def submit(self, image: ImagePatches) -> EncodedImage:
        self.device.check()
        if self.closed:
            raise RuntimeError("image encoder is closed")
        g = self.description.geometry
        rows = image.grid[0] * image.grid[1] * image.grid[2]
        if image.pixels.dtype != "float32" or image.pixels.shape != (rows, g.image.patch_width):
            raise ValueError("image patches differ from the bound encoder")
        coordinates, indices, coefficients = spatial_controls(
            image.grid, g.image.merge, g.table_side
        )
        program = self._programs.get(rows)
        if program is None:
            definition = define(
                self.description, {k: v.spec for k, v in self.bindings.items()}, rows
            )
            program = ops.compile(
                definition.function,
                signature=definition.signature,
                device=self.device,
                constants=cast(Mapping[int | str, ops.Resource | ops.Binding], self.bindings),
                options=definition.options,
            )
            self._programs[rows] = program
        with ExitStack() as cleanup:
            visible = np.tile(np.array([[0, rows]], np.int32), (rows, 1))
            payloads = [image.pixels.array(), coordinates, visible, *indices, *coefficients]
            payloads.extend(np.array([i], np.int32) for i in range(3))
            inputs = []
            for value in payloads:
                dtype = ops.DType.I32 if value.dtype == np.int32 else ops.DType.F32
                resource = self.device.upload(ops.TensorSpec(value.shape, dtype), value.tobytes())
                cleanup.callback(resource.close)
                inputs.append(resource)
            execution = program.submit(*inputs)
            return EncodedImage(image.identity, execution, cleanup.pop_all())

    def close(self):
        if not self.closed:
            for program in self._programs.values():
                program.close()
            self._programs.clear()
            self.closed = True
