"""Artifact-to-Magnitensor residency with TileLang-only numerical conversion."""

from __future__ import annotations

import math
import struct
from contextlib import ExitStack
from typing import Protocol

import magnitensor as mt
from magnitude_engine.weights.descriptor import (
    Stored,
    StoredAffinePlanes,
    StoredDense,
    StoredQuantized,
    WeightDescriptor,
    WeightTransform,
)
from magnitude_engine.weights.identity import ArtifactIdentity

IMPORT_CHUNK_BYTES = 8 * 1024**2


class WeightFormat(Protocol):
    identity: ArtifactIdentity

    def stored(self, descriptor: WeightDescriptor) -> Stored: ...
    def close(self) -> None: ...


class TensorWeights:
    """One canonical Magnitensor resource for each architecture weight role."""

    def __init__(self, format: WeightFormat, device: mt.Device):
        self.format = format
        self.device = device
        self._resident: dict[tuple[str, mt.DType], mt.Resource] = {}
        self._specs: dict[tuple[str, mt.DType], mt.TensorSpec] = {}
        self._cleanup = ExitStack()
        self._closed = False

    @property
    def identity(self) -> ArtifactIdentity:
        return self.format.identity

    def resident(self, descriptor: WeightDescriptor, dtype: mt.DType) -> mt.Resource:
        if self._closed:
            raise RuntimeError("weight residency is closed")
        key = descriptor.name, dtype
        existing = self._resident.get(key)
        if existing is not None:
            return existing
        stored = self.format.stored(descriptor)
        if isinstance(stored, StoredDense):
            resource = self._dense(descriptor, stored, dtype)
        elif isinstance(stored, StoredQuantized):
            resource = self._quantized(descriptor, stored, dtype)
        elif isinstance(stored, StoredAffinePlanes):
            resource = self._planes(descriptor, stored, dtype)
        else:
            raise TypeError("unknown stored weight representation")
        self._cleanup.callback(resource.close)
        self._resident[key] = resource
        return resource

    def spec(self, descriptor: WeightDescriptor, dtype: mt.DType) -> mt.TensorSpec:
        """Describe canonical residency without reading payload bytes or allocating it."""
        if self._closed:
            raise RuntimeError("weight residency is closed")
        key = descriptor.name, dtype
        existing = self._resident.get(key)
        if existing is not None:
            return existing.spec
        cached = self._specs.get(key)
        if cached is not None:
            return cached
        stored = self.format.stored(descriptor)
        if isinstance(stored, StoredQuantized):
            representation = _representation(
                stored.representation, descriptor.shape, self.device.capabilities
            )
            result = mt.TensorSpec(descriptor.shape, dtype, representation=representation)
        elif isinstance(stored, StoredAffinePlanes):
            result = mt.TensorSpec(
                descriptor.shape,
                dtype,
                representation=mt.Affine(
                    mt.Code(stored.bits),
                    stored.group,
                    mt.DirectCoefficients(
                        mt.DType(stored.scales.dtype.value),
                        mt.DType(stored.biases.dtype.value),
                    ),
                ),
            )
        elif isinstance(stored, StoredDense):
            result = mt.TensorSpec(descriptor.shape, dtype)
        else:
            raise TypeError("unknown stored weight representation")
        self._specs[key] = result
        return result

    def _dense(
        self, descriptor: WeightDescriptor, stored: StoredDense, dtype: mt.DType
    ) -> mt.Resource:
        source_dtype = mt.DType(stored.dtype.value)
        content = stored.source.read(stored.offset, stored.nbytes)
        elements = math.prod(descriptor.shape)
        if descriptor.transform == WeightTransform.IDENTITY:
            if source_dtype == dtype:
                return self.device.upload(mt.TensorSpec(descriptor.shape, source_dtype), content)
            source_spec = mt.TensorSpec(
                descriptor.shape,
                mt.DType.U16 if source_dtype == mt.DType.BF16 else source_dtype,
            )
            source = self.device.upload(source_spec, content)
            try:
                compiled = mt.compile(
                    (
                        (lambda value: mt.decode_bfloat16(value, dtype))
                        if source_dtype == mt.DType.BF16
                        else (lambda value: mt.cast(value, dtype))
                    ),
                    signature=mt.Signature((mt.Argument(source_spec, "source"),)),
                    device=self.device,
                    constants={},
                    options=mt.CompileOptions(mode="residency"),
                )
                try:
                    execution = compiled.submit(source)
                    execution.completion.wait()
                    result = execution.outputs[0]
                    if result.spec != mt.TensorSpec(descriptor.shape, dtype):
                        result.close()
                        raise RuntimeError("dense weight conversion produced an invalid resource")
                    return result
                finally:
                    compiled.close()
            finally:
                source.close()
        if descriptor.transform != WeightTransform.NEGATIVE_EXP:
            raise ValueError(f"unsupported weight transform {descriptor.transform}")
        source = self.device.upload(
            mt.TensorSpec(
                (elements,), mt.DType.U16 if source_dtype == mt.DType.BF16 else source_dtype
            ),
            content,
        )
        try:
            compiled = mt.compile(
                lambda value: mt.exp(
                    mt.decode_bfloat16(value, mt.DType.F32)
                    if source_dtype == mt.DType.BF16
                    else mt.cast(value, mt.DType.F32)
                )
                * -1.0,
                signature=mt.Signature((mt.Argument(source.spec, "source"),)),
                device=self.device,
                constants={},
                options=mt.CompileOptions(mode="residency"),
            )
            try:
                execution = compiled.submit(source)
                execution.completion.wait()
                result = execution.outputs[0]
                if result.spec.shape != (elements,) or result.spec.dtype != mt.DType.F32:
                    result.close()
                    raise RuntimeError("weight transform produced an invalid resource")
                reshaped = result.view(mt.TensorSpec(descriptor.shape, mt.DType.F32))
                result.close()
                return reshaped
            finally:
                compiled.close()
        finally:
            source.close()

    def _quantized(
        self, descriptor: WeightDescriptor, stored: StoredQuantized, dtype: mt.DType
    ) -> mt.Resource:
        if descriptor.transform != WeightTransform.IDENTITY:
            raise ValueError("quantized weights cannot carry a post-import transform")
        representation = _representation(
            stored.representation, descriptor.shape, self.device.capabilities
        )
        target_spec = mt.TensorSpec(descriptor.shape, dtype, representation=representation)
        target = self.device.allocate(target_spec)
        codec = stored.codec
        elements = math.prod(descriptor.shape)
        if elements % codec.block_elements:
            target.close()
            raise ValueError("quantized weight ends in a partial source block")
        tiles = elements // codec.block_elements
        staged_tiles = min(tiles, max(1, IMPORT_CHUNK_BYTES // codec.block_bytes))
        stage_spec = mt.TensorSpec((staged_tiles * codec.block_bytes,), mt.DType.U8)
        extent_spec = mt.TensorSpec((2,), mt.DType.I32)

        def import_chunk(source, extent, *, target):
            return mt.quantized_import(
                source,
                target,
                extent,
                codec=codec,
                staged_tiles=staged_tiles,
            )

        compiled = mt.compile(
            import_chunk,
            signature=mt.Signature(
                (mt.Argument(stage_spec, "source"), mt.Argument(extent_spec, "extent")),
                {"target": mt.Argument(target_spec, "target", mt.ValueKind.RESOURCE)},
            ),
            device=self.device,
            constants={},
            options=mt.CompileOptions(mode="residency"),
        )
        try:
            for first in range(0, tiles, staged_tiles):
                valid = min(staged_tiles, tiles - first)
                content = stored.source.read(
                    stored.offset + first * codec.block_bytes,
                    valid * codec.block_bytes,
                )
                content += bytes(stage_spec.storage_nbytes - len(content))
                stage = self.device.upload(stage_spec, content)
                extent = self.device.upload(extent_spec, struct.pack("=ii", valid, first))
                try:
                    execution = compiled.submit(stage, extent, resources={"target": target})
                    execution.completion.wait()
                    for output in execution.outputs:
                        output.close()
                finally:
                    extent.close()
                    stage.close()
        except BaseException:
            target.close()
            raise
        finally:
            compiled.close()
        return target

    def _planes(
        self, descriptor: WeightDescriptor, stored: StoredAffinePlanes, dtype: mt.DType
    ) -> mt.Resource:
        # MLX affine planes are already in Magnitensor's canonical plane order.
        representation = mt.Affine(
            mt.Code(stored.bits),
            stored.group,
            mt.DirectCoefficients(
                mt.DType(stored.scales.dtype.value), mt.DType(stored.biases.dtype.value)
            ),
        )
        content = b"".join(
            plane.source.read(plane.offset, plane.nbytes)
            for plane in (stored.codes, stored.scales, stored.biases)
        )
        return self.device.upload(
            mt.TensorSpec(descriptor.shape, dtype, representation=representation), content
        )

    def close(self) -> None:
        if not self._closed:
            self._cleanup.close()
            self._resident.clear()
            self._specs.clear()
            self._closed = True


def _representation(
    value: mt.Affine | mt.Codebook,
    shape: tuple[int, ...],
    capabilities: mt.Capabilities,
):
    if (
        isinstance(value, mt.Affine)
        and len(shape) == 3
        and capabilities.matrix_instructions
        and _scale_min_hierarchy(value)
    ):
        return mt.Affine(
            value.code,
            value.group,
            mt.DirectCoefficients(mt.DType.F32, mt.DType.F32),
        )
    return value


def _scale_min_hierarchy(value: mt.Affine | mt.Codebook) -> bool:
    coefficients = value.coefficients if isinstance(value, mt.Affine) else None
    return (
        isinstance(value, mt.Affine)
        and isinstance(coefficients, mt.HierarchicalCoefficients)
        and value.code.low_bits == 4
        and value.code.high_bits in (0, 1)
        and value.group == 32
        and coefficients.supergroup == 256
        and coefficients.local_scale_bits == 6
        and coefficients.local_bias_bits == 6
        and coefficients.bias_sign == -1
    )
