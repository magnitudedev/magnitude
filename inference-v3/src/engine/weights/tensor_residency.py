"""Artifact metadata bindings; live imports and allocations belong exclusively to ops."""

from __future__ import annotations

import hashlib
from typing import Protocol

import ops

from .descriptor import Stored, StoredAffinePlanes, StoredDense, StoredQuantized, WeightDescriptor, WeightTransform
from .identity import ArtifactIdentity
from .residency import WeightResidency


class WeightFormat(Protocol):
    identity: ArtifactIdentity
    def stored(self, descriptor: WeightDescriptor) -> Stored: ...
    def close(self) -> None: ...


def describe_binding(format: WeightFormat, descriptor: WeightDescriptor, dtype: ops.DType,
                     residency: ops.Residency = ops.Residency.RESIDENT) -> ops.Binding:
    """Describe artifact storage without opening a device or reading tensor payloads.

    The format owner must outlive consumers of the returned source binding. Live
    imports, cache claims and their retirement remain DeviceRuntime's concern.
    """
    stored = format.stored(descriptor)
    elements = 1
    for size in descriptor.shape:
        elements *= size

    def plane(source, offset, length, group_elements, group_bytes):
        return ops.SourcePlane(ops.SourceSpan(source, offset, length), group_elements, group_bytes)

    if isinstance(stored, StoredDense):
        source_dtype = ops.DType(stored.dtype.value)
        spec = ops.TensorSpec(descriptor.shape, dtype)
        planes = (plane(stored.source, stored.offset, stored.nbytes, 1, source_dtype.itemsize),)
        recipe = ops.DenseImport(source_dtype, ops.Transform(descriptor.transform.value))
    elif isinstance(stored, StoredQuantized):
        if descriptor.transform != WeightTransform.IDENTITY:
            raise ValueError("encoded weights require identity value transformation")
        codec = stored.codec
        if elements % codec.block_elements:
            raise ValueError("encoded role ends inside a source block")
        spec = ops.TensorSpec(descriptor.shape, dtype,
                              representation=ops.execution_representation(stored.representation))
        planes = (plane(stored.source, stored.offset, elements // codec.block_elements * codec.block_bytes,
                        codec.block_elements, codec.block_bytes),)
        recipe = ops.EncodedImport(codec, codec.block_elements, codec.block_bytes)
    elif isinstance(stored, StoredAffinePlanes):
        if descriptor.transform != WeightTransform.IDENTITY:
            raise ValueError("encoded planes require identity value transformation")
        scale_dtype, bias_dtype = ops.DType(stored.scales.dtype.value), ops.DType(stored.biases.dtype.value)
        representation = ops.Affine(ops.Code(stored.bits), stored.group,
                                    ops.DirectCoefficients(scale_dtype, bias_dtype))
        spec = ops.TensorSpec(descriptor.shape, dtype, representation=representation)
        planes = (
            plane(stored.codes.source, stored.codes.offset, stored.codes.nbytes, 8 // stored.bits, 1),
            plane(stored.scales.source, stored.scales.offset, stored.scales.nbytes, stored.group, scale_dtype.itemsize),
            plane(stored.biases.source, stored.biases.offset, stored.biases.nbytes, stored.group, bias_dtype.itemsize),
        )
        recipe = ops.CanonicalImport()
    else:
        raise TypeError("unsupported stored weight representation")
    identity = hashlib.sha256(f"{format.identity}:{descriptor.model_dump_json()}:{dtype}".encode()).hexdigest()
    return ops.Binding(spec, identity, residency, planes, recipe)


class TensorWeights(WeightResidency):
    """Resident binding policy, with no eager reads, GPU code or resource owner."""

    residency = ops.Residency.RESIDENT

    def __init__(self, format: WeightFormat, device: ops.DeviceRuntime):
        self.format, self.device = format, device
        self._bindings: dict[tuple[str, ops.DType], ops.Binding] = {}
        self._closed = False

    @property
    def identity(self) -> ArtifactIdentity:
        return self.format.identity

    def bind(self, descriptor: WeightDescriptor, dtype: ops.DType) -> ops.Binding:
        if self._closed:
            raise RuntimeError("weight provider is closed")
        key = descriptor.model_dump_json(), dtype
        existing = self._bindings.get(key)
        if existing is not None:
            if existing.spec.shape != descriptor.shape:
                raise ValueError("one weight role cannot have conflicting geometry")
            return existing
        binding = describe_binding(self.format, descriptor, dtype, self.residency)
        self._bindings[key] = binding
        return binding

    def spec(self, descriptor: WeightDescriptor, dtype: ops.DType) -> ops.TensorSpec:
        return self.bind(descriptor, dtype).spec

    def close(self) -> None:
        if not self._closed:
            # Engine releases its artifact retention policy; ops remains the sole
            # live allocation/import owner and preserves outstanding leases.
            self.device.evict_bindings(tuple(self._bindings.values()))
            self._bindings.clear()
            self._closed = True


class StreamedWeights(TensorWeights):
    """The same values and formulas, with bounded transient access permitted."""

    residency = ops.Residency.STREAMED
