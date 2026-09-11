"""One owner of the resident weights of one device context.

Every consumer of a weight gets a view of the same allocation. A group is the
concatenated residency of several weights that share an input width, so their
contraction can read the input once; whether a container can be grouped is a
property of its stored layout, not of the model that asked for the weights.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Protocol

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Prepared,
    Tensor,
    TensorSpec,
)
from magnitude_engine.weights.binding import resident_representation
from magnitude_engine.weights.descriptor import (
    Stored,
    StoredAffinePlanes,
    StoredBlocks,
    StoredDense,
    WeightDescriptor,
    WeightTransform,
)
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.representation import (
    EncodedBlocks,
    HierarchicalAffine,
    PlanarAffine,
    Representation,
    plane_offsets,
    resident_bytes,
)


class WeightFormat(Protocol):
    identity: ArtifactIdentity

    def stored(self, descriptor: WeightDescriptor) -> Stored: ...
    def close(self) -> None: ...


class ResidentWeight:
    """One resident weight and the representation its kernels read it in."""

    def __init__(
        self,
        descriptor: WeightDescriptor,
        representation: Representation,
        storage: Tensor,
        stored: Stored,
        *,
        elements: int | None = None,
        row_offset: int = 0,
        rows: int | None = None,
    ):
        self.descriptor = descriptor
        self.representation = representation
        self.stored = stored
        """What the container held. Retained so an independent reader can check
        this weight against its source without reopening the container."""
        self._storage = storage
        self._elements = math.prod(descriptor.shape) if elements is None else elements
        self._row_offset, self._rows = row_offset, rows
        self.nbytes = (
            storage.spec.nbytes
            if rows is None
            else resident_bytes(representation, rows * descriptor.shape[-1])
        )

    @property
    def context(self) -> DeviceContext:
        return self._storage.context

    def acquire(self, specs: tuple[TensorSpec, ...]) -> tuple[Tensor, ...]:
        """Views of this weight's storage matching a schedule's weight operands."""
        if len(specs) == 1:
            if specs[0].nbytes != self.nbytes:
                raise ValueError("consumer representation differs from resident weight size")
            start = 0
            if self._rows is not None and isinstance(
                self.representation, (EncodedBlocks, HierarchicalAffine)
            ):
                start = resident_bytes(
                    self.representation, self._row_offset * self.descriptor.shape[-1]
                )
            return (self._storage.view(specs[0], start),)
        representation = self.representation
        if not isinstance(representation, PlanarAffine) or len(specs) != 3:
            raise ValueError("only planar affine weights bind separate plane operands")
        offsets = plane_offsets(representation, self._elements)
        assert offsets.biases is not None
        starts = (offsets.low * 4, offsets.scales * 4, offsets.biases * 4)
        views: list[Tensor] = []
        try:
            for spec, start in zip(specs, starts, strict=True):
                stride = spec.nbytes // spec.shape[0]
                views.append(self._storage.view(spec, start + self._row_offset * stride))
            return tuple(views)
        except BaseException:
            for view in views:
                view.close()
            raise

    def single(self, spec: TensorSpec) -> Tensor:
        return self.acquire((spec,))[0]

    def close(self) -> None:
        self._storage.close()


class ResidentGroup:
    """Several weights sharing one input width, resident as one allocation."""

    def __init__(self, packed: ResidentWeight, widths: tuple[int, ...], inputs: int):
        self.packed, self.widths, self.inputs = packed, widths, inputs

    @property
    def representation(self) -> Representation:
        return self.packed.representation

    @property
    def context(self) -> DeviceContext:
        return self.packed.context

    def acquire(self, specs: tuple[TensorSpec, ...]) -> tuple[Tensor, ...]:
        return self.packed.acquire(specs)


class Weights:
    def __init__(
        self, format: WeightFormat, context: DeviceContext, capability: Capability | None = None
    ):
        self.format, self.context = format, context
        self.capability = context.capability if capability is None else capability
        self._resident: dict[str, ResidentWeight] = {}
        self._groups: dict[tuple[str, ...], ResidentGroup] = {}
        self._cleanup = ExitStack()
        self._closed = False

    @property
    def identity(self) -> ArtifactIdentity:
        return self.format.identity

    def _check(self) -> None:
        self.context.check()
        if self._closed:
            raise RuntimeError("weight residency is closed")

    def existing(self, descriptor: WeightDescriptor) -> ResidentWeight | None:
        """The already resident weight of this name, including a group member."""
        return self._resident.get(descriptor.name)

    def resident(self, descriptor: WeightDescriptor) -> ResidentWeight:
        self._check()
        if descriptor.name in self._resident:
            return self._resident[descriptor.name]
        stored = self.format.stored(descriptor)
        elements = math.prod(descriptor.shape)
        representation = resident_representation(stored, descriptor.shape, self.capability)
        storage = self._materialize(descriptor, stored, representation, elements)
        weight = ResidentWeight(descriptor, representation, storage, stored, elements=elements)
        self._cleanup.callback(weight.close)
        self._resident[descriptor.name] = weight
        return weight

    def group(self, descriptors: tuple[WeightDescriptor, ...]) -> ResidentGroup | None:
        """Concatenated residency, when the stored layout allows it."""
        self._check()
        if not descriptors:
            raise ValueError("a projection group needs at least one weight")
        key = tuple(descriptor.name for descriptor in descriptors)
        if key in self._groups:
            return self._groups[key]
        if len(set(key)) != len(key) or any(len(d.shape) != 2 for d in descriptors):
            raise ValueError("a projection group is distinct logical matrices")
        if len({d.shape[1] for d in descriptors}) != 1:
            raise ValueError("a projection group shares one input width")
        stored = tuple(self.format.stored(descriptor) for descriptor in descriptors)
        if any(descriptor.name in self._resident for descriptor in descriptors):
            raise ValueError("projection groups must be declared before individual residency")
        widths = tuple(descriptor.shape[0] for descriptor in descriptors)
        inputs = descriptors[0].shape[1]
        packed = WeightDescriptor(name="+".join(key), shape=(sum(widths), inputs))
        elements = sum(widths) * inputs
        if all(isinstance(entry, StoredAffinePlanes) for entry in stored):
            affine = tuple(entry for entry in stored if isinstance(entry, StoredAffinePlanes))
            representation = resident_representation(affine[0], packed.shape, self.capability)
            assert isinstance(representation, PlanarAffine)
            storage = self._affine_planes(affine, representation, elements, widths, inputs)
        elif all(isinstance(entry, StoredBlocks) for entry in stored):
            blocks = tuple(entry for entry in stored if isinstance(entry, StoredBlocks))
            if len({entry.layout for entry in blocks}) != 1 or any(
                descriptor.transform != WeightTransform.IDENTITY for descriptor in descriptors
            ):
                return None
            representation = resident_representation(blocks[0], packed.shape, self.capability)
            assert isinstance(representation, (EncodedBlocks, HierarchicalAffine))
            storage = self._block_rows(blocks, representation, widths, inputs)
        else:
            return None
        weight = ResidentWeight(packed, representation, storage, stored[0], elements=elements)
        self._cleanup.callback(weight.close)
        start = 0
        for descriptor, width, entry in zip(descriptors, widths, stored, strict=True):
            self._resident[descriptor.name] = ResidentWeight(
                descriptor,
                representation,
                storage,
                entry,
                elements=elements,
                row_offset=start,
                rows=width,
            )
            start += width
        result = ResidentGroup(weight, widths, inputs)
        self._groups[key] = result
        return result

    # ------------------------------------------------------------ materializing

    def _block_rows(
        self,
        stored: tuple[StoredBlocks, ...],
        representation: EncodedBlocks | HierarchicalAffine,
        widths: tuple[int, ...],
        inputs: int,
    ) -> Tensor:
        """Concatenate compatible row-major compact weights without changing their layout."""
        from magnitude_engine.platform.storage import ConcatenatedSource

        pieces = []
        for entry, width in zip(stored, widths, strict=True):
            nbytes = resident_bytes(representation, width * inputs)
            pieces.append((entry.source, entry.offset, nbytes))
        source = ConcatenatedSource(tuple(pieces))
        return self.context.upload_source(TensorSpec((source.size,), DType.U8), source, 0)

    def _materialize(
        self,
        descriptor: WeightDescriptor,
        stored: Stored,
        representation: Representation,
        elements: int,
    ) -> Tensor:
        if isinstance(stored, StoredAffinePlanes):
            assert isinstance(representation, PlanarAffine)
            return self._affine_planes(
                (stored,), representation, elements, (descriptor.shape[0],), descriptor.shape[1]
            )
        if isinstance(stored, StoredDense):
            return self._convert(descriptor, stored, elements)
        assert isinstance(stored, StoredBlocks)
        nbytes = resident_bytes(representation, elements)
        if descriptor.transform != WeightTransform.IDENTITY:
            raise ValueError("a container's blocked weights carry no declared transform")
        return self.context.upload_source(
            TensorSpec((nbytes,), DType.U8), stored.source, stored.offset
        )

    def _convert(self, descriptor: WeightDescriptor, stored: StoredDense, elements: int) -> Tensor:
        """Widen a stored floating parameter and apply its declared transform."""
        from magnitude_engine.kernels.copy.convert import convert

        if stored.dtype not in (DType.BF16, DType.F32):
            raise ValueError("a stored floating parameter must be BF16 or F32")
        if stored.dtype == DType.F32 and descriptor.transform == WeightTransform.IDENTITY:
            return self.context.upload_source(
                TensorSpec((elements,), DType.F32), stored.source, stored.offset
            )
        with ExitStack() as cleanup:
            source = self.context.upload_source(
                TensorSpec((elements,), stored.dtype), stored.source, stored.offset
            )
            cleanup.callback(source.close)
            kernel = self.context.specialize(
                convert,
                elements,
                stored.dtype,
                descriptor.transform,
                capability=self.capability,
            )
            target = self.context.allocate(TensorSpec((elements,), DType.F32))
            cleanup.callback(target.close)
            self.context.submit((Prepared(self.context, kernel, (source, target)),)).wait()
            owned = target.view(target.spec)
        return owned

    def _affine_planes(
        self,
        stored: tuple[StoredAffinePlanes, ...],
        representation: PlanarAffine,
        elements: int,
        widths: tuple[int, ...],
        inputs: int,
    ) -> Tensor:
        """Place every stored plane into the one allocation its readers address."""
        from magnitude_engine.platform.storage import ConcatenatedSource

        offsets = plane_offsets(representation, elements)
        assert offsets.biases is not None
        nbytes = offsets.words * 4
        rows = sum(widths)
        plan = (
            (offsets.low * 4, tuple(entry.codes for entry in stored), inputs // 8, DType.U32),
            (
                offsets.scales * 4,
                tuple(entry.scales for entry in stored),
                inputs // representation.group,
                representation.coefficient_dtype,
            ),
            (
                offsets.biases * 4,
                tuple(entry.biases for entry in stored),
                inputs // representation.group,
                representation.coefficient_dtype,
            ),
        )
        with ExitStack() as cleanup:
            target = self.context.allocate(TensorSpec((nbytes,), DType.U8))
            cleanup.callback(target.close)
            for start, planes, columns, dtype in plan:
                spec = TensorSpec((rows, columns), dtype)
                if spec.nbytes != sum(plane.nbytes for plane in planes):
                    raise ValueError("stored affine plane extents differ from the group geometry")
                source = ConcatenatedSource(
                    tuple((plane.source, plane.offset, plane.nbytes) for plane in planes)
                )
                view = target.view(TensorSpec((spec.nbytes,), DType.U8), start)
                try:
                    self.context.write_source(view, source)
                finally:
                    view.close()
            owned = target.view(target.spec)
        return owned

    def close(self) -> None:
        if not self._closed:
            self._cleanup.close()
            self._resident.clear()
            self._groups.clear()
            self._closed = True
