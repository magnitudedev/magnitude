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
    SubmissionError,
    Tensor,
    TensorSpec,
    Ticket,
)
from magnitude_engine.weights.binding import resident_representation
from magnitude_engine.weights.descriptor import (
    Stored,
    StoredAffinePlanes,
    StoredDense,
    StoredQuantized,
    WeightDescriptor,
    WeightTransform,
)
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.representation import (
    Affine,
    Codebook,
    Representation,
    WeightLayout,
    canonical_layout,
    resident_bytes,
)

IMPORT_CHUNK_BYTES = 8 * 1024**2


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
        *,
        full_rows: int | None = None,
        row_offset: int = 0,
        rows: int | None = None,
    ):
        self.descriptor = descriptor
        self.representation = representation
        self._storage = storage
        matrix = len(descriptor.shape) > 1
        logical_rows = (descriptor.shape[0] if matrix else 1) if rows is None else rows
        backing_rows = logical_rows if full_rows is None else full_rows
        columns = descriptor.shape[-1] if matrix else descriptor.shape[0]
        self.layout = WeightLayout(
            representation,
            backing_rows,
            columns,
            first_row=row_offset,
            row_count=logical_rows,
        )
        self.nbytes = resident_bytes(representation, logical_rows * columns)

    @property
    def context(self) -> DeviceContext:
        return self._storage.context

    def acquire(self, specs: tuple[TensorSpec, ...]) -> tuple[Tensor, ...]:
        """The one backing operand read through this weight's logical layout."""
        if len(specs) != 1 or specs[0].nbytes != self._storage.spec.nbytes:
            raise ValueError("a resident weight binds exactly one complete backing allocation")
        return (self._storage.view(specs[0]),)

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
    def layout(self) -> WeightLayout:
        return self.packed.layout

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
        weight = ResidentWeight(descriptor, representation, storage)
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
            assert isinstance(representation, Affine)
            storage = self._affine_planes(affine, representation, elements, widths, inputs)
        elif all(isinstance(entry, StoredQuantized) for entry in stored):
            quantized = tuple(entry for entry in stored if isinstance(entry, StoredQuantized))
            if len({entry.representation for entry in quantized}) != 1 or any(
                descriptor.transform != WeightTransform.IDENTITY for descriptor in descriptors
            ):
                return None
            representation = resident_representation(quantized[0], packed.shape, self.capability)
            assert isinstance(representation, (Affine, Codebook))
            storage = self._quantized_rows(quantized, representation, widths, inputs)
        else:
            return None
        weight = ResidentWeight(packed, representation, storage)
        self._cleanup.callback(weight.close)
        start = 0
        for descriptor, width in zip(descriptors, widths, strict=True):
            self._resident[descriptor.name] = ResidentWeight(
                descriptor,
                representation,
                storage,
                full_rows=sum(widths),
                row_offset=start,
                rows=width,
            )
            start += width
        result = ResidentGroup(weight, widths, inputs)
        self._groups[key] = result
        return result

    # ------------------------------------------------------------ materializing

    def _quantized_rows(
        self,
        stored: tuple[StoredQuantized, ...],
        representation: Affine | Codebook,
        widths: tuple[int, ...],
        inputs: int,
    ) -> Tensor:
        """Import each source directly into its final canonical row range."""
        elements = sum(widths) * inputs
        target = self.context.allocate(
            TensorSpec((resident_bytes(representation, elements),), DType.U8)
        )
        try:
            first_row = 0
            for entry, width in zip(stored, widths, strict=True):
                self._relayout(
                    entry,
                    target,
                    representation,
                    elements,
                    first_row * inputs,
                    width * inputs,
                )
                first_row += width
            return target
        except BaseException:
            target.close()
            raise

    def _materialize(
        self,
        descriptor: WeightDescriptor,
        stored: Stored,
        representation: Representation,
        elements: int,
    ) -> Tensor:
        if isinstance(stored, StoredAffinePlanes):
            assert isinstance(representation, Affine)
            return self._affine_planes(
                (stored,), representation, elements, (descriptor.shape[0],), descriptor.shape[1]
            )
        if isinstance(stored, StoredDense):
            return self._convert(descriptor, stored, elements)
        assert isinstance(stored, StoredQuantized)
        assert isinstance(representation, (Affine, Codebook))
        if descriptor.transform != WeightTransform.IDENTITY:
            raise ValueError("a container's quantized weights carry no declared transform")
        return self._quantized_rows(
            (stored,), representation, (descriptor.shape[0],), descriptor.shape[-1]
        )

    def _relayout(
        self,
        stored: StoredQuantized,
        target: Tensor,
        representation: Affine | Codebook,
        total_elements: int,
        target_element: int,
        source_elements: int,
    ) -> None:
        """Transform bounded source chunks directly into canonical residency."""
        from magnitude_engine.kernels.copy.relayout import relayout

        codec = stored.codec
        if source_elements % codec.block_elements or target_element % codec.block_elements:
            raise ValueError("quantized rows must contain complete source blocks")
        total_tiles = source_elements // codec.block_elements
        staged_tiles = min(total_tiles, max(1, IMPORT_CHUNK_BYTES // codec.block_bytes))
        stage_bytes = staged_tiles * codec.block_bytes
        kernel = self.context.specialize(
            relayout,
            total_elements,
            representation,
            codec,
            staged_tiles,
            capability=self.capability,
        )
        pending: list[Ticket] = []
        try:
            for first in range(0, total_tiles, staged_tiles):
                # Bound staging to two slots. Retiring the older slot leaves the
                # immediately preceding kernel running while this source read occurs.
                if len(pending) == 2:
                    pending.pop(0).wait()
                valid = min(staged_tiles, total_tiles - first)
                source_offset = stored.offset + first * codec.block_bytes
                content = stored.source.read(source_offset, valid * codec.block_bytes)
                if len(content) != valid * codec.block_bytes:
                    raise ValueError("artifact source returned a short quantized read")
                content += bytes(stage_bytes - len(content))
                with ExitStack() as cleanup:
                    stage = self.context.upload(kernel.signature[0], content)
                    cleanup.callback(stage.close)
                    extent = self.context.indices(
                        (valid, target_element // codec.block_elements + first)
                    )
                    cleanup.callback(extent.close)
                    command = Prepared(self.context, kernel, (stage, target, extent))
                    cleanup.callback(command.close)
                    try:
                        pending.append(self.context.submit((command,)))
                    except SubmissionError as error:
                        pending.append(error.ticket)
                        raise
            for ticket in pending:
                ticket.wait()
        except BaseException as primary:
            cleanup_errors: list[BaseException] = []
            for ticket in pending:
                if not ticket.done:
                    try:
                        ticket.wait()
                    except BaseException as error:
                        cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "quantized import and in-flight cleanup failed", (primary, *cleanup_errors)
                ) from primary
            raise

    def _convert(self, descriptor: WeightDescriptor, stored: StoredDense, elements: int) -> Tensor:
        """Widen a stored floating parameter and apply its declared transform."""
        from magnitude_engine.kernels.copy.convert import convert

        if stored.dtype not in (DType.F16, DType.BF16, DType.F32):
            raise ValueError("a stored floating parameter must be F16, BF16 or F32")
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
        representation: Affine,
        elements: int,
        widths: tuple[int, ...],
        inputs: int,
    ) -> Tensor:
        """Place every stored plane into the one allocation its readers address."""
        from magnitude_engine.platform.storage import ConcatenatedSource

        layout = canonical_layout(representation, elements)
        assert layout.biases is not None
        nbytes = layout.nbytes
        rows = sum(widths)
        plan = (
            (layout.low, tuple(entry.codes for entry in stored), inputs // 8, DType.U32),
            (
                layout.scales,
                tuple(entry.scales for entry in stored),
                inputs // representation.group,
                stored[0].scales.dtype,
            ),
            (
                layout.biases,
                tuple(entry.biases for entry in stored),
                inputs // representation.group,
                stored[0].biases.dtype,
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
