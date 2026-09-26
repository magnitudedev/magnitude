"""Immutable numerical values and bounded source access, independent of containers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, TYPE_CHECKING

from .tensor.graph import _stable
from .tensor.types import DType, TensorSpec

if TYPE_CHECKING:
    from .runtime.resources import Resource


class SourceKind(StrEnum):
    FILE = "file"
    MEMORY = "memory"
    GENERATED = "generated"
    COMPOSITE = "composite"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Source-path provenance, not a claim about physical disk traffic or speed."""

    identity: str
    revision: str
    kind: SourceKind
    location: str | None = None
    dependencies: tuple[SourceInfo, ...] = ()

    def __post_init__(self):
        if not self.identity or not self.revision:
            raise ValueError("source provenance requires identity and snapshot revision")
        if self.kind == SourceKind.COMPOSITE and not self.dependencies:
            raise ValueError("composite source provenance must name its backing sources")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(_stable(self), sort_keys=True).encode()).hexdigest()


class ByteSource(Protocol):
    @property
    def info(self) -> SourceInfo: ...
    @property
    def size(self) -> int: ...
    def read(self, offset: int, length: int) -> bytes: ...
    def read_into(self, offset: int, destination: memoryview) -> int: ...


@dataclass(frozen=True, slots=True)
class MemorySource:
    """Immutable in-memory source with value-derived provenance."""

    content: bytes
    info: SourceInfo = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "content", bytes(self.content))
        digest = hashlib.sha256(self.content).hexdigest()
        object.__setattr__(self, "info", SourceInfo(f"memory:{digest}", digest, SourceKind.MEMORY))

    @property
    def size(self) -> int:
        return len(self.content)

    def read(self, offset: int, length: int) -> bytes:
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("memory source read outside range")
        return self.content[offset:offset + length]

    def read_into(self, offset: int, destination: memoryview) -> int:
        length = destination.nbytes
        if offset < 0 or offset + length > self.size:
            raise ValueError("memory source read outside range")
        destination[:] = memoryview(self.content)[offset:offset + length]
        return length


class Residency(StrEnum):
    RESIDENT = "resident"
    STREAMED = "streamed"


class Transform(StrEnum):
    IDENTITY = "identity"
    NEGATIVE_EXP = "negative_exp"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    source: ByteSource = field(compare=False, repr=False)
    offset: int
    length: int

    def __post_init__(self):
        if self.offset < 0 or self.length <= 0 or self.offset + self.length > self.source.size:
            raise ValueError("binding source span exceeds its immutable source")


@dataclass(frozen=True, slots=True)
class ZeroSource:
    """Virtual initialized storage for bounded padding, with no eager allocation."""
    size: int

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("zero source requires a positive byte extent")

    @property
    def info(self) -> SourceInfo:
        return SourceInfo(f"zero:{self.size}", "zero-v1", SourceKind.GENERATED)

    def read(self, offset: int, length: int) -> bytes:
        if min(offset, length) < 0 or offset + length > self.size:
            raise ValueError("zero-source read exceeds its extent")
        return bytes(length)

    def read_into(self, offset: int, destination: memoryview) -> int:
        if offset < 0 or offset + destination.nbytes > self.size:
            raise ValueError("zero-source read exceeds its extent")
        # Fixed tiny fill block, never a second allocation proportional to a tile.
        zero = memoryview(bytes(256))
        for first in range(0, destination.nbytes, len(zero)):
            count = min(len(zero), destination.nbytes - first)
            destination[first:first + count] = zero[:count]
        return destination.nbytes


@dataclass(frozen=True, slots=True)
class SegmentedSource:
    """An immutable physical concatenation of selected byte spans, not arithmetic."""
    spans: tuple[SourceSpan, ...]
    info: SourceInfo = field(init=False)

    def __post_init__(self):
        if not self.spans:
            raise ValueError("segmented source requires at least one span")
        payload = tuple((span.source.info.fingerprint, span.offset, span.length) for span in self.spans)
        identity = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
        object.__setattr__(self, "info", SourceInfo(
            f"segments:{identity}", identity, SourceKind.COMPOSITE,
            dependencies=tuple(dict.fromkeys(span.source.info for span in self.spans)),
        ))

    @property
    def size(self) -> int:
        return sum(span.length for span in self.spans)

    def read(self, offset: int, length: int) -> bytes:
        output = bytearray(length)
        self.read_into(offset, memoryview(output))
        return bytes(output)

    def read_into(self, offset: int, destination: memoryview) -> int:
        length = destination.nbytes
        if min(offset, length) < 0 or offset + length > self.size:
            raise ValueError("segmented read exceeds its source extent")
        start = 0
        for span in self.spans:
            lo, hi = max(offset, start), min(offset + length, start + span.length)
            if lo < hi:
                count = span.source.read_into(span.offset + lo - start, destination[lo - offset:hi - offset])
                if count != hi - lo:
                    raise OSError("segmented source returned a short read")
            start += span.length
            if start >= offset + length:
                break
        return length


@dataclass(frozen=True, slots=True)
class SourcePlane:
    """Bytes for each logical group, including canonical coefficient planes."""
    span: SourceSpan
    group_elements: int
    group_bytes: int

    def region(self, first: int, count: int) -> SourcePlane:
        if first % self.group_elements or count % self.group_elements:
            raise ValueError("a source region must contain complete encoding groups")
        offset = first // self.group_elements * self.group_bytes
        length = count // self.group_elements * self.group_bytes
        if offset + length > self.span.length:
            raise ValueError("logical region exceeds its source plane")
        return replace(self, span=SourceSpan(self.span.source, self.span.offset + offset, length))


@dataclass(frozen=True, slots=True)
class DenseImport:
    source_dtype: DType
    transform: Transform = Transform.IDENTITY


@dataclass(frozen=True, slots=True)
class CanonicalImport:
    """Source planes already match the declared ops representation byte layout."""


@dataclass(frozen=True, slots=True)
class EncodedImport:
    codec: object
    block_elements: int
    block_bytes: int


type ImportRecipe = DenseImport | CanonicalImport | EncodedImport


@dataclass(frozen=True, slots=True)
class Binding:
    spec: TensorSpec
    value_identity: str
    residency: Residency = Residency.RESIDENT
    planes: tuple[SourcePlane, ...] = ()
    recipe: ImportRecipe = CanonicalImport()
    resource: Resource | None = field(default=None, compare=False, repr=False)
    root_identity: str | None = None

    def __post_init__(self):
        if not self.value_identity or not self.spec.static:
            raise ValueError("bindings require immutable value identity and concrete specification")
        if self.root_identity is None:
            object.__setattr__(self, "root_identity", self.value_identity)
        elif not self.root_identity:
            raise ValueError("binding root identity must be nonempty")
        if bool(self.planes) == (self.resource is not None):
            raise ValueError("a binding has either a source or an existing resource")
        if self.resource is not None:
            if self.resource.spec != self.spec or self.residency != Residency.RESIDENT:
                raise ValueError("an existing resource must be a resident binding with matching spec")
        for plane in self.planes:
            if plane.group_elements <= 0 or plane.group_bytes <= 0:
                raise ValueError("invalid source plane group geometry")
            if self.spec.elements % plane.group_elements:
                raise ValueError("binding ends inside a source encoding group")
            if self.spec.elements // plane.group_elements * plane.group_bytes != plane.span.length:
                raise ValueError("source plane does not describe the complete logical value")

    @classmethod
    def existing(cls, resource: Resource, *, identity: str) -> Binding:
        return cls(resource.spec, identity, resource=resource)

    @property
    def source_bytes(self):
        return sum(plane.span.length for plane in self.planes)

    @property
    def region_alignment(self):
        return math.lcm(*(plane.group_elements for plane in self.planes)) if self.planes else 1

    def region(self, first: int, count: int, *, shape: tuple[int, ...]) -> Binding:
        if self.resource is not None:
            raise ValueError("resident resource subviews are execution leases, not source regions")
        if first < 0 or count <= 0 or first + count > self.spec.elements or math.prod(shape) != count:
            raise ValueError("invalid logical binding region")
        spec = TensorSpec(shape, self.spec.dtype, representation=self.spec.representation)
        return Binding(spec, f"{self.value_identity}/elements/{first}:{count}", self.residency,
                       tuple(plane.region(first, count) for plane in self.planes), self.recipe,
                       root_identity=self.root_identity)

    @property
    def fingerprint(self):
        # No file handles, provider objects, live addresses or timing observations.
        payload = (_stable(self.spec), self.value_identity, self.root_identity, self.residency,
                   tuple((plane.span.source.info.fingerprint, plane.span.offset, plane.span.length,
                          plane.group_elements, plane.group_bytes)
                         for plane in self.planes), _stable(self.recipe))
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
