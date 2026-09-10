"""GGUF directory parsing; no tensor arithmetic or filesystem access.

Encoding geometry follows ggml block layouts. A directory is validated before
any tensor storage is uploaded. Logical shapes use outermost-first ordering.
"""

from __future__ import annotations

import math
import struct
from enum import IntEnum, StrEnum

from pydantic import Field

from magnitude_engine.data import Record
from magnitude_engine.platform.storage import ByteSource


class Encoding(IntEnum):
    F32 = 0
    F16 = 1
    Q8_0 = 8
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    IQ4_XS = 23

    @property
    def block_elements(self) -> int:
        return 1 if self in (Encoding.F32, Encoding.F16) else 32 if self == Encoding.Q8_0 else 256

    @property
    def block_bytes(self) -> int:
        return {
            Encoding.F32: 4,
            Encoding.F16: 2,
            Encoding.Q8_0: 34,
            Encoding.Q4_K: 144,
            Encoding.Q5_K: 176,
            Encoding.Q6_K: 210,
            Encoding.IQ4_XS: 136,
        }[self]


class ByteOrder(StrEnum):
    LITTLE = "little"
    BIG = "big"


type Scalar = str | bool | int | float


class Metadata(Record):
    name: str
    value: Scalar | tuple[Scalar, ...]


class Tensor(Record):
    name: str
    shape: tuple[int, ...]
    encoding: Encoding
    offset: int = Field(ge=0)
    nbytes: int = Field(gt=0)


class Directory(Record):
    version: int
    byte_order: ByteOrder
    alignment: int
    data_offset: int
    metadata: tuple[Metadata, ...]
    tensors: tuple[Tensor, ...]

    def tensor(self, name: str) -> Tensor:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(f"GGUF tensor {name!r} not found")

    def value(self, name: str) -> Scalar | tuple[Scalar, ...]:
        for entry in self.metadata:
            if entry.name == name:
                return entry.value
        raise KeyError(f"GGUF metadata {name!r} not found")


class InvalidGGUF(ValueError):
    """Malformed container or a representation this reader cannot interpret."""


class _Reader:
    def __init__(self, source: ByteSource, header_limit: int):
        self.source = source
        self.offset = 0
        self.end = min(source.size, header_limit)
        self.order = "<"

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > self.end:
            raise InvalidGGUF(f"truncated or oversized GGUF header at byte {self.offset}")
        result = self.source.read(self.offset, size)
        if len(result) != size:
            raise InvalidGGUF("short source read")
        self.offset += size
        return result

    def integer(self, code: str) -> int:
        return int(struct.unpack(self.order + code, self.take(struct.calcsize(code)))[0])

    def string(self) -> str:
        try:
            return self.take(self.integer("Q")).decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidGGUF("invalid UTF-8 in GGUF header") from error

    def scalar(self, kind: int) -> Scalar:
        if kind == 8:
            return self.string()
        if kind == 7:
            value = self.integer("B")
            if value not in (0, 1):
                raise InvalidGGUF("invalid GGUF boolean")
            return bool(value)
        codes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}
        if kind not in codes:
            raise InvalidGGUF(f"unsupported GGUF metadata type {kind}")
        code = codes[kind]
        value = struct.unpack(self.order + code, self.take(struct.calcsize(code)))[0]
        return float(value) if kind in (6, 12) else int(value)

    def value(self) -> Scalar | tuple[Scalar, ...]:
        kind = self.integer("I")
        if kind != 9:
            return self.scalar(kind)
        element_type, count = self.integer("I"), self.integer("Q")
        if element_type == 9 or count > self.end - self.offset:
            raise InvalidGGUF("nested or oversized GGUF array")
        return tuple(self.scalar(element_type) for _ in range(count))


def read_directory(source: ByteSource, *, header_limit: int = 256 * 1024 * 1024) -> Directory:
    reader = _Reader(source, header_limit)
    if reader.take(4) != b"GGUF":
        raise InvalidGGUF("not a GGUF container")
    raw_version = reader.take(4)
    if raw_version == b"\x00\x00\x00\x03":
        reader.order = ">"
    version = struct.unpack(reader.order + "I", raw_version)[0]
    if version not in (2, 3):
        raise InvalidGGUF(f"unsupported GGUF version {version}")
    tensor_count, metadata_count = reader.integer("Q"), reader.integer("Q")
    if tensor_count + metadata_count > (reader.end - reader.offset) // 12:
        raise InvalidGGUF("GGUF entry counts exceed header bounds")
    metadata: list[Metadata] = []
    metadata_names: set[str] = set()
    alignment = 32
    for _ in range(metadata_count):
        name, value = reader.string(), reader.value()
        if name in metadata_names:
            raise InvalidGGUF(f"duplicate metadata {name!r}")
        metadata_names.add(name)
        metadata.append(Metadata(name=name, value=value))
        if name == "general.alignment":
            if type(value) is not int or value <= 0 or value & (value - 1):
                raise InvalidGGUF("alignment must be a positive power of two")
            alignment = value
    tensors: list[Tensor] = []
    names: set[str] = set()
    for _ in range(tensor_count):
        name, rank = reader.string(), reader.integer("I")
        if name in names or not name or not 1 <= rank <= 4:
            raise InvalidGGUF(f"invalid or duplicate tensor directory entry {name!r}")
        names.add(name)
        dims = tuple(reader.integer("Q") for _ in range(rank))
        try:
            encoding = Encoding(reader.integer("I"))
        except ValueError as error:
            raise InvalidGGUF(f"unsupported encoding on tensor {name!r}: {error}") from error
        offset = reader.integer("Q")
        if any(dim == 0 for dim in dims) or dims[0] % encoding.block_elements:
            raise InvalidGGUF(f"invalid block geometry on tensor {name!r}")
        if offset % alignment:
            raise InvalidGGUF(f"misaligned tensor {name!r}")
        tensors.append(
            Tensor(
                name=name,
                shape=tuple(reversed(dims)),
                encoding=encoding,
                offset=offset,
                nbytes=math.prod(dims) // encoding.block_elements * encoding.block_bytes,
            )
        )
    data_offset = (reader.offset + alignment - 1) // alignment * alignment
    end = data_offset
    for tensor in sorted(tensors, key=lambda tensor: tensor.offset):
        start = data_offset + tensor.offset
        if start < end or start + tensor.nbytes > source.size:
            raise InvalidGGUF(f"overlapping or truncated tensor {tensor.name!r}")
        end = start + tensor.nbytes
    return Directory(
        version=version,
        byte_order=ByteOrder.LITTLE if reader.order == "<" else ByteOrder.BIG,
        alignment=alignment,
        data_offset=data_offset,
        metadata=tuple(metadata),
        tensors=tuple(tensors),
    )
