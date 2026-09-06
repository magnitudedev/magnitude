"""Physical tensor regions and validated, header-only safetensors catalogs."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


@dataclass(frozen=True)
class FileSlice:
    path: Path
    offset: int
    size: int

    def __post_init__(self) -> None:
        if self.offset < 0 or self.size < 0:
            raise ValueError("file ranges cannot be negative")

    def slice(self, offset: int, size: int) -> FileSlice:
        if min(offset, size) < 0 or offset + size > self.size:
            raise ValueError("slice exceeds tensor storage")
        return FileSlice(self.path, self.offset + offset, size)


@dataclass(frozen=True)
class TensorRegion:
    name: str
    shape: tuple[int, ...]
    dtype: str
    storage: FileSlice

    def __post_init__(self) -> None:
        if (
            any(type(n) is not int or n < 0 for n in self.shape)
            or self.dtype not in DTYPE_BYTES
            or math.prod(self.shape) * DTYPE_BYTES[self.dtype] != self.storage.size
        ):
            raise ValueError(f"invalid physical tensor geometry: {self.name}")

    def row(self, index: int) -> FileSlice:
        if not self.shape or not 0 <= index < self.shape[0]:
            raise IndexError("tensor row out of bounds")
        width = self.storage.size // self.shape[0]
        return self.storage.slice(index * width, width)


@dataclass(frozen=True)
class TensorCatalog:
    tensors: dict[str, TensorRegion]
    metadata: dict[Path, dict[str, str]]

    @classmethod
    def inspect(cls, root: Path) -> TensorCatalog:
        root = root.expanduser().absolute()
        index = root / "model.safetensors.index.json"
        weight_map = read_json(index).get("weight_map") if index.exists() else None
        if weight_map is not None and (
            not isinstance(weight_map, dict)
            or any(not isinstance(v, str) for v in weight_map.values())
        ):
            raise ValueError("invalid safetensors index")
        names = (
            sorted(set(weight_map.values()))
            if weight_map is not None
            else sorted(
                p.name for p in root.glob("*.safetensors") if p.name != "consolidated.safetensors"
            )
        )
        if not names:
            raise ValueError("artifact has no safetensors shards")
        tensors, metadata = {}, {}
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("shard index must use paths within the artifact")
            path = root / relative
            records, information = inspect_shard(path)
            for record in records:
                if record.name in tensors:
                    raise ValueError(f"duplicate tensor across shards: {record.name}")
                if weight_map is not None and weight_map.get(record.name) != name:
                    raise ValueError("shard and weight index disagree")
                tensors[record.name] = record
            metadata[path] = information
        if weight_map is not None and set(weight_map) != set(tensors):
            raise ValueError("weight index references missing tensor data")
        return cls(tensors, metadata)


def inspect_shard(path: Path) -> tuple[tuple[TensorRegion, ...], dict[str, str]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100_000_000:
            raise ValueError("invalid safetensors header length")
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError("truncated safetensors header")
    header = json.loads(encoded, object_pairs_hook=_unique)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    metadata = header.pop("__metadata__", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("safetensors metadata must contain strings")
    result = []
    for name, description in header.items():
        if not isinstance(description, dict):
            raise ValueError("invalid tensor description")
        offsets = description.get("data_offsets")
        shape = description.get("shape")
        dtype = description.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError("tensor dtype must be a string")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or not isinstance(shape, list)
        ):
            raise ValueError("invalid tensor bounds")
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError("negative or reversed tensor bounds")
        result.append(
            TensorRegion(
                name,
                tuple(shape),
                dtype,
                FileSlice(path, 8 + length + start, end - start),
            )
        )
    position = 8 + length
    for record in sorted(result, key=lambda r: (r.storage.offset, r.storage.size)):
        if record.storage.offset != position:
            raise ValueError("safetensors payload has a gap or overlap")
        position += record.storage.size
    if path.stat().st_size != position:
        raise ValueError("safetensors payload is truncated or has trailing data")
    return tuple(result), metadata
