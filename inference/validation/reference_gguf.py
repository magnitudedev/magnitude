#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0"]
# ///
"""GGUF access for the independent model references.

uv run inference/validation/reference_gguf.py header PATH.gguf
uv run inference/validation/reference_gguf.py header --repository ORG/NAME --revision SHA PATH-IN-REPO
uv run inference/validation/reference_gguf.py self-test

Typed headers read from a local file or from immutable Hugging Face revisions by HTTP byte range,
tensor bytes from either, exact dequantization to float32, and a streaming writer that keeps every
metadata value's GGUF type. Dequantization uses the NumPy decoders of `gguf` 0.19.0, whose
`quants.py` is identical to gguf-py at the pinned llama.cpp revision `18443257a30c` (it only lacks
one assertion); the quantized formats are defined by ggml, so this is the format's own decoder and
is independent of the engine under test.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import time
import urllib.error
import urllib.request

import numpy as np
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize as ggml_dequantize

RETRIES = 6  # transient network failures on long range reads (backoff 1, 2, 4, 8, 16 s)
ALIGNMENT_KEY = "general.alignment"
DEFAULT_ALIGNMENT = 32
SCALARS = {
    GGUFValueType.UINT8: "B", GGUFValueType.INT8: "b", GGUFValueType.UINT16: "H", GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I", GGUFValueType.INT32: "i", GGUFValueType.FLOAT32: "f", GGUFValueType.BOOL: "?",
    GGUFValueType.UINT64: "Q", GGUFValueType.INT64: "q", GGUFValueType.FLOAT64: "d",
}


class HeaderIncomplete(Exception):
    """The bytes read so far end inside the header."""


@dataclass(frozen=True)
class Value:
    """One metadata value with its GGUF type (and the element type of an array)."""
    kind: GGUFValueType
    value: object
    item: GGUFValueType | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind.name, "item": self.item.name if self.item is not None else None, "value": self.value}

    @staticmethod
    def from_json(record: dict) -> "Value":
        item = GGUFValueType[record["item"]] if record["item"] is not None else None
        return Value(GGUFValueType[record["kind"]], record["value"], item)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # GGUF order: ne[0] (innermost, row length) first
    type: GGMLQuantizationType
    offset: int  # relative to the file's data section

    @property
    def nbytes(self) -> int:
        block, size = GGML_QUANT_SIZES[self.type]
        if self.shape[0] % block:
            raise ValueError(f"{self.name}: row length {self.shape[0]} is not a multiple of {self.type.name} blocks")
        return int(np.prod(self.shape[1:], dtype=np.int64)) * self.shape[0] // block * size

    @property
    def array_shape(self) -> tuple[int, ...]:
        """NumPy (row-major) shape: the GGUF shape reversed."""
        return tuple(reversed(self.shape))


@dataclass(frozen=True)
class Header:
    metadata: dict[str, Value]
    tensors: dict[str, TensorInfo]
    data_offset: int

    def get(self, key: str):
        return self.metadata[key].value

    def to_json(self) -> dict:
        return {"metadata": {key: value.to_json() for key, value in self.metadata.items()},
                "tensors": [{"name": t.name, "shape": list(t.shape), "type": t.type.name, "offset": t.offset}
                            for t in self.tensors.values()],
                "data_offset": self.data_offset}

    @staticmethod
    def from_json(record: dict) -> "Header":
        tensors = {t["name"]: TensorInfo(t["name"], tuple(t["shape"]), GGMLQuantizationType[t["type"]], t["offset"])
                   for t in record["tensors"]}
        metadata = {key: Value.from_json(value) for key, value in record["metadata"].items()}
        return Header(metadata, tensors, record["data_offset"])


class _Cursor:
    def __init__(self, data: bytes):
        self.data, self.offset = data, 0

    def take(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise HeaderIncomplete
        chunk = self.data[self.offset:self.offset + size]
        self.offset += size
        return chunk

    def unpack(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("Q")).decode("utf-8")

    def value(self, kind: GGUFValueType) -> Value:
        if kind in SCALARS:
            return Value(kind, self.unpack(SCALARS[kind]))
        if kind == GGUFValueType.STRING:
            return Value(kind, self.string())
        if kind == GGUFValueType.ARRAY:
            item = GGUFValueType(self.unpack("I"))
            count = self.unpack("Q")
            if item in SCALARS:
                fmt = SCALARS[item]
                values = list(struct.unpack(f"<{count}{fmt}", self.take(count * struct.calcsize(fmt))))
            else:
                values = [self.value(item).value for _ in range(count)]
            return Value(kind, values, item)
        raise ValueError(f"unsupported GGUF value type {kind}")


def parse_header(data: bytes) -> Header:
    """Parse a complete header from the leading bytes of a GGUF file (HeaderIncomplete if too short)."""
    cursor = _Cursor(data)
    if cursor.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = cursor.unpack("I")
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}")
    tensor_count, metadata_count = cursor.unpack("Q"), cursor.unpack("Q")
    metadata = {}
    for _ in range(metadata_count):
        key = cursor.string()
        metadata[key] = cursor.value(GGUFValueType(cursor.unpack("I")))
    tensors = {}
    for _ in range(tensor_count):
        name = cursor.string()
        dims = cursor.unpack("I")
        shape = tuple(cursor.unpack("Q") for _ in range(dims))
        tensors[name] = TensorInfo(name, shape, GGMLQuantizationType(cursor.unpack("I")), cursor.unpack("Q"))
    alignment = metadata[ALIGNMENT_KEY].value if ALIGNMENT_KEY in metadata else DEFAULT_ALIGNMENT
    data_offset = -(-cursor.offset // alignment) * alignment
    return Header(metadata, tensors, data_offset)


class Source:
    """One GGUF file: its header and byte-range access to its tensor data."""

    header: Header

    def read(self, offset: int, size: int) -> bytes:
        raise NotImplementedError

    def tensor_bytes(self, name: str) -> np.ndarray:
        info = self.header.tensors[name]
        return np.frombuffer(self.read(self.header.data_offset + info.offset, info.nbytes), dtype=np.uint8)


class LocalSource(Source):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.map = np.memmap(self.path, dtype=np.uint8, mode="r")
        size = 1 << 20
        while True:
            try:
                self.header = parse_header(bytes(self.map[:size]))
                break
            except HeaderIncomplete:
                if size >= self.map.size:
                    raise
                size *= 4

    def read(self, offset: int, size: int) -> bytes:
        return self.map[offset:offset + size]

    def tensor_bytes(self, name: str) -> np.ndarray:
        info = self.header.tensors[name]
        start = self.header.data_offset + info.offset
        return self.map[start:start + info.nbytes]

    def describe(self) -> str:
        return str(self.path)


def hub_url(repository: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repository}/resolve/{revision}/{path}"


class RemoteSource(Source):
    """A file at an immutable Hugging Face revision, read by HTTP byte range."""

    def __init__(self, repository: str, revision: str, path: str, cache: Path | None = None):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"remote reads need an immutable 40-hex revision, got {revision!r}")
        self.repository, self.revision, self.path = repository, revision, path
        self.url = hub_url(repository, revision, path)
        cached = cache / f"{repository.replace('/', '__')}__{revision}__{path.replace('/', '__')}.json" if cache else None
        if cached is not None and cached.is_file():
            self.header = Header.from_json(json.loads(cached.read_text()))
            return
        size = 1 << 20
        while True:
            data = self.read(0, size, exact=False)
            try:
                self.header = parse_header(data)
                break
            except HeaderIncomplete:
                if len(data) < size:
                    raise
                size *= 4
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(self.header.to_json()))

    def read(self, offset: int, size: int, exact: bool = True) -> bytes:
        """Bytes [offset, offset+size); with exact=False a read past the end of the file is cut short."""
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={offset}-{offset + size - 1}"})
        for attempt in range(RETRIES):
            try:
                chunks = []
                with urllib.request.urlopen(request, timeout=600) as response:
                    if response.status != 206:
                        raise RuntimeError(f"{self.url}: expected a partial response, got HTTP {response.status}")
                    while block := response.read(1 << 24):
                        chunks.append(block)
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        data = b"".join(chunks)
        if exact and len(data) != size:
            raise RuntimeError(f"{self.url}: short range read ({len(data)} of {size} bytes)")
        return data

    def describe(self) -> str:
        return f"{self.repository}@{self.revision}/{self.path}"


SHARD = re.compile(r"^(?P<prefix>.*)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.gguf)$")


def shard_paths(path: str) -> list[str]:
    """Every shard path of a split GGUF given any one of them (a single file is its own list)."""
    match = SHARD.match(path)
    if match is None:
        return [path]
    count = int(match["count"])
    return [f"{match['prefix']}-{index:05d}-of-{count:05d}{match['suffix']}" for index in range(1, count + 1)]


class Package:
    """A (possibly split) GGUF model: the first shard's metadata and every shard's tensors."""

    def __init__(self, sources: list[Source]):
        self.sources = sources
        self.metadata = sources[0].header.metadata
        self.location: dict[str, Source] = {}
        for source in sources:
            for name in source.header.tensors:
                if name in self.location:
                    raise ValueError(f"tensor {name} appears in more than one shard")
                self.location[name] = source

    @staticmethod
    def local(path: Path) -> "Package":
        path = Path(path)
        return Package([LocalSource(path.with_name(name)) for name in shard_paths(path.name)])

    @staticmethod
    def remote(repository: str, revision: str, path: str, cache: Path | None = None) -> "Package":
        return Package([RemoteSource(repository, revision, shard, cache) for shard in shard_paths(path)])

    @property
    def architecture(self) -> str:
        return self.metadata["general.architecture"].value

    def key(self, suffix: str, default=None):
        """Metadata `<architecture>.<suffix>`; `default` when absent (None means the key is required)."""
        key = f"{self.architecture}.{suffix}"
        if key in self.metadata:
            return self.metadata[key].value
        if default is None:
            raise KeyError(f"missing metadata {key}")
        return default

    def has(self, name: str) -> bool:
        return name in self.location

    def info(self, name: str) -> TensorInfo:
        return self.location[name].header.tensors[name]

    def raw(self, name: str) -> np.ndarray:
        return self.location[name].tensor_bytes(name)

    def tensor(self, name: str) -> np.ndarray:
        """Exact float32 values of a tensor, in NumPy order (the GGUF shape reversed)."""
        return dequantize(self.info(name), self.raw(name))

    def names(self) -> list[str]:
        return list(self.location)


def dequantize(info: TensorInfo, raw: np.ndarray) -> np.ndarray:
    if len(raw) != info.nbytes:
        raise ValueError(f"{info.name}: {len(raw)} bytes, expected {info.nbytes}")
    block, size = GGML_QUANT_SIZES[info.type]
    rows = np.asarray(raw, dtype=np.uint8).reshape(-1, info.shape[0] // block * size)
    values = ggml_dequantize(rows, info.type)
    # F32 rows come back as a read-only view of the file; the references need owned arrays.
    return np.require(np.asarray(values, dtype=np.float32).reshape(info.array_shape), requirements="W")


def quantize(values: np.ndarray, kind: GGMLQuantizationType) -> np.ndarray:
    """Encode float32 values (NumPy order) to GGUF bytes shaped [rows, row bytes]."""
    from gguf.quants import quantize as ggml_quantize
    values = np.ascontiguousarray(values, dtype=np.float32)
    if kind == GGMLQuantizationType.F32:
        encoded = values.view(np.uint8)
    elif kind == GGMLQuantizationType.F16:
        encoded = values.astype(np.float16).view(np.uint8)
    else:
        encoded = ggml_quantize(values, kind)
    return encoded.reshape(-1, encoded.shape[-1])


def _string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _value(kind: GGUFValueType, value, item: GGUFValueType | None) -> bytes:
    if kind in SCALARS:
        return struct.pack("<" + SCALARS[kind], value)
    if kind == GGUFValueType.STRING:
        return _string(value)
    if kind == GGUFValueType.ARRAY:
        head = struct.pack("<IQ", item, len(value))
        if item in SCALARS:
            return head + struct.pack(f"<{len(value)}{SCALARS[item]}", *value)
        return head + b"".join(_value(item, element, None) for element in value)
    raise ValueError(f"unsupported GGUF value type {kind}")


class Writer:
    """Streaming GGUF v3 writer, the inverse of `parse_header`: declare the tensor directory, then write
    each tensor's bytes in declaration order (so arbitrarily large tensors never sit in memory together)."""

    def __init__(self, path: Path, metadata: dict[str, Value]):
        self.path, self.metadata = Path(path), metadata
        self.alignment = metadata[ALIGNMENT_KEY].value if ALIGNMENT_KEY in metadata else DEFAULT_ALIGNMENT
        self.order: list[TensorInfo] = []
        self.offset = 0

    def declare(self, name: str, shape: tuple[int, ...], kind: GGMLQuantizationType) -> None:
        """`shape` in GGUF order (ne[0] first)."""
        info = TensorInfo(name, tuple(shape), kind, self.offset)
        self.order.append(info)
        self.offset += -(-info.nbytes // self.alignment) * self.alignment

    def begin(self) -> None:
        header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(self.order), len(self.metadata)))
        for key, value in self.metadata.items():
            header += _string(key) + struct.pack("<I", value.kind) + _value(value.kind, value.value, value.item)
        for info in self.order:
            header += _string(info.name) + struct.pack(f"<I{len(info.shape)}QIQ", len(info.shape), *info.shape,
                                                       info.type, info.offset)
        header += bytes(-len(header) % self.alignment)
        self.stream = self.path.open("wb")
        self.stream.write(header)
        self.data_start, self.next = len(header), 0

    def write(self, name: str, data: np.ndarray | bytes) -> None:
        info = self.order[self.next]
        data = bytes(data) if isinstance(data, (bytes, bytearray)) else np.ascontiguousarray(data).view(np.uint8).tobytes()
        if name != info.name or len(data) != info.nbytes:
            raise ValueError(f"write order: got {name} ({len(data)} B), expected {info.name} ({info.nbytes} B)")
        self.stream.seek(self.data_start + info.offset)
        self.stream.write(data)
        self.next += 1

    def close(self) -> None:
        if self.next != len(self.order):
            raise ValueError(f"only {self.next} of {len(self.order)} tensors written")
        self.stream.truncate(self.data_start + self.offset)
        self.stream.close()


def metadata_json(metadata: dict[str, Value]) -> dict:
    """Compact JSON view of metadata (long arrays truncated), for provenance records."""
    view = {}
    for key, value in metadata.items():
        if value.kind == GGUFValueType.ARRAY and len(value.value) > 64:
            view[key] = {"length": len(value.value), "head": value.value[:8]}
        else:
            view[key] = value.value
    return view


HEADER_CACHE = Path(__file__).resolve().parent / "results" / "headers"


def self_test() -> None:
    """Writer/parser round trip over every value type and several tensor encodings."""
    import tempfile
    rng = np.random.default_rng(0)
    metadata = {
        "general.architecture": Value(GGUFValueType.STRING, "synthetic"),
        "synthetic.count": Value(GGUFValueType.UINT32, 7), "synthetic.offset": Value(GGUFValueType.INT64, -3),
        "synthetic.scale": Value(GGUFValueType.FLOAT32, 0.5), "synthetic.flag": Value(GGUFValueType.BOOL, True),
        "synthetic.ids": Value(GGUFValueType.ARRAY, [1, 2, 3], GGUFValueType.INT32),
        "synthetic.words": Value(GGUFValueType.ARRAY, ["a", "", "ç"], GGUFValueType.STRING),
        "synthetic.none": Value(GGUFValueType.ARRAY, [], GGUFValueType.STRING),
    }
    kinds = [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16,
             GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0]
    tensors = {f"t.{kind.name}": rng.standard_normal((3, 64)).astype(np.float32) for kind in kinds}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.gguf"
        writer = Writer(path, metadata)
        encoded = {name: quantize(values, kind) for (name, values), kind in zip(tensors.items(), kinds)}
        for (name, values), kind in zip(tensors.items(), kinds):
            writer.declare(name, tuple(reversed(values.shape)), kind)
        writer.begin()
        for name, data in encoded.items():
            writer.write(name, data)
        writer.close()
        package = Package.local(path)
        assert package.metadata == metadata, "metadata round trip"
        for (name, values), kind in zip(tensors.items(), kinds):
            decoded = package.tensor(name)
            expected = dequantize(TensorInfo(name, (64, 3), kind, 0), encoded[name].reshape(-1))
            assert np.array_equal(decoded, expected), name
            tolerance = {"F32": 0, "F16": 1e-3, "BF16": 1e-2, "Q8_0": 2e-2, "Q4_0": 0.3}[kind.name]
            assert np.abs(decoded - values).max() <= tolerance * np.abs(values).max(), name
    print("reference_gguf self-test: ok")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    header = commands.add_parser("header", help="print a package's metadata and tensor directory")
    header.add_argument("path")
    header.add_argument("--repository")
    header.add_argument("--revision")
    commands.add_parser("self-test", help="writer/parser round trip")
    options = parser.parse_args()
    if options.command == "self-test":
        self_test()
        return
    if options.repository:
        package = Package.remote(options.repository, options.revision, options.path, HEADER_CACHE)
    else:
        package = Package.local(Path(options.path))
    print(json.dumps({"metadata": metadata_json(package.metadata),
                      "tensors": {name: [list(package.info(name).shape), package.info(name).type.name]
                                  for name in package.names()}}, indent=1))


if __name__ == "__main__":
    main()
