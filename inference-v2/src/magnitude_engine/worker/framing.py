"""Bounded JSON control envelopes and lossless binary buffers on private streams."""

import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

VERSION = 3
MAX_FRAME_BYTES = 48 << 20
MAX_BUFFER_BYTES = 512 << 20
MAX_BUFFERS = 16


@dataclass
class Frame:
    generation: str
    message: dict
    buffers: tuple[bytes, ...] = ()

    def close(self) -> None:
        self.buffers = ()


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate protocol field")
        result[key] = value
    return result


def _read(stream: BinaryIO, count: int) -> bytes:
    parts = bytearray()
    while len(parts) < count:
        part = stream.read(count - len(parts))
        if not part:
            raise EOFError("private worker stream ended before frame completion")
        parts.extend(part)
    return bytes(parts)


def _constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in worker frame: {value}")


def _lengths(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) > MAX_BUFFERS
        or any(type(n) is not int or not 0 < n <= MAX_BUFFER_BYTES for n in value)
        or sum(value) > MAX_BUFFER_BYTES
    ):
        raise ValueError("worker buffers exceed protocol size bound")
    return tuple(value)


@dataclass(frozen=True)
class Header:
    generation: str
    message: dict
    lengths: tuple[int, ...]

    def read(self, stream: BinaryIO) -> Frame:
        return Frame(self.generation, self.message, tuple(_read(stream, n) for n in self.lengths))

    def discard(self, stream: BinaryIO) -> None:
        """Consume the payload in bounded scratch space, preserving the next frame."""
        remaining = sum(self.lengths)
        while remaining:
            count = min(remaining, 65536)
            _read(stream, count)
            remaining -= count


def read_header(stream: BinaryIO, generation: str | None = None) -> Header:
    length = struct.unpack(">I", _read(stream, 4))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds protocol size bound")
    value = json.loads(_read(stream, length), object_pairs_hook=_object, parse_constant=_constant)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "generation", "message", "buffers"}
        or type(value["version"]) is not int
        or value["version"] != VERSION
        or not isinstance(value["generation"], str)
        or not 1 <= len(value["generation"]) <= 128
        or not isinstance(value["message"], dict)
        or (generation is not None and value["generation"] != generation)
    ):
        raise ValueError("worker frame has an incompatible version, generation or envelope")
    return Header(value["generation"], value["message"], _lengths(value["buffers"]))


def read_frame(stream: BinaryIO, generation: str | None = None) -> Frame:
    return read_header(stream, generation).read(stream)


def _write(stream: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("worker stream made no write progress")
        remaining = remaining[written:]


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    if not isinstance(frame.buffers, tuple) or any(not isinstance(b, bytes) for b in frame.buffers):
        raise ValueError("worker buffers must be immutable bytes")
    lengths = _lengths([len(buffer) for buffer in frame.buffers])
    payload = json.dumps(
        {
            "version": VERSION,
            "generation": frame.generation,
            "message": frame.message,
            "buffers": lengths,
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds protocol size bound")
    _write(stream, struct.pack(">I", len(payload)) + payload)
    for buffer in frame.buffers:
        _write(stream, buffer)
    stream.flush()
