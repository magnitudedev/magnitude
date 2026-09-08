"""Bounded JSON control envelopes and lossless binary buffers on private streams."""

import json
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import BinaryIO

from magnitude_engine.resources.budget import Reservation

VERSION = 3
MAX_FRAME_BYTES = 48 << 20
MAX_BUFFER_BYTES = 512 << 20
MAX_BUFFERS = 16


@dataclass
class Frame:
    generation: str
    message: dict
    buffers: tuple[bytes, ...] = ()
    lease: Reservation | None = field(default=None, compare=False, repr=False)
    error: MemoryError | None = field(default=None, compare=False, repr=False)

    def close(self) -> None:
        self.buffers = ()
        if self.lease is not None:
            self.lease.close()
            self.lease = None


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


def read_frame(
    stream: BinaryIO,
    generation: str | None = None,
    *,
    reserve: Callable[[dict, int], Reservation] | None = None,
) -> Frame:
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
    lengths = _lengths(value["buffers"])
    frame = Frame(value["generation"], value["message"])
    try:
        if lengths and reserve is not None:
            try:
                frame.lease = reserve(frame.message, sum(lengths))
            except MemoryError as error:
                # Rejection consumes bounded scratch, preserving framing for control
                # and later requests without materializing an unadmitted payload.
                frame.error = error
                remaining = sum(lengths)
                while remaining:
                    count = min(remaining, 65536)
                    _read(stream, count)
                    remaining -= count
                return frame
        frame.buffers = tuple(_read(stream, count) for count in lengths)
        return frame
    except BaseException:
        frame.close()
        raise


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
