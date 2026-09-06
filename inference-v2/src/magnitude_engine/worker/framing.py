"""Bounded, generation-scoped JSON frames over private binary streams."""

import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

VERSION = 1
MAX_FRAME_BYTES = 48 << 20


@dataclass(frozen=True)
class Frame:
    generation: str
    message: dict


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


def read_frame(stream: BinaryIO, generation: str | None = None) -> Frame:
    length = struct.unpack(">I", _read(stream, 4))[0]
    if not 0 < length <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds protocol size bound")
    value = json.loads(_read(stream, length), object_pairs_hook=_object, parse_constant=_constant)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "generation", "message"}
        or type(value["version"]) is not int
        or value["version"] != VERSION
        or not isinstance(value["generation"], str)
        or not 1 <= len(value["generation"]) <= 128
        or not isinstance(value["message"], dict)
        or (generation is not None and value["generation"] != generation)
    ):
        raise ValueError("worker frame has an incompatible version, generation or envelope")
    return Frame(value["generation"], value["message"])


def write_frame(stream: BinaryIO, frame: Frame) -> None:
    payload = json.dumps(
        {"version": VERSION, "generation": frame.generation, "message": frame.message},
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if not 0 < len(payload) <= MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds protocol size bound")
    remaining = memoryview(struct.pack(">I", len(payload)) + payload)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("worker stream made no write progress")
        remaining = remaining[written:]
    stream.flush()
