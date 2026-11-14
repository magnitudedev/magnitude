"""The storage-resolution perturbation (`inference/docs/precision.md`, "Storage resolution supplies the
yardstick"): every stored value moves to the adjacent representable value of its stored format below, with
probability 1/24, or above, with probability 1/24, and otherwise stays. A move with no adjacent finite value in
the format's range is no move.

Values move as codes, before anything decodes them, so "adjacent" is exact: a format's codes are integers
ordered by value (`Format`), and a move adds -1 or +1 to that order. Each plane moves in its own format: a
weight's codes and its stored scales separately (MXFP4's E2M1 codes and E8M0 block scales, F8_E4M3 codes and
their UE8M0 tiles, BF16, F16 and F32 at their own step), and a family's stored rows likewise (DeepSeek V4.1's
trained-rounding KV, index-key and compressed rows, as they are formed).

The draw is a counter-based hash of (seed, plane, element index): a function of the value's identity alone, so
a value read twice, in any order or on any thread, moves the same way, and no state is carried between reads.
Indices, counts and masks are not stored values and are never perturbed: only the planes named here move.

A reference opens in it as its `perturbed:<seed>` variant (`families.open_reference`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import hashlib

import numba
import numpy as np
import torch

from gguf.constants import GGMLQuantizationType as Q

VARIANT = "perturbed:"
CHANCES = 24  # a value moves down on draw 0, up on draw 1, of 24 equally likely
MXFP4_BLOCK, MXFP4_BYTES = 32, 17  # values per block; one E8M0 byte, then 16 bytes of nibble pairs


@dataclass(frozen=True)
class Format:
    """A stored format's codes as integers ordered by value: sign-magnitude when `signed` (the top of `bits`
    is the sign; +0 and -0 are one value), unsigned otherwise. `largest` is the largest finite magnitude code:
    a code above it is not a finite value and never moves, and nothing moves past it."""
    name: str
    bits: int
    signed: bool
    largest: int


F32 = Format("F32", 32, True, 0x7F7F_FFFF)
F16 = Format("F16", 16, True, 0x7BFF)
BF16 = Format("BF16", 16, True, 0x7F7F)
E4M3 = Format("E4M3", 8, True, 0x7E)  # FN: 0x7F and 0xFF are NaN; 448 is 0x7E
E2M1 = Format("E2M1", 4, True, 0x7)  # 0, 0.5, 1, 1.5, 2, 3, 4, 6
E8M0 = Format("E8M0", 8, False, 254)  # 2^(b - 127); 255 is NaN (OCP MX)
# An FP8 weight's UE8M0 tile exponent, within the range where its products are exact in float32
# (precision.md, "Block-scaled FP8 weights"): a file holds no byte above 246.
UE8M0 = Format("UE8M0", 8, False, 246)

ELEMENTS = {Q.F32: (F32, np.uint32), Q.F16: (F16, np.uint16), Q.BF16: (BF16, np.uint16), Q.F8_E4M3: (E4M3, np.uint8)}
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def move(codes: np.ndarray, steps: np.ndarray, format: Format) -> np.ndarray:
    """`codes` (unsigned integers holding `format`'s codes) each moved `steps` (-1, 0 or +1, broadcast)
    adjacent values along the value order; a code with no adjacent finite value that way stays."""
    # the narrowest signed integers that hold every ordinal and its move
    work = np.int16 if format.bits <= 8 else np.int32 if format.bits <= 16 else np.int64
    wide = codes.astype(work)
    if format.signed:
        sign = work(1 << (format.bits - 1))
        magnitude = wide & (sign - 1)
        ordinal = np.where(wide & sign, -magnitude, magnitude)
        low = -format.largest
    else:
        magnitude = ordinal = wide
        low = 0
    moved = ordinal + steps
    stays = (steps == 0) | (magnitude > format.largest) | (moved < low) | (moved > format.largest)
    encoded = np.where(moved < 0, sign | -moved, moved) if format.signed else moved
    return np.where(stays, wide, encoded).astype(codes.dtype)


# The top-32-bit hash values whose draw, (top * CHANCES) >> 32, is 0 (below DRAW_DOWN) or 0 or 1 (below DRAW_UP).
DRAW_DOWN, DRAW_UP = (np.uint64(-(-(draw << 32) // CHANCES)) for draw in (1, 2))


@numba.njit(inline="always")
def draw(key: np.uint64, index: int) -> int:
    """The draw of element `index` of the plane whose key is `key`: 0, 1 or 2 (down, stay, up), the row index of
    the `moves` tables. SplitMix64 over the key plus the index, then the top 32 bits scaled to one of `CHANCES`."""
    x = np.uint64(index) * np.uint64(0x9E3779B97F4A7C15) + key
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    top = x >> np.uint64(32)
    return 0 if top < DRAW_DOWN else 2 if top < DRAW_UP else 1


@numba.njit(nogil=True, cache=True)
def draws_of(key: np.uint64, index: np.ndarray, out: np.ndarray) -> None:
    for i in range(index.size):
        out[i] = draw(key, index[i])


@numba.njit(nogil=True, cache=True)
def elements_moved(values: np.ndarray, rows: np.ndarray, key: np.uint64, table: np.ndarray, out: np.ndarray) -> None:
    """Codes [n, width] of GGUF rows `rows` [n], moved through `table` (`moves`) into `out`: element i of row r is
    element r * width + i of the plane."""
    width = values.shape[1]
    for r in range(values.shape[0]):
        first = rows[r] * width
        for i in range(width):
            out[r, i] = table[draw(key, first + i), values[r, i]]


@numba.njit(nogil=True, cache=True)
def mxfp4_moved(blocks: np.ndarray, rows: np.ndarray, scale_key: np.uint64, value_key: np.uint64,
                scale_moves: np.ndarray, value_moves: np.ndarray, out: np.ndarray) -> None:
    """MXFP4 blocks [n, blocks a row, 17] of GGUF rows `rows` [n], moved into `out`: block b of row r is block
    r * blocks + b of the scale plane, and its value k (gguf-py's order: low nibbles 0..15, high 16..31) element
    32 * that + k of the value plane."""
    count = blocks.shape[1]
    for r in range(blocks.shape[0]):
        for b in range(count):
            block = rows[r] * count + b
            out[r, b, 0] = scale_moves[draw(scale_key, block), blocks[r, b, 0]]
            first = block * MXFP4_BLOCK
            for k in range(MXFP4_BLOCK // 2):
                pair = blocks[r, b, 1 + k]
                low = value_moves[draw(value_key, first + k), pair & 0x0F]
                high = value_moves[draw(value_key, first + MXFP4_BLOCK // 2 + k), pair >> 4]
                out[r, b, 1 + k] = low | (high << 4)


@cache
def moves(format: Format) -> np.ndarray:
    """`move` tabulated for a format of at most 16 bits: [draw (0 down, 1 stay, 2 up), code] -> moved code."""
    codes = np.arange(1 << format.bits, dtype=np.uint8 if format.bits <= 8 else np.uint16)
    return np.stack([move(codes, np.int8(step), format) for step in (-1, 0, 1)])


@dataclass(frozen=True)
class Perturbation:
    """The perturbation of draw `seed`."""
    seed: int

    @staticmethod
    def of(variant: str | None) -> "Perturbation | None":
        """The perturbation a reference variant names (`perturbed:<seed>`, a decimal seed); None for any other."""
        if variant is None or not variant.startswith(VARIANT):
            return None
        seed = variant[len(VARIANT):]
        if not seed.isdecimal() or str(int(seed)) != seed:
            raise ValueError(f"{variant!r}: a perturbed variant is perturbed:<seed>, a decimal seed without leading zeros")
        return Perturbation(int(seed))

    def steps(self, plane: str, index: np.ndarray) -> np.ndarray:
        """The move (-1, 0 or +1, int8) of each element `index` (non-negative integers) of `plane` (`draw`)."""
        return self.draws(plane, index) - np.int8(1)

    def key(self, plane: str) -> np.uint64:
        """The draw key of `plane` under this seed: every draw of the plane is a function of it and an index."""
        return np.uint64(int.from_bytes(hashlib.blake2b(f"{self.seed}/{plane}".encode(), digest_size=8).digest(),
                                        "little"))

    def draws(self, plane: str, index: np.ndarray) -> np.ndarray:
        """`steps` plus one: 0, 1 or 2 (down, stay, up) as int8, the row index of the `moves` tables (`draw`)."""
        flat = np.ascontiguousarray(index, dtype=np.int64).reshape(-1)
        out = np.empty(flat.shape, dtype=np.int8)
        draws_of(self.key(plane), flat, out)
        return out.reshape(np.shape(index))

    def stored(self, name: str, kind: Q, width: int, data: np.ndarray, rows: np.ndarray,
               encoding: Format | None = None) -> np.ndarray:
        """GGUF rows `rows` [n] of tensor `name` (type `kind`, `width` elements a row), stored as `data`
        [n, row bytes], moved. Element i of row r is element r * width + i of the tensor's plane. An I8 tensor
        moves only as a companion scale, whose `encoding` its reader names; every other type of the file
        carries its own format, and a type without one is refused."""
        rows = np.asarray(rows, dtype=np.int64)
        data = np.ascontiguousarray(data, dtype=np.uint8).reshape(len(rows), -1)
        if kind == Q.MXFP4:
            blocks = data.reshape(len(rows), -1, MXFP4_BYTES)
            moved = np.empty_like(blocks)
            mxfp4_moved(blocks, rows, self.key(name + "/scales"), self.key(name), moves(E8M0), moves(E2M1), moved)
            return moved.reshape(data.shape)
        if kind == Q.I8 and encoding is not None:
            format, dtype = encoding, np.uint8
        elif kind in ELEMENTS and encoding is None:
            format, dtype = ELEMENTS[kind]
        else:
            raise ValueError(f"{name}: no storage-resolution perturbation of {kind.name}"
                             + (f" as {encoding.name}" if encoding is not None else ""))
        values = data.view(dtype)
        if format.bits > 16:
            return move(values, self.steps(name, rows[:, None] * width + np.arange(width)), format).view(np.uint8)
        moved = np.empty_like(values)
        elements_moved(values, rows, self.key(name), moves(format), moved)
        return moved.view(np.uint8)

    def blocks(self, plane: str, codes: torch.Tensor, code: Format, scales: torch.Tensor,
               scale: Format) -> tuple[torch.Tensor, torch.Tensor]:
        """Stored rows formed as code values `codes` [..., R, n, size] of `code` times block scales `scales`
        [..., R, n, 1] of `scale`, each plane moved. A row's identity is its index R in the store (a position
        or a compressed group), so leading dimensions (sequences of a batch) draw alike, and a row's draw does
        not depend on the rows after it."""
        rows, count, size = codes.shape[-3:]
        index = np.arange(rows * count * size, dtype=np.int64).reshape(rows, count, size)
        moved = from_codes(move(to_codes(codes, code), self.steps(plane, index), code), code, codes)
        scale_index = np.arange(rows * count, dtype=np.int64).reshape(rows, count, 1)
        moved_scales = from_codes(move(to_codes(scales, scale), self.steps(plane + "/scales", scale_index), scale),
                                  scale, scales)
        return moved, moved_scales


def to_codes(values: torch.Tensor, format: Format) -> np.ndarray:
    """The codes of `values`, each exactly a finite value of `format` (an invariant: refused otherwise)."""
    flat = values.detach().to("cpu", torch.float32)
    if format == E4M3:
        codes = flat.to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    elif format == E2M1:
        magnitude = torch.searchsorted(E2M1_VALUES, flat.abs()).clamp(max=len(E2M1_VALUES) - 1)
        codes = (torch.signbit(flat).to(torch.int64) << 3 | magnitude).to(torch.uint8).numpy()
    elif format == E8M0:
        _, exponent = torch.frexp(flat)  # 2^e = 0.5 * 2^(e + 1)
        codes = (exponent + 126).clamp(0, 255).to(torch.uint8).numpy()
    else:
        raise ValueError(f"no stored-row codes of {format.name}")
    if not torch.equal(from_codes(codes, format, flat), flat):
        raise AssertionError(f"stored values outside {format.name}'s codes")
    return codes


def from_codes(codes: np.ndarray, format: Format, like: torch.Tensor) -> torch.Tensor:
    """The float32 values of `format`'s `codes`, on `like`'s device."""
    codes = torch.from_numpy(np.ascontiguousarray(codes))
    if format == E4M3:
        values = codes.view(torch.float8_e4m3fn).float()
    elif format == E2M1:
        values = E2M1_VALUES[(codes & 7).long()] * (1 - 2 * (codes >> 3).float())
    elif format == E8M0:
        values = torch.ldexp(torch.ones(codes.shape), codes.to(torch.int32) - 127)
    else:
        raise ValueError(f"no stored-row codes of {format.name}")
    return values.to(like.device)
