"""Shared affine arithmetic and bounded row-reuse schedules."""

from magnitude_engine.kernels.core import Source

TILE = Source("contractions/tile.metal", (Source("contractions/encoded.metal"),))


def row_tile(rows: int, maximum: int) -> int:
    # Minimize repeated weight tiles, then distribute tails evenly.
    tiles = (rows + maximum - 1) // maximum
    return (rows + tiles - 1) // tiles


def packing(width: int, outputs: int, bits: int, group_size: int) -> int | None:
    if bits not in (4, 8):
        return None
    pack = (64 if outputs % 8 == 0 else 32) // bits
    if width % (32 * pack) or group_size % pack or width % group_size:
        return None
    return pack
