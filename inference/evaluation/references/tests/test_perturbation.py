"""The storage-resolution perturbation's draw and its moves: each value moves down and up with probability
1/24 each, to the adjacent value of its own format, and not past the format's range."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as Q

from magnitude_references import perturbation
from magnitude_references.perturbation import (BF16, E2M1, E4M3, E8M0, F16, F32, UE8M0, Format, Perturbation, move,
                                               to_codes)

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def values_of(format: Format) -> tuple[np.ndarray, np.ndarray]:
    """Every code of a format (up to 16 bits) and its value, NaN where the code is no finite value in range."""
    count = 1 << format.bits
    codes = np.arange(count, dtype=np.uint16 if format.bits > 8 else np.uint8)
    if format == E2M1:
        values = E2M1_MAGNITUDES[codes & 7] * np.where(codes >> 3, -1.0, 1.0)
    elif format == E4M3:
        values = torch.from_numpy(codes).view(torch.float8_e4m3fn).double().numpy()
    elif format in (E8M0, UE8M0):
        values = np.where(codes <= format.largest, np.ldexp(1.0, codes.astype(np.int32) - 127), np.nan)
    elif format == BF16:
        values = torch.from_numpy(codes.view(np.int16)).view(torch.bfloat16).double().numpy()
    else:
        values = codes.view(np.float16).astype(np.float64)
    return codes, np.where(np.isfinite(values), values, np.nan)


@pytest.mark.parametrize("format", [E2M1, E4M3, E8M0, UE8M0, BF16, F16], ids=lambda f: f.name)
def test_a_move_is_to_the_adjacent_value_and_stays_at_the_range_edges(format: Format) -> None:
    codes, values = values_of(format)
    finite = np.unique(values[~np.isnan(values)])
    for step, side in ((1, "right"), (-1, "left")):
        moved = move(codes, np.full(codes.shape, step), format)
        at = np.searchsorted(finite, values, side=side) + (0 if step == 1 else -1)
        edge = np.isnan(values) | (at < 0) | (at >= len(finite))
        assert np.array_equal(moved[edge], codes[edge]), f"{format.name}: a move past the range or from a non-value"
        inside = ~edge
        assert np.array_equal(values[moved[inside]], finite[at[inside]]), f"{format.name}: not the adjacent value"
        assert not np.isnan(values[codes[edge]]).all()  # a range edge with a value stays
    assert np.array_equal(move(codes, np.zeros(codes.shape, dtype=np.int64), format), codes)


def test_float32_moves_are_nextafter() -> None:
    samples = np.array([0.0, -0.0, 1.0, -1.0, 1.5, 2.0 ** -149, -(2.0 ** -149), 2.0 ** -126, 3.4e38, -3.4e38,
                        np.finfo(np.float32).max, -np.finfo(np.float32).max, np.inf, -np.inf, np.nan], dtype=np.float32)
    codes = samples.view(np.uint32)
    for step, toward in ((1, np.inf), (-1, -np.inf)):
        moved = move(codes, np.full(codes.shape, step), F32).view(np.float32)
        finite = np.isfinite(samples)
        expected = np.nextafter(samples[finite], np.float32(toward))
        expected = np.where(np.isfinite(expected), expected, samples[finite])  # the largest stays
        assert np.array_equal(moved[finite], expected)
        assert np.array_equal(moved[~finite].view(np.uint32), codes[~finite])  # not values: they stay


def test_the_draw_moves_down_and_up_one_in_24_each() -> None:
    count = 1 << 22
    steps = Perturbation(5).steps("blk.0.attn_q_a.weight", np.arange(count))
    probability = 1 / perturbation.CHANCES
    spread = 6 * np.sqrt(count * probability * (1 - probability))
    for step in (-1, 1):
        assert abs(np.count_nonzero(steps == step) - count * probability) < spread, step
    assert set(np.unique(steps).tolist()) == {-1, 0, 1}


def test_the_draw_is_a_function_of_seed_plane_and_index() -> None:
    index = np.arange(1 << 16)
    draw = Perturbation(5).steps("x", index)
    assert np.array_equal(Perturbation(5).steps("x", index[::-1])[::-1], draw)  # any order, the same draws
    assert np.array_equal(Perturbation(5).steps("x", index[1000:2000]), draw[1000:2000])
    assert not np.array_equal(Perturbation(6).steps("x", index), draw)
    assert not np.array_equal(Perturbation(5).steps("x/scales", index), draw)


def test_the_draw_is_splitmix64_of_the_plane_key_plus_the_index() -> None:
    """The draw, written out in Python integers: SplitMix64's finalizer over key + golden * index, then the top 32
    bits scaled to one of `CHANCES` (0 down, 1 up)."""
    mask = (1 << 64) - 1

    def step(key: int, index: int) -> int:
        x = (index * 0x9E3779B97F4A7C15 + key) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        draw = (((x ^ (x >> 31)) >> 32) * perturbation.CHANCES) >> 32
        return -1 if draw == 0 else 1 if draw == 1 else 0

    seed = Perturbation(3)
    index = np.random.default_rng(2).integers(0, 1 << 40, 1 << 12)
    expected = [step(int(seed.key("blk.4.ffn_gate_exps.weight")), int(i)) for i in index]
    assert seed.steps("blk.4.ffn_gate_exps.weight", index).tolist() == expected


def test_the_variant_names_its_seed() -> None:
    assert Perturbation.of("perturbed:17") == Perturbation(17)
    assert Perturbation.of(None) is None and Perturbation.of("activations") is None
    for malformed in ("perturbed:", "perturbed:x", "perturbed:-1", "perturbed:07"):
        with pytest.raises(ValueError, match="decimal seed"):
            Perturbation.of(malformed)


def test_an_integer_plane_moves_only_as_a_named_scale() -> None:
    data = np.arange(32, dtype=np.uint8).reshape(1, 32)
    with pytest.raises(ValueError, match="no storage-resolution perturbation of I8"):
        Perturbation(1).stored("table.index", Q.I8, 32, data, np.arange(1))
    with pytest.raises(ValueError, match="no storage-resolution perturbation of Q4_K"):
        Perturbation(1).stored("x", Q.Q4_K, 256, np.zeros((1, 144), dtype=np.uint8), np.arange(1))
    moved = Perturbation(1).stored("x.scale", Q.I8, 32, data, np.arange(1), UE8M0)
    assert np.abs(moved.astype(np.int64) - data).max() <= 1


def test_stored_row_codes_round_trip_and_refuse_other_values() -> None:
    grid = torch.tensor([0.0, -0.0, 0.5, -6.0, 3.0])
    assert torch.equal(perturbation.from_codes(to_codes(grid, E2M1), E2M1, grid), grid)
    with pytest.raises(AssertionError, match="outside E2M1"):
        to_codes(torch.tensor([2.5]), E2M1)
    scales = torch.ldexp(torch.ones(4), torch.tensor([-127, -1, 0, 127]))
    assert to_codes(scales, E8M0).tolist() == [0, 126, 127, 254]
    with pytest.raises(AssertionError, match="outside E8M0"):
        to_codes(torch.tensor([3.0]), E8M0)


def test_tabulated_moves_equal_the_definition() -> None:
    """`stored` evaluates moves through tables (`moves`, and MXFP4's compiled blocks); each equals `move` with
    `steps`, the definition: every code of every format of at most 16 bits under every draw, and MXFP4 blocks nibble
    by nibble."""
    for format in (BF16, F16, E4M3, E2M1, E8M0, UE8M0):
        codes = np.arange(1 << format.bits, dtype=np.uint8 if format.bits <= 8 else np.uint16)
        for draw, step in enumerate((-1, 0, 1)):
            assert np.array_equal(perturbation.moves(format)[draw], move(codes, np.int8(step), format)), format
    rows, blocks = np.array([3, 4, 9, 1000]), 16
    data = np.random.default_rng(8).integers(0, 256, (len(rows), blocks * 17), dtype=np.uint8)
    moved = Perturbation(4).stored("blk.1.ffn_down_exps.weight", Q.MXFP4, blocks * 32, data, rows)
    block = rows[:, None] * blocks + np.arange(blocks)
    raw = data.reshape(len(rows), blocks, 17)
    scales = move(raw[..., 0], Perturbation(4).steps("blk.1.ffn_down_exps.weight/scales", block), E8M0)
    nibbles = np.concatenate((raw[..., 1:] & np.uint8(0x0F), raw[..., 1:] >> np.uint8(4)), axis=-1)
    nibbles = move(nibbles, Perturbation(4).steps("blk.1.ffn_down_exps.weight", block[..., None] * 32 + np.arange(32)),
                   E2M1)
    expected = np.concatenate((scales[..., None], nibbles[..., :16] | (nibbles[..., 16:] << np.uint8(4))), axis=-1)
    assert np.array_equal(moved, expected.reshape(data.shape))
