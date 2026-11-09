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
