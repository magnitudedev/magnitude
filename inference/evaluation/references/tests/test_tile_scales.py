"""F8_E4M3 weights under tile grids (`Weights.tile_scale`): each grid's element code is named by its own GGUF type,
I8 UE8M0 bytes or F32 scales, so one file holds grids of both, and no file-wide key is read."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

from magnitude_references.gguf import Package, Value, Writer
from magnitude_references.model import Weights
from magnitude_references.perturbation import Perturbation

CPU = torch.device("cpu")
# name: (E4M3 codes' NumPy shape, grid's NumPy shape, grid type)
GRIDDED = {
    "blk.0.attn_q_b": ((64, 256), (2, 8), Q.I8),  # 32x32 UE8M0 tiles, DeepSeek V4.1's dense form
    "blk.0.ffn_down": ((256, 256), (2, 2), Q.F32),  # 128x128 F32 tiles, GLM-5.3-Flash's
    "blk.0.ffn_up_exps": ((3, 256, 128), (3, 2, 1), Q.F32),  # a stack of experts, one grid per expert
}


def write(path: Path, grids: dict[str, np.ndarray] | None = None, seed: int = 0) -> None:
    """A file of `GRIDDED`'s weights: random finite E4M3 codes and their grids (UE8M0 bytes 100..140, F32 scales
    of amax / 448's kind, both signs), or the grids `grids` names."""
    rng = np.random.default_rng(seed)
    tensors: list[tuple[str, np.ndarray, Q]] = []
    for name, (shape, grid, kind) in GRIDDED.items():
        codes = rng.integers(0, 256, shape, dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] &= 0xBF  # no NaN codes
        if kind == Q.I8:
            scales = rng.integers(100, 141, grid, dtype=np.uint8).view(np.int8)
        else:
            scales = (rng.uniform(1e-4, 1e-1, grid) * rng.choice([-1, 1], grid)).astype(np.float32)
        tensors += [(name + ".weight", codes, Q.F8_E4M3), (name + ".scale", (grids or {}).get(name, scales), kind)]
    writer = Writer(path, {"general.architecture": Value(V.STRING, "test")})
    for name, data, kind in tensors:
        writer.declare(name, tuple(reversed(data.shape)), kind)
    writer.begin()
    for name, data, _ in tensors:
        writer.write(name, data)
    writer.close()


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tile-scales") / "mixed.gguf"
    write(path)
    return path


def exact(package: Package, name: str) -> torch.Tensor:
    """`name`'s values in float64: each code's E4M3 value times its tile's factor, exact (a 4-bit significand
    times an 8-bit power of two or a 24-bit F32 significand)."""
    shape, grid, kind = GRIDDED[name]
    codes = torch.from_numpy(np.array(package.raw(name + ".weight"))).view(torch.float8_e4m3fn).double().view(shape)
    raw = np.array(package.raw(name + ".scale"))
    if kind == Q.I8:
        factors = torch.exp2(torch.from_numpy(raw.view(np.uint8).astype(np.float64) - 127))
    else:
        factors = torch.from_numpy(raw.view(np.float32).astype(np.float64))
    factors = factors.view(grid)
    tile = (shape[-2] // grid[-2], shape[-1] // grid[-1])
    return codes * factors.repeat_interleave(tile[0], -2).repeat_interleave(tile[1], -1)


def test_one_file_reads_ue8m0_and_f32_grids_each_in_its_own_code(mixed: Path) -> None:
    """Every value is its code times its tile's factor as one float32 product, rounded once: exact under UE8M0, and
    correctly rounded under F32 (some products round), for a matrix, rows of it, a stack and each expert."""
    package = Package.local(mixed)
    weights = Weights(package, CPU, False)
    for name, (shape, _, kind) in GRIDDED.items():
        expected = exact(package, name)
        value = weights(name + ".weight")
        assert torch.equal(value, expected.float()), name
        inexact = int((value.double() != expected).sum())
        assert inexact == 0 if kind == Q.I8 else inexact > 0, (name, inexact)
        if len(shape) == 3:
            for index in range(shape[0]):
                assert torch.equal(weights.expert(name + ".weight", index), value[index]), (name, index)
        else:
            rows = torch.tensor([[17, 0], [17, 63]])
            assert torch.equal(weights.rows(name + ".weight", rows), value[rows]), name
    assert not weights.unread()


def test_a_perturbed_file_reads_each_grid_in_its_own_code(mixed: Path) -> None:
    """A perturbed reference reads a UE8M0 grid as UE8M0 bytes and an F32 grid as F32 elements: each value moves
    once, the same however it is read, a stacked expert's matrix alone and in its stack."""
    package = Package.local(mixed)
    plain, moved = Weights(package, CPU, False), Weights(package, CPU, False, Perturbation(5))
    for name, (shape, _, _) in GRIDDED.items():
        value = moved(name + ".weight")
        assert torch.equal(value, moved(name + ".weight")), name
        assert not torch.equal(value, plain(name + ".weight")), name
        if len(shape) == 3:
            for index in range(shape[0]):
                assert torch.equal(moved.expert(name + ".weight", index), value[index]), (name, index)


@pytest.mark.parametrize("name, grid, reason", [
    ("blk.0.ffn_down", np.array([[1.0, np.inf], [1.0, 1.0]], np.float32), "not finite"),
    ("blk.0.ffn_down", np.array([[1.0, 1.0], [np.nan, 1.0]], np.float32), "not finite"),
    ("blk.0.attn_q_b", np.full((2, 8), 247, np.uint8).view(np.int8), "above 246"),
], ids=["inf", "nan", "ue8m0-247"])
def test_an_unreadable_grid_element_is_refused(tmp_path: Path, name: str, grid: np.ndarray, reason: str) -> None:
    path = tmp_path / "refused.gguf"
    write(path, {name: grid})
    with pytest.raises(ValueError, match=reason):
        Weights(Package.local(path), CPU, False)(name + ".weight")


def test_a_grid_of_no_codes_type_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "f16.gguf"
    writer = Writer(path, {"general.architecture": Value(V.STRING, "test")})
    writer.declare("x.weight", (128, 128), Q.F8_E4M3)
    writer.declare("x.scale", (1, 1), Q.F16)
    writer.begin()
    writer.write("x.weight", np.zeros((128, 128), np.uint8))
    writer.write("x.scale", np.ones((1, 1), np.float16))
    writer.close()
    with pytest.raises(ValueError, match="I8 or F32"):
        Weights(Package.local(path), CPU, False)("x.weight")
