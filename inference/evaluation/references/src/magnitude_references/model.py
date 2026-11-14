"""Shared float32 building blocks for the independent model references.

Each family reference (`families/<family>.py`) reimplements the released modeling code
(transformers 5.17.0 `models/<family>/modeling_*.py` and the model's own config) in float32 over the
exact dequantized GGUF weights. It reads tensors by GGUF name and undoes the llama.cpp converter's
transforms itself (row permutations, splits, stored reparameterizations), so the forward pass follows
the model definition, never llama.cpp's graph. Every tensor the forward reads comes through
`Weights`, which dequantizes on demand; with `cache=False` each layer's weights are decoded when the
layer runs and dropped after, so a model larger than memory streams layer by layer over a batch of
sequences.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import cache, partial
import math
import os
import weakref

import numpy as np
import torch

from gguf.constants import GGMLQuantizationType

from magnitude_references import digest
from magnitude_references.gguf import Package, TensorInfo, dequantize
from magnitude_references.perturbation import UE8M0 as UE8M0_CODES, Format, Perturbation

F32 = torch.float32
# Where a row's bits can be its own (`digest.platform`'s rows_exact: Accelerate, nvpl), every product runs in calls
# of one shape: ROW_TILE rows (the last call padded with zero rows), and over a key axis KEY_TILE keys. There a GEMM's
# row is a function of that row and the call's shape alone, so with the shape fixed a row's bits cannot depend on how
# many rows share its call; any fixed sizes give that. The sizes are a speed choice: a short call (one expert's rows)
# pads by little, a long one runs as whole tiles; an op on rows that rounds a row by the tensor's length runs in
# ROW_TILE-row calls too (`pointwise`). Elsewhere (x86 MKL, which also rounds a row by its place in the call and by
# the call's row count, so no call shape makes a row its own) `linear`, `products` and `pointwise` call whole, as the
# released code does, whose bits they then are (`tiled`). Attention (`attention`, `gathered_attention`) runs in
# ROW_TILE-row blocks and KEY_TILE-key tiles on every platform: its softmax is summed tile by tile (`attend`), which is
# its definition here, not a platform's choice; within a block its products follow `tiled`.
ROW_TILE = 128
KEY_TILE = 128
MAX_UE8M0 = UE8M0_CODES.largest  # the largest byte whose E4M3 products are exact in float32
# A torch operation's result can depend on its OpenMP team size (a product's split of its reduction), so a
# reference fixes every team it runs, never the host: `THREADS` on the thread that builds and runs it (`Weights`
# sets it there) and 1 on each pool thread, since a thread that never sets its own takes the last size set on any
# thread. Products go to the platform's BLAS, whose team is the calling thread's, never to oneDNN (ACL on aarch64),
# which sizes its own team from the cores the process may run on: on GB10 a thread set to 1 ran a 20-thread ACL team of
# its own. The bits are then a function of the tree and the platform's BLAS; the pools are sized by the
# host, and order nothing. Being part of what the bits are, THREADS is a constant, not the host's core count; it is
# the most cores every host that builds references has (Apple M4 Max 16, GB10 20), so no host's team is
# oversubscribed. A host with fewer would time-slice the team: slower, the same bits.
THREADS = 16
torch.backends.mkldnn.enabled = False
WORKERS = os.cpu_count() or 1
# Two pools, so no task waits on its own pool: a dense tensor's row parts (`Weights.decode`), and a mixture's
# experts (`mixture`), which may read dense tensors.
DECODERS = ThreadPoolExecutor(WORKERS, thread_name_prefix="decode", initializer=torch.set_num_threads, initargs=(1,))
EXPERTS = ThreadPoolExecutor(WORKERS, thread_name_prefix="expert", initializer=torch.set_num_threads, initargs=(1,))


def submit(pool: ThreadPoolExecutor, task: Callable, *arguments) -> Future:
    """`task(*arguments)` on `pool`, under the calling thread's torch function modes, which are per thread: a
    reference run in float64 (`checks.Float64`) runs so on every worker it uses."""
    modes = torch.overrides._get_current_function_mode_stack()

    def run():
        with ExitStack() as stack:
            for mode in modes:
                stack.enter_context(mode)
            return task(*arguments)
    return pool.submit(run)


@dataclass(frozen=True)
class TileEncoding:
    """The element code of an F8_E4M3 weight's tile grid `<base>.scale`, named by the grid's GGUF type: the format a
    perturbation moves its elements in (None: the type's own), the refusal of a file's grid bytes, and the float32
    factors of stored grid bytes [n, grid cols]."""
    perturbed: Format | None
    refuse: Callable[[str, np.ndarray], None]
    factors: Callable[[np.ndarray], np.ndarray]


def _refuse_ue8m0(name: str, data: np.ndarray) -> None:
    if data.size and (largest := int(data.view(np.uint8).max())) > MAX_UE8M0:
        raise ValueError(f"{name}: UE8M0 byte {largest} is above {MAX_UE8M0}, where products leave float32")


def _refuse_f32(name: str, data: np.ndarray) -> None:
    if not np.isfinite(data.view(np.float32)).all():
        raise ValueError(f"{name}: a tile scale is not finite")


# `inference/docs/precision.md`, "Block-scaled FP8 weights": a grid's GGUF type names its element code. I8: one
# UE8M0 byte b per tile, factor 2^(b - 127), every product exact in float32. F32: the checkpoint's F32 scale per
# tile; a product is correctly rounded once.
TILE_ENCODINGS = {
    GGMLQuantizationType.I8: TileEncoding(
        UE8M0_CODES, _refuse_ue8m0,
        lambda data: np.ldexp(np.float32(1), data.view(np.uint8).astype(np.int32) - 127).astype(np.float32)),
    GGMLQuantizationType.F32: TileEncoding(None, _refuse_f32, lambda data: data.view(np.float32)),
}


class Weights:
    """Float32 tensors of a GGUF package by GGUF name, dequantized on demand.

    A stored tensor's value includes its `<base>.scale` companion: the NVFP4 second-level scale, or, for an
    F8_E4M3 tensor, its tile grid in the element code the grid's own GGUF type names (`TILE_ENCODINGS`: I8 UE8M0
    bytes, F32 scales), value = code * factor as one float32 product, so one file may hold grids of both. The tile
    is the quotient of the two shapes' last two axes (32x32 for dense DeepSeek weights, 1x32 for the Engram table,
    128x128 for GLM-5.3-Flash), and a stack of expert matrices [experts, rows, cols] has one grid per expert
    [experts, grid rows, grid cols].

    A mixture's experts run in parallel (`mixture`), each decoding its own matrices; any other tensor decodes in
    parts of its rows on a pool (`decode`). A decode is the same function of the same bytes on whatever thread it
    runs, so reads may come from any thread.

    Every read takes its bytes through `stored`, which applies the storage-resolution `perturbation` when the
    reference runs one (`perturbation.py`), so each stored value moves once, by its identity, however it is
    read."""

    def __init__(self, package: Package, device: torch.device, cache: bool, perturbation: Perturbation | None = None):
        self.package, self.device, self.cache, self.perturbation = package, device, cache, perturbation
        torch.set_num_threads(THREADS)  # this thread's team size (`THREADS`): the reference runs here
        assert not torch.backends.mkldnn.enabled, "oneDNN sizes its own teams from the host: a reference keeps it off"
        self.values: dict[str, torch.Tensor] = {}
        self.read: set[str] = set()
        self.spare: defaultdict[int, list[np.ndarray]] = defaultdict(list)  # float32 buffers `lend` reuses

    @contextmanager
    def uncached(self) -> Iterator[None]:
        """Within the block, every read decodes afresh and nothing is kept; the cached values are dropped on entry.
        For a copy of the weights read once (a released model's parameters)."""
        self.values.clear()
        cache, self.cache = self.cache, False
        try:
            yield
        finally:
            self.cache = cache

    def has(self, name: str) -> bool:
        return self.package.has(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.package.info(name).array_shape

    def __call__(self, name: str) -> torch.Tensor:
        if name in self.values:
            return self.values[name]
        value = self.scaled(name, torch.from_numpy(self.decode(name)).to(self.device))
        self.read.add(name)
        if self.cache:
            self.values[name] = value
        return value

    def decode(self, name: str) -> np.ndarray:
        """`name`'s float32 values (NumPy order), its GGUF rows stored, moved and decoded in parts on the decode
        pool: each row's bytes, and so its values, are its own, so the parts are the whole tensor's."""
        info = self.package.info(name)
        count = int(np.prod(info.shape[1:], dtype=np.int64))
        out = np.empty((count, info.shape[0]), dtype=np.float32)
        step = -(-count // WORKERS)

        def part(start: int) -> None:
            rows = slice(start, min(start + step, count))
            shape = (info.shape[0], rows.stop - start)
            dequantize(TensorInfo(name, shape, info.type, 0), self.stored(name, rows).reshape(-1), out[rows])

        for decoding in [submit(DECODERS, part, start) for start in range(0, count, step)]:
            decoding.result()
        return out.reshape(info.array_shape)

    def stored(self, name: str, rows: np.ndarray | slice = slice(None), encoding: Format | None = None) -> np.ndarray:
        """The stored bytes [rows, row bytes] of `name`'s GGUF `rows` (each `shape[0]` elements; all by
        default), moved by the perturbation, if any (`encoding`: an I8 companion scale's format)."""
        info = self.package.info(name)
        count = int(np.prod(info.shape[1:], dtype=np.int64))
        data = self.package.raw(name).reshape(count, info.nbytes // count)[rows]
        if self.perturbation is None:
            return data
        index = np.arange(count)[rows] if isinstance(rows, slice) else rows
        return self.perturbation.stored(name, info.type, info.shape[0], data, index, encoding)

    def scaled(self, name: str, value: torch.Tensor, rows: torch.Tensor | None = None,
               expert: int | None = None) -> torch.Tensor:
        """Stored values of `name` (only its `rows`, or only its `expert`'s matrix, when given) times their
        companion scale."""
        if (scale := self.global_scale(name)) is not None:
            return value * (scale if expert is None else scale[expert])
        info = self.package.info(name)
        if info.type != GGMLQuantizationType.F8_E4M3:
            return value
        if rows is not None:
            index = rows.cpu().numpy()
        elif expert is not None:
            index = np.arange(expert * info.shape[1], (expert + 1) * info.shape[1])
        else:
            index = np.arange(int(np.prod(info.shape[1:], dtype=np.int64)))
        factors = self.tile_scale(name, index)  # [rows, grid cols], each over its tile of columns
        return (value.reshape(*factors.shape, -1) * factors[..., None]).view_as(value)

    def tile_scale(self, name: str, rows: np.ndarray) -> torch.Tensor:
        """The tile factors of F8_E4M3 `name`'s GGUF `rows` (rows of its matrix, or of a stack's matrices one after
        another), one per row and tile of columns: [rows, grid cols]."""
        scale_name = name.removesuffix(".weight") + ".scale"
        info, scale = self.package.info(name), self.package.info(scale_name)
        tiles = TILE_ENCODINGS.get(scale.type)
        if tiles is None or len(scale.shape) != len(info.shape) or len(info.shape) not in (2, 3) \
                or scale.shape[2:] != info.shape[2:]:
            raise ValueError(f"{name}: a tile-scaled matrix or stack needs a grid {scale_name} of "
                             f"{' or '.join(kind.name for kind in TILE_ENCODINGS)} of its rank and stack, got {scale}")
        (total, cols), (grid_rows, grid_cols) = info.array_shape[-2:], scale.array_shape[-2:]
        tile_rows, tile_cols = total // grid_rows, cols // grid_cols
        if tile_rows * grid_rows != total or tile_cols * grid_cols != cols:
            raise ValueError(f"{name}: {scale_name} {scale.array_shape} does not tile {info.array_shape}")
        # The file's grid decides the refusal: a perturbed element stays within its format's finite range.
        tiles.refuse(scale_name, self.package.raw(scale_name))
        # A row of a stack's flattened matrices lies in the grid row of the same flattened position.
        grid, inverse = np.unique(rows // tile_rows, return_inverse=True)
        codes = self.stored(scale_name, grid, tiles.perturbed)[inverse]
        self.read.add(scale_name)
        return torch.from_numpy(tiles.factors(codes)).to(self.device)

    def global_scale(self, name: str) -> torch.Tensor | None:
        """An NVFP4 tensor's second-level scale (`<base>.scale`: [1], or [experts] for stacked experts).
        The format's value is fp4 * ue4m3 block scale * this scale; ggml applies it to the matmul output,
        which is the same product."""
        if self.package.info(name).type != GGMLQuantizationType.NVFP4:
            return None
        scale_name = name.removesuffix(".weight") + ".scale"
        if not self.package.has(scale_name):
            return None
        self.read.add(scale_name)
        return torch.from_numpy(self.package.tensor(scale_name)).to(self.device)

    def expert(self, name: str, index: int) -> torch.Tensor:
        """One expert's matrix [rows, cols] of a stacked expert tensor (GGUF shape [cols, rows, experts])."""
        key = f"{name}#{index}"
        if key in self.values:
            return self.values[key]
        value = self.decode_expert(name, index)
        self.read.add(name)
        if self.cache:
            self.values[key] = value
        return value

    def decode_expert(self, name: str, index: int) -> torch.Tensor:
        info = self.package.info(name)
        if len(info.shape) != 3:
            raise ValueError(f"{name} is not a stacked expert tensor: {info.shape}")
        rows = info.shape[1]
        raw = self.stored(name, slice(index * rows, (index + 1) * rows)).reshape(-1)
        value = self.lend(lambda out: dequantize(TensorInfo(name, info.shape[:2], info.type, 0), raw, out),
                          rows * info.shape[0])
        return self.scaled(name, value.to(self.device), expert=index)

    def lend(self, decode: Callable[[np.ndarray], np.ndarray], count: int) -> torch.Tensor:
        """`decode`'s values written into a float32 buffer of `count` elements, as a tensor. A streaming reference
        (`cache` off) takes the buffer from those its earlier expert matrices returned, and returns this one when
        the array the tensor wraps dies, which is after every tensor sharing its memory: a matrix's pages are
        allocated and faulted in once, not once per read."""
        spare = self.spare[count]
        try:
            buffer = spare.pop()  # empty while caching: nothing is returned
        except IndexError:
            buffer = np.empty(count, dtype=np.float32)
        array = decode(buffer)
        if not self.cache:
            weakref.finalize(array, spare.append, buffer)
        return torch.from_numpy(array)

    def rows(self, name: str, indices: torch.Tensor) -> torch.Tensor:
        """Rows `indices` (any shape) of a 2-D tensor, dequantizing only the rows used (large tables)."""
        if name in self.values:
            return self.values[name][indices]
        info = self.package.info(name)
        if len(info.shape) != 2:
            raise ValueError(f"{name} is not a matrix: {info.shape}")
        unique, inverse = torch.unique(indices.reshape(-1), return_inverse=True)
        selected = self.stored(name, unique.cpu().numpy()).reshape(-1)
        table = torch.from_numpy(dequantize(TensorInfo(name, (info.shape[0], len(unique)), info.type, 0), selected)).to(self.device)
        table = self.scaled(name, table, unique)
        self.read.add(name)
        return table[inverse.to(self.device)].reshape(*indices.shape, info.shape[0])

    def unread(self) -> list[str]:
        """Tensors of the package the forward never read (a strict reference binds every tensor)."""
        return sorted(set(self.package.names()) - self.read)


# ---------------------------------------------------------------------------------------------
# Elementwise and normalization


def padded(x: torch.Tensor, dim: int, tile: int, value: float | bool = 0) -> torch.Tensor:
    """x along `dim` filled with `value` to a multiple of `tile`."""
    dim %= x.dim()
    extra = -x.shape[dim] % tile
    return torch.cat((x, x.new_full((*x.shape[:dim], extra, *x.shape[dim + 1:]), value)), dim) if extra else x


def row_tiles(product, *xs: torch.Tensor, dim: int) -> torch.Tensor:
    """product(*tiles) over the rows of xs along `dim`, ROW_TILE rows a call, the last padded with zero rows; its
    result's rows along the same `dim`. A row's bits depend on its own row alone (see ROW_TILE)."""
    rows = xs[0].shape[dim]
    tiles = zip(*(padded(x, dim, ROW_TILE).split(ROW_TILE, dim) for x in xs))
    return torch.cat([product(*(tile.contiguous() for tile in group)) for group in tiles], dim).narrow(dim, 0, rows)


@cache
def tiled(device: str) -> bool:
    """Whether `linear`, `products` and `pointwise` on `device` (a device type) run in fixed row and key tiles: where
    its platform makes a row's bits its own (`digest.platform`), else in the released code's whole calls (see
    ROW_TILE)."""
    return digest.platform(device)["rows_exact"]


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [..., K] · weightᵀ (+ bias), in row tiles or whole (`tiled`)."""
    if not tiled(x.device.type):
        return torch.nn.functional.linear(x, weight, bias)
    product = row_tiles(lambda rows: torch.nn.functional.linear(rows, weight, bias), x.reshape(-1, x.shape[-1]), dim=0)
    return product.view(*x.shape[:-1], product.shape[-1])


def products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [..., M, D] · b [..., N, D]ᵀ -> [..., M, N], in row tiles of a and key tiles of b: each GEMM [ROW_TILE, D] ·
    [D, KEY_TILE], so an entry depends on its own row and key alone; or whole (`tiled`)."""
    if not tiled(a.device.type):
        return a @ b.transpose(-1, -2)
    keys = b.shape[-2]
    tiles = [tile.transpose(-1, -2).contiguous() for tile in padded(b, -2, KEY_TILE).split(KEY_TILE, -2)]
    return row_tiles(lambda rows: torch.cat([rows @ tile for tile in tiles], -1), a, dim=-2).narrow(-1, 0, keys)


def attend(scores: torch.Tensor, weigh, sinks: torch.Tensor | None) -> torch.Tensor:
    """Softmax over each row's keys, scores [..., S] with S a multiple of KEY_TILE and -inf a key it does not see, from
    the row's max (clamped at -1e30) with exp(sink) in the denominator when `sinks` (broadcast to [..., 1]) is given;
    returns the sum over key tiles, in order, of weigh(tile index, the tile's weights [..., KEY_TILE]). Its sums run in
    fixed tiles, an empty tile adding exact zeros, so a row's bits depend on its visible keys alone."""
    peak = scores.amax(-1, keepdim=True).clamp(min=-1e30)
    weights = torch.exp(scores - peak).split(KEY_TILE, -1)
    total = sum(tile.sum(-1, keepdim=True) for tile in weights)
    if sinks is not None:
        total = total + torch.exp(sinks - peak)
    return sum(weigh(index, (tile / total).contiguous()) for index, tile in enumerate(weights))


def pointwise(op, x: torch.Tensor) -> torch.Tensor:
    """op(x) of an op on rows (elementwise, or along the last axis), in ROW_TILE-row calls where `tiled`, so every
    call has one length: torch cuts a parallel op into chunks by the tensor's length, and its vectorised body and
    scalar tail round some ops differently, so on a whole tensor a row's bits would depend on how many rows it has.
    The forward calls such an op only through here (tests/test_rows.py)."""
    if not tiled(x.device.type):
        return op(x)
    rows = row_tiles(op, x.reshape(-1, x.shape[-1]), dim=0)
    return rows.view(*x.shape[:-1], *rows.shape[1:])


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """transformers' RMSNorm in float32: x * rsqrt(mean(x^2) + eps) (* weight)."""
    normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return normalized if weight is None else normalized * weight


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return pointwise(torch.sigmoid, x)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """torch's LayerNorm over the last axis, with its weight and bias."""
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x * rsqrt(sum x^2 + eps) over the last axis (FLA's l2norm)."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def silu(x: torch.Tensor) -> torch.Tensor:
    return pointwise(torch.nn.functional.silu, x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return pointwise(partial(torch.nn.functional.gelu, approximate="tanh"), x)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def softplus(x: torch.Tensor) -> torch.Tensor:
    return pointwise(torch.nn.functional.softplus, x)


def causal_conv(u: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv of u [B, T, C] from a zero window, no bias: out[t, c] = sum_k kernel[c, k] u[t-(K-1)+k, c]."""
    taps = kernel.shape[1]
    padded = torch.nn.functional.pad(u, (0, 0, taps - 1, 0))
    return sum(padded[:, k:k + u.shape[1]] * kernel[:, k] for k in range(taps))


# ---------------------------------------------------------------------------------------------
# Manifold-constrained hyper-connections (Sinkhorn mixes over S residual streams)


def sinkhorn_mixes(streams: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, iterations: int,
                   eps: float, norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pre [.., S], post [.., S], comb [.., S, S]) of a sublayer site from its streams [.., S, H]: with
    m = (flatten(X) fn^T) / rms(flatten(X)), pre = sigmoid(m[:S] a0 + b) + eps, post = 2 sigmoid(m[S:2S] a1 + b),
    comb = Sinkhorn(softmax(m[2S:] a2 + b) + eps) over `iterations` normalisations."""
    count = streams.shape[-2]
    flat = streams.flatten(-2)
    mixed = linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = sigmoid(mixed[..., :count] * scale[0] + base[:count]) + eps
    post = 2 * sigmoid(mixed[..., count:2 * count] * scale[1] + base[count:2 * count])
    comb = torch.softmax((mixed[..., 2 * count:] * scale[2] + base[2 * count:]).unflatten(-1, (count, count)), dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def stream_collapse(streams: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """sum_c pre[c] X[c]: a sublayer's input from the streams [.., S, H]."""
    return (pre[..., None] * streams).sum(-2)


def stream_expand(out: torch.Tensor, streams: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """X'[k] = post[k] y + sum_j comb[j, k] X[j]: the streams after a sublayer's output y. As DeepSeek's `hc_post`,
    the residual mix is products summed over the source stream, not a product of matrices, whose BLAS kernel (fused
    multiply-adds on MKL) rounds differently."""
    return post[..., None] * out[..., None, :] + (comb[..., None] * streams[..., None, :]).sum(-3)


# ---------------------------------------------------------------------------------------------
# Rotary position embedding (transformers `modeling_rope_utils`, rotate-half layout)


def default_inv_freq(base: float, dim: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(F32) / dim))


def yarn_inv_freq(base: float, dim: int, factor: float, original_context: int, beta_fast: float = 32,
                  beta_slow: float = 1, truncate: bool = True, attention_factor: float | None = None,
                  mscale: float | None = None, mscale_all_dim: float | None = None) -> tuple[torch.Tensor, float]:
    """`_compute_yarn_parameters`: blended inverse frequencies and the cos/sin amplitude."""
    def get_mscale(scale: float, m: float = 1) -> float:
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        attention_factor = (float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim))
                            if mscale and mscale_all_dim else get_mscale(factor))

    def correction_dim(rotations: float) -> float:
        return (dim * math.log(original_context / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=F32) - low) / (high - low), 0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).to(F32) / dim)
    extrapolation, interpolation = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    extrapolation_factor = 1 - ramp
    return interpolation * (1 - extrapolation_factor) + extrapolation * extrapolation_factor, attention_factor


def proportional_inv_freq(base: float, head_dim: int, partial: float, factor: float = 1.0) -> torch.Tensor:
    """`_compute_proportional_rope_parameters`: the first `partial` of the pairs rotate with frequencies
    over the whole head width; the remaining pairs have frequency zero (identity)."""
    angles = int(partial * head_dim // 2)
    rotated = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).to(F32) / head_dim))
    inv_freq = torch.cat((rotated, torch.zeros(head_dim // 2 - angles, dtype=F32)))
    return inv_freq / factor


@dataclass(frozen=True)
class Rotary:
    """cos/sin tables [positions, 2 * pairs] for one rotary configuration."""
    cos: torch.Tensor
    sin: torch.Tensor

    @staticmethod
    def build(positions: torch.Tensor, inv_freq: torch.Tensor, amplitude: float = 1.0) -> "Rotary":
        freqs = positions.to(F32)[:, None] * inv_freq.to(positions.device)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return Rotary(emb.cos() * amplitude, emb.sin() * amplitude)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., T, D] with D == table width: x*cos + rotate_half(x)*sin."""
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * self.cos + rotated * self.sin

    def apply_partial(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the leading table-width dims of x and pass the rest through."""
        width = self.cos.shape[-1]
        return torch.cat((self.apply(x[..., :width]), x[..., width:]), dim=-1)

    def apply_pairs(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """x [..., T, D] with D == table width, rotated as adjacent pairs (2j, 2j+1) by frequency j (the
        complex-number view); `inverse` rotates by the conjugate."""
        pairs = x.shape[-1] // 2
        cos, sin = self.cos[..., :pairs], self.sin[..., :pairs]
        if inverse:
            sin = -sin
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def unpermute_rows(weight: torch.Tensor, head_count: int) -> torch.Tensor:
    """Inverse of llama.cpp's `LlamaModel.permute` / `_unpermute_for_rope` (used for NORM-rope GGUFs):
    GGUF row 2j+s of a head is checkpoint (rotate-half) row s*(D/2)+j. Works on [rows] and [rows, cols]."""
    rows = weight.shape[0]
    return weight.reshape(head_count, rows // head_count // 2, 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def permute_rows(weight: torch.Tensor, head_count: int) -> torch.Tensor:
    """llama.cpp's permutation itself (rotate-half order to adjacent pairs)."""
    rows = weight.shape[0]
    return weight.reshape(head_count, 2, rows // head_count // 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


# ---------------------------------------------------------------------------------------------
# Attention


def visibility(query_positions: torch.Tensor, key_positions: torch.Tensor, *, causal: bool = True,
               window: int | None = None) -> torch.Tensor:
    """Boolean [T, S]: query t may attend key s. `window` W keeps keys with q - k < W (the query's own
    token included), transformers' sliding-window convention."""
    delta = query_positions[:, None] - key_positions[None, :]
    visible = torch.ones_like(delta, dtype=torch.bool)
    if causal:
        visible &= delta >= 0
    if window is not None:
        visible &= delta < window
    return visible


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, visible: torch.Tensor, scale: float,
              sinks: torch.Tensor | None = None) -> torch.Tensor:
    """Grouped-query softmax attention in float32.

    q [B, Hq, T, D], k [B, Hk, S, D], v [B, Hk, S, Dv], visible [T, S] (or [B, T, S]). `sinks` [Hq]
    adds exp(sink_h) to head h's softmax denominator (a logit with a zero value).
    Returns [B, T, Hq * Dv] (heads concatenated per row).
    """
    batch, heads, rows, width = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    if heads != groups * kv_heads:
        raise ValueError(f"{heads} query heads do not group over {kv_heads} kv heads")
    values = padded(v, 2, KEY_TILE).split(KEY_TILE, 2)
    # hidden keys and the key padding are -inf; the row padding sees every key, finite
    hidden = padded(~(visible if visible.dim() == 3 else visible[None]), -1, KEY_TILE, True)[:, None]
    sink = None if sinks is None else sinks.view(1, heads, 1, 1)

    def block(q: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # a kv head's query heads (each consecutive `groups`) multiply it as one matrix of their rows, so no kv
        # head is copied per query head
        grouped = q.reshape(batch, kv_heads, groups * ROW_TILE, width)
        scores = products(grouped, k).view(batch, heads, ROW_TILE, -1) * scale
        scores = padded(scores, -1, KEY_TILE).masked_fill(hidden, float("-inf"))

        def weigh(index: int, weights: torch.Tensor) -> torch.Tensor:
            mixed = weights.view(batch, kv_heads, groups * ROW_TILE, KEY_TILE) @ values[index]
            return mixed.view(batch, heads, ROW_TILE, -1)
        return attend(scores, weigh, sink)

    out = row_tiles(block, q, hidden.expand(batch, -1, -1, -1), dim=-2)
    return out.transpose(1, 2).reshape(batch, rows, heads * v.shape[-1])


def gathered_attention(q: torch.Tensor, kv: torch.Tensor, sinks: torch.Tensor | None, slots: torch.Tensor,
                       scale: float) -> torch.Tensor:
    """Attention over each query row's own key rows, in float32, as DeepSeek V4's `sparse_attn` defines it (its
    `kernel.py` lines 310-403): the rows `slots` [B, T, K] of kv [B, N, D] (keys and values both), -1 an empty slot;
    head h's scores q [B, T, H, D] · row, their softmax from the rows' max (clamped at -1e30) with exp(sink_h) in the
    denominator when `sinks` [H] is given (`attend`); its rows weighted by it. Returns [B, T, H, D] in q's dtype.

    A row's bits depend on its set of rows alone, not on their order, the empty slots or K: its rows go in position
    order, empties last, padded to whole key tiles."""
    empty = kv.shape[1]  # sorts after every row
    slots = padded(slots.long().masked_fill(slots < 0, empty), -1, KEY_TILE, empty).sort(dim=-1).values
    sink = None if sinks is None else sinks.float().view(1, 1, -1, 1)

    def block(q: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        rows = kv[torch.arange(kv.shape[0])[:, None, None], slots.clamp(max=empty - 1)].float()  # [B, R, K, D]
        tiles = [tile.contiguous() for tile in rows.split(KEY_TILE, 2)]
        scores = torch.cat([torch.einsum("bmhd,bmkd->bmhk", q.float(), tile) for tile in tiles], -1) * scale
        scores = scores.masked_fill((slots == empty)[:, :, None], float("-inf"))
        return attend(scores, lambda index, weights: torch.einsum("bmhk,bmkd->bmhd", weights, tiles[index]), sink)

    return row_tiles(block, q, slots, dim=1).to(q.dtype)


def heads(x: torch.Tensor, count: int) -> torch.Tensor:
    """[B, T, count*D] -> [B, count, T, D]."""
    batch, rows, width = x.shape
    return x.view(batch, rows, count, width // count).transpose(1, 2)


# ---------------------------------------------------------------------------------------------
# Routed experts


def route_top_k(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k largest scores per row (torch.topk, as transformers selects)."""
    return torch.topk(scores, k, dim=-1).indices


def lower_index_top(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Indices of the `count` largest scores along the last axis, ties to the lower index."""
    return torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :count]


def unit_selection(scores: torch.Tensor, ends: torch.Tensor, top: int, rate: int) -> torch.Tensor:
    """The rows each query row reads under an indexer selecting units of `rate` rows, as slots [B, T, K]
    (`gathered_attention`'s form, -1 an empty slot), K = min(top, U)·rate + rate - 1.

    scores [B, T, U]: each row's score of every unit u, which covers the rows [u·rate, (u+1)·rate). ends [T]: each
    row's visible rows [0, end), so its complete units are the first end // rate. A row reads the rows of its best
    min(top, count) complete units (ties to the lower unit) and its tail, the visible rows no complete unit covers,
    [count·rate, end): while count <= top, every visible row.
    """
    batch, steps, units = scores.shape
    counts = ends // rate
    best_count = min(top, units)
    visible = torch.arange(units, device=scores.device)[None, :] < counts[:, None]
    best = lower_index_top(scores.masked_fill(~visible, float("-inf")), best_count)  # [B, T, best]
    taken = torch.arange(best_count, device=scores.device)[None, :] < counts.clamp(max=best_count)[:, None]
    unit_rows = best[..., None] * rate + torch.arange(rate, device=scores.device)  # [B, T, best, rate]
    picked = torch.where(taken[..., None], unit_rows, -1).flatten(-2)
    tail = (counts * rate)[:, None] + torch.arange(rate - 1, device=scores.device)  # [T, rate - 1]
    tail = torch.where(tail < ends[:, None], tail, -1)
    return torch.cat((picked, tail.expand(batch, -1, -1)), -1)


def mixture(x: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, expert, weighted: bool = False) -> torch.Tensor:
    """sum_k weights[:, k] * expert(e_k, x) over the selected experts of every row.

    x [N, H]; selected, weights [N, K]; expert(index, rows [n, H]) -> [n, H_out]. A `weighted` expert applies its
    rows' weights [n, 1] itself, inside its computation: expert(index, rows, weights). The experts' parts compute in
    parallel (`EXPERTS`), and their rows add into the sum in the order of the experts, so the sum is the same as one
    thread's.
    """
    parts = [(index, *torch.where(selected == index)) for index in torch.unique(selected).tolist()]

    def part(index: int, rows: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        if weighted:
            return expert(index, x[rows], weights[rows, slot, None])
        return expert(index, x[rows]) * weights[rows, slot, None]

    computing = [submit(EXPERTS, part, *arguments) for arguments in parts]
    out = None
    try:
        for (_, rows, _), computed in zip(parts, computing):
            y = computed.result()
            if out is None:
                out = torch.zeros(x.shape[0], y.shape[-1], dtype=F32, device=x.device)
            out.index_add_(0, rows, y)
    finally:
        for computed in computing:
            computed.cancel()
    return out


# ---------------------------------------------------------------------------------------------
# Reference interface


class Reference:
    """A family reference: `hidden(tokens)` runs the decoder, `logits(hidden)` the readout.

    `tokens` is int64 [B, T]; every sequence starts at position 0 with empty state. Every family is
    causal, so a row's output depends only on the tokens up to it: right-padding a batch of shorter
    sequences leaves their rows exact. `taps` names layers whose *input* residual is returned
    (the DFlash/DSpark target features).
    """

    architecture: str
    vocab_size: int
    hidden_size: int
    block_count: int
    # Whether the logits move only by float32 rounding when the summation order changes. A reference
    # that rounds to trained codes is not: another order moves a value across a code boundary, a
    # whole code step, which no rounding bound admits (`cache.admit`), so it builds on the CPU alone.
    continuous: bool = True
    # Named measurement configurations beside the model's own numerics: the constructor's keyword
    # arguments of each. A variant is part of a reference's identity (`digest.implementation`).
    variants: dict[str, dict] = {}

    def __init__(self, package: Package, weights: Weights, layers: int | None):
        self.package, self.weights = package, weights

    @classmethod
    def open(cls, package: Package, device: torch.device, cache: bool, layers: int | None = None,
             variant: str | None = None) -> Reference:
        """The reference of `package`, in its model numerics, the named measurement `variant` (`variants`), or its
        numerics over storage-resolution-perturbed stored values (`perturbed:<seed>`, `perturbation.py`).
        `cache=False` streams: every layer's weights are dequantized when the layer runs and dropped after."""
        perturbation = Perturbation.of(variant)
        if variant is not None and perturbation is None and variant not in cls.variants:
            raise ValueError(f"{package.architecture} has no reference variant {variant!r}: "
                             f"{sorted(cls.variants)} or perturbed:<seed>")
        options = cls.variants[variant] if variant in cls.variants else {}
        return cls(package, Weights(package, device, cache, perturbation), layers, **options)

    def hidden(self, tokens: torch.Tensor, taps: tuple[int, ...] = ()) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        """The LM head (untied `output`, else the tied embedding) without any logit transform; draft
        models read the target's head this way."""
        return linear(hidden, self.weights("token_embd.weight" if self.tied else "output.weight"))

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.hidden(tokens)
        return self.logits(hidden)

    # transformers cross-check (checks.py): the family's config and state dict ----------------------

    tied: bool

    def hf_config(self) -> dict:
        raise NotImplementedError

    def hf_parameters(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def hf_derived(self) -> list[str]:
        """State-dict entries transformers ties to supplied ones."""
        return ["lm_head.weight"] if self.tied else []

    def hf_build(self, config):
        """The transformers model for `hf_config()` (a causal LM by default), float32, eager attention."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_config(config, dtype=F32, attn_implementation="eager")

    def hf_unused_prefixes(self) -> tuple[str, ...]:
        """State-dict prefixes of submodules a text-only forward never runs (e.g. a vision tower)."""
        return ()

    def hf_checkpoint(self, name: str, read) -> torch.Tensor:
        """A transformers parameter rebuilt from the released checkpoint; `read(key)` loads one
        checkpoint tensor as float32 and raises KeyError when the checkpoint lacks it."""
        return read(name)


def layer_count(package: Package, layers: int | None) -> int:
    total = package.key("block_count")
    if layers is not None and not 0 < layers <= total:
        raise ValueError(f"--layers {layers} outside 1..{total}")
    return total if layers is None else layers


def per_layer(value, count: int) -> list:
    """A per-layer metadata value: an array of `count` entries or one scalar for every layer."""
    if isinstance(value, list):
        if len(value) < count:
            raise ValueError(f"per-layer array has {len(value)} entries, expected {count}")
        return value[:count]
    return [value] * count


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", F32).numpy()
