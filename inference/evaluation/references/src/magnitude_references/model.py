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

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import os

import numpy as np
import torch

from gguf.constants import GGMLQuantizationType

from magnitude_references.gguf import Package, TensorInfo, dequantize
from magnitude_references.perturbation import UE8M0 as UE8M0_CODES, Format, Perturbation

F32 = torch.float32
QUERY_BLOCK = 1024  # attention query rows per score block
# F8_E4M3 weights with UE8M0 block exponents (`inference/docs/precision.md`, "UE8M0-scaled FP8 weights")
SCALE_ENCODING = "magnitude.scale_encoding"
UE8M0 = "ue8m0"
MAX_UE8M0 = UE8M0_CODES.largest  # the largest byte whose E4M3 products are exact in float32
DECODERS = ThreadPoolExecutor(os.cpu_count() or 1, thread_name_prefix="expert-decode")
EXPERT_WINDOW = 2 * (os.cpu_count() or 1)  # expert matrices decoded ahead of their reads, at most


class Weights:
    """Float32 tensors of a GGUF package by GGUF name, dequantized on demand.

    A stored tensor's value includes its `<base>.scale` companion: the NVFP4 second-level scale, or,
    for an F8_E4M3 tensor of a file declaring `magnitude.scale_encoding = "ue8m0"`, one UE8M0 byte b
    (I8) per block, value = code * 2^(b - 127). The block grid is the one the two shapes define (32x32
    for dense DeepSeek weights, 1x32 for the Engram table). Both products are exact in float32 (UE8M0
    bytes above 246 are refused).

    Expert matrices a layer is about to read (`plan_experts`) decode ahead on a thread pool, at most
    `EXPERT_WINDOW` at a time, so their decoding overlaps each other and the layer's products; a decode is
    the same function of the same bytes wherever it runs.

    Every read takes its bytes through `stored`, which applies the storage-resolution `perturbation` when the
    reference runs one (`perturbation.py`), so each stored value moves once, by its identity, however it is
    read."""

    def __init__(self, package: Package, device: torch.device, cache: bool, perturbation: Perturbation | None = None):
        self.package, self.device, self.cache, self.perturbation = package, device, cache, perturbation
        self.values: dict[str, torch.Tensor] = {}
        self.read: set[str] = set()
        self.planned: deque[tuple[str, int]] = deque()
        self.decoding: dict[str, Future[torch.Tensor]] = {}

    def has(self, name: str) -> bool:
        return self.package.has(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return self.package.info(name).array_shape

    def __call__(self, name: str) -> torch.Tensor:
        if name in self.values:
            return self.values[name]
        info = self.package.info(name)
        value = self.scaled(name, torch.from_numpy(dequantize(info, self.stored(name).reshape(-1))).to(self.device))
        self.read.add(name)
        if self.cache:
            self.values[name] = value
        return value

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
        if self.package.info(name).type != GGMLQuantizationType.F8_E4M3:
            return value
        return value * self.block_scale(name, rows)

    def block_scale(self, name: str, rows: torch.Tensor | None) -> torch.Tensor:
        """The UE8M0 factors of an F8_E4M3 matrix, expanded to its elements ([rows, cols])."""
        encoding = self.package.metadata.get(SCALE_ENCODING)
        if encoding is None or encoding.value != UE8M0:
            raise ValueError(f"{name}: F8_E4M3 needs {SCALE_ENCODING} = {UE8M0}, the file declares {encoding}")
        scale_name = name.removesuffix(".weight") + ".scale"
        info, scale = self.package.info(name), self.package.info(scale_name)
        if len(info.shape) != 2 or scale.type != GGMLQuantizationType.I8 or len(scale.shape) != 2:
            raise ValueError(f"{name}: a block-scaled matrix needs an I8 matrix {scale_name}, got {scale}")
        (total, cols), (grid_rows, grid_cols) = info.array_shape, scale.array_shape
        block_rows, block_cols = total // grid_rows, cols // grid_cols
        if block_rows * grid_rows != total or block_cols * grid_cols != cols:
            raise ValueError(f"{name}: {scale_name} {scale.array_shape} does not tile {info.array_shape}")
        file_bytes = self.package.raw(scale_name).view(np.uint8)
        if file_bytes.size and (largest := int(file_bytes.max())) > MAX_UE8M0:
            raise ValueError(f"{scale_name}: UE8M0 byte {largest} is above {MAX_UE8M0}, where products leave float32")
        # The rows used are picked before the codes widen: a table's grid is as large as its codes' rows. A
        # perturbed byte stays within UE8M0's range, so the file's bytes decide the refusal.
        if rows is None:
            codes = self.stored(scale_name, encoding=UE8M0_CODES).view(np.uint8).repeat(block_rows, axis=0)
        else:
            codes = self.stored(scale_name, rows.cpu().numpy() // block_rows, UE8M0_CODES).view(np.uint8)
        exponents = codes.astype(np.int32) - 127
        self.read.add(scale_name)
        factors = np.ldexp(np.float32(1), exponents).astype(np.float32)  # exact powers of two
        return torch.from_numpy(factors.repeat(block_cols, axis=1)).to(self.device)

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
        if (decoding := self.decoding.pop(key, None)) is not None:
            value = decoding.result()
            self.decode_planned()
        else:
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
        single = TensorInfo(name, info.shape[:2], info.type, 0)
        return self.scaled(name, torch.from_numpy(dequantize(single, raw)).to(self.device), expert=index)

    def plan_experts(self, names: tuple[str, ...], indices: list[int]) -> None:
        """The matrices `names` of each of `indices`, in that order, as the layer will read them
        (`expert`): decoded ahead, a window at a time. The last plan was read whole."""
        if self.planned or self.decoding:
            raise RuntimeError("a layer left planned expert matrices unread")
        self.planned = deque((name, index) for index in indices for name in names
                             if f"{name}#{index}" not in self.values)
        self.decode_planned()

    def decode_planned(self) -> None:
        while self.planned and len(self.decoding) < EXPERT_WINDOW:
            name, index = self.planned.popleft()
            self.decoding[f"{name}#{index}"] = DECODERS.submit(self.decode_expert, name, index)

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


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """transformers' RMSNorm in float32: x * rsqrt(mean(x^2) + eps) (* weight)."""
    normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return normalized if weight is None else normalized * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


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
    batch, heads, rows, _ = q.shape
    groups = heads // k.shape[1]
    if heads != groups * k.shape[1]:
        raise ValueError(f"{heads} query heads do not group over {k.shape[1]} kv heads")
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    mask = visible if visible.dim() == 3 else visible[None]
    # Query rows in blocks bound the score matrix to [B, H, QUERY_BLOCK, S]; each row's softmax is
    # independent of the others, so blocking does not change the definition.
    out = []
    for start in range(0, rows, QUERY_BLOCK):
        end = min(start + QUERY_BLOCK, rows)
        scores = torch.matmul(q[:, :, start:end], k.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask[:, None, start:end], float("-inf"))
        if sinks is not None:
            scores = torch.cat((scores, sinks.view(1, heads, 1, 1).expand(batch, heads, end - start, 1)), dim=-1)
        out.append(torch.matmul(torch.softmax(scores, dim=-1)[..., :v.shape[-2]], v))
    return torch.cat(out, dim=2).transpose(1, 2).reshape(batch, rows, heads * v.shape[-1])


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


def mixture(x: torch.Tensor, selected: torch.Tensor, weights: torch.Tensor, expert, weighted: bool = False) -> torch.Tensor:
    """sum_k weights[:, k] * expert(e_k, x) over the selected experts of every row.

    x [N, H]; selected, weights [N, K]; expert(index, rows [n, H]) -> [n, H_out]. A `weighted` expert applies its
    rows' weights [n, 1] itself, inside its computation: expert(index, rows, weights).
    """
    out = None
    for index in torch.unique(selected).tolist():
        rows, slot = torch.where(selected == index)
        if weighted:
            y = expert(index, x[rows], weights[rows, slot, None])
        else:
            y = expert(index, x[rows]) * weights[rows, slot, None]
        if out is None:
            out = torch.zeros(x.shape[0], y.shape[-1], dtype=F32, device=x.device)
        out.index_add_(0, rows, y)
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
