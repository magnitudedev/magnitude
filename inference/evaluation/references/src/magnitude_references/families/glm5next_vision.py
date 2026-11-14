"""Independent float32 reference of GLM-5.3-Flash's vision tower and merger, from a `glm5v` projector GGUF
(`mmproj-*.gguf`, llama.cpp v0.9.0's converter), for one image.

Model definition: transformers 5.17.0 `Glm5NextVisionModel` (`models/glm5_next/modeling_glm5_next.py`), in float32 with
no activation roundings:

- the patch convolution over the image's two identical temporal copies (the two stored kernels, one per frame) plus its
  bias, no position table;
- 24 pre-norm blocks: RMSNorm, fused `qkv` with bias, RMSNorm of each query and key head, 2-D rotary (axial: row and
  column coordinates, `theta` 10000, layout [row, column, row, column] over the 64-wide head, `rotate_half`), full
  attention over the image's patches at scale 1/sqrt(64), `proj`; RMSNorm and a SwiGLU feed-forward with bias whose
  gate is clamped to at most `swiglu_clamp` and up to [-clamp, clamp];
- a closing RMSNorm; the `downsample` Conv2d (kernel and stride 2) over each 2 x 2 cell, with bias; the merger's `proj`
  (no bias), LayerNorm, erf GELU, and a clamped SwiGLU without bias.

Patch rows arrive cell by cell (2 x 2 cells in raster order, each cell's patches in raster order), as the processor
flattens them. Images this module takes are aligned to whole cells (sides multiples of 28) and inside the processor's
pixel bounds, so the released processor neither resizes nor pads them.
"""
from __future__ import annotations

import numpy as np
import torch

from magnitude_references.gguf import Package
from magnitude_references.model import F32, layer_norm, linear, rms_norm, silu

# The vision rotary's base: transformers' default, which the projector GGUF does not store (llama.cpp v0.9.0's
# `Glm5NextVisionModel` writes no `clip.vision` rotary key, and gguf-py defines none for it).
ROTARY_BASE = 10_000.0
# The merger's LayerNorm: transformers' `LayerNorm(out_hidden_size)` at PyTorch's default epsilon, not `rms_norm_eps`.
POST_NORM_EPSILON = 1e-5


def patches(image: np.ndarray, mean: list[float], std: list[float], patch: int, merge: int,
            frames: int = 2) -> tuple[np.ndarray, tuple[int, int]]:
    """The patch rows `[h * w, 3 * frames * patch^2]` and the grid `(h, w)` of an RGB image `[H, W, 3]` of 8-bit values
    whose sides are whole cells: transformers' `Glm5NextImageProcessor.patchify` of the normalized image."""
    height, width, _ = image.shape
    cell = patch * merge
    if height % cell or width % cell:
        raise ValueError(f"image sides must be multiples of {cell}, got {height} x {width}")
    pixels = (image.astype(np.float32) / 255.0 - np.array(mean, np.float32)) / np.array(std, np.float32)
    pixels = pixels.transpose(2, 0, 1)  # [C, H, W]
    gh, gw = height // patch, width // patch
    blocks = pixels.reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
    blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)  # cell row, cell column, row in cell, column in cell, C, y, x
    rows = np.broadcast_to(blocks[:, :, :, :, :, None], (*blocks.shape[:5], frames, patch, patch))
    return np.ascontiguousarray(rows.reshape(gh * gw, 3 * frames * patch * patch)), (gh, gw)


def coordinates(grid: tuple[int, int], merge: int) -> torch.Tensor:
    """Each patch row's `(row, column)` in the cell-by-cell order."""
    gh, gw = grid
    rows = []
    for cell_row in range(gh // merge):
        for cell_column in range(gw // merge):
            for inner_row in range(merge):
                for inner_column in range(merge):
                    rows.append((cell_row * merge + inner_row, cell_column * merge + inner_column))
    return torch.tensor(rows, dtype=F32)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary(grid: tuple[int, int], merge: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin `[rows, width]`: the axial layout [row, column, row, column] of `width / 4` frequencies each."""
    spatial = width // 2
    inverse = 1.0 / (ROTARY_BASE ** (torch.arange(0, spatial, 2, dtype=F32) / spatial))
    angles = coordinates(grid, merge)[:, :, None] * inverse  # [rows, 2, width / 4]
    flat = angles.reshape(angles.shape[0], -1)
    flat = torch.cat([flat, flat], dim=-1)
    return flat.cos(), flat.sin()


class Glm5NextVisionReference:
    """The tower and merger over one projector file's tensors, in F32."""

    def __init__(self, package: Package):
        if package.metadata["clip.projector_type"].value != "glm5v":
            raise ValueError("not a glm5v projector")
        key = lambda suffix: package.metadata[f"clip.vision.{suffix}"].value  # noqa: E731
        self.package = package
        self.depth, self.heads = key("block_count"), key("attention.head_count")
        self.patch, self.merge = key("patch_size"), key("spatial_merge_size")
        self.eps, self.limit = key("attention.layer_norm_epsilon"), key("swiglu_clamp")
        self.mean, self.std = list(key("image_mean")), list(key("image_std"))

    def t(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.package.tensor(name)).to(F32)

    def clamped_swiglu(self, x: torch.Tensor, gate: str, up: str, down: str, biased: bool) -> torch.Tensor:
        bias = (lambda name: self.t(f"{name}.bias")) if biased else (lambda name: None)
        g = linear(x, self.t(f"{gate}.weight"), bias(gate)).clamp(max=self.limit)
        u = linear(x, self.t(f"{up}.weight"), bias(up)).clamp(-self.limit, self.limit)
        return linear(silu(g) * u, self.t(f"{down}.weight"), bias(down))

    def tower(self, rows: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """The tower's closing-norm rows `[h * w, hidden]` of patch rows `[h * w, 3 * 2 * patch^2]`."""
        patch, hidden = self.patch, self.t("v.patch_embd.bias").shape[0]
        x = rows.reshape(-1, 3, 2, patch, patch)
        frames = torch.stack([self.t("v.patch_embd.weight"), self.t("v.patch_embd.weight.1")], dim=2)
        h = torch.einsum("ncfyx,ocfyx->no", x, frames) + self.t("v.patch_embd.bias")
        width = hidden // self.heads
        cos, sin = rotary(grid, self.merge, width)
        cos, sin = cos[:, None, :], sin[:, None, :]
        for block in range(self.depth):
            p = f"v.blk.{block}."
            n = rms_norm(h, self.t(p + "ln1.weight"), self.eps)
            qkv = linear(n, self.t(p + "attn_qkv.weight"), self.t(p + "attn_qkv.bias"))
            q, k, v = qkv.reshape(-1, 3, self.heads, width).unbind(1)
            q = rms_norm(q, self.t(p + "attn_q_norm.weight"), self.eps)
            k = rms_norm(k, self.t(p + "attn_k_norm.weight"), self.eps)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            scores = torch.einsum("qhd,khd->hqk", q, k) * width ** -0.5
            attended = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(-1, hidden)
            h = h + linear(attended, self.t(p + "attn_out.weight"), self.t(p + "attn_out.bias"))
            n = rms_norm(h, self.t(p + "ln2.weight"), self.eps)
            h = h + self.clamped_swiglu(n, p + "ffn_gate", p + "ffn_up", p + "ffn_down", biased=True)
        return rms_norm(h, self.t("v.post_ln.weight"), self.eps)

    def merged(self, h: torch.Tensor) -> torch.Tensor:
        """Image features `[cells, out]` of the tower's rows."""
        g = self.merge * self.merge
        cells = h.reshape(-1, g, h.shape[-1])  # [cells, 2 * 2, hidden], raster order within the cell
        weight = self.t("mm.patch_merger.weight")  # [out, hidden, 2, 2]
        x = torch.einsum("ncs,ocs->no", cells.transpose(1, 2), weight.reshape(*weight.shape[:2], g)) \
            + self.t("mm.patch_merger.bias")
        x = linear(x, self.t("mm.model.fc.weight"))
        x = torch.nn.functional.gelu(layer_norm(x, self.t("mm.post_norm.weight"), self.t("mm.post_norm.bias"), POST_NORM_EPSILON))
        return self.clamped_swiglu(x, "mm.gate", "mm.up", "mm.down", biased=False)

    def features(self, image: np.ndarray) -> torch.Tensor:
        """The image features `[cells, out]` of an RGB image `[H, W, 3]` of 8-bit values."""
        rows, grid = patches(image, self.mean, self.std, self.patch, self.merge)
        with torch.no_grad():
            return self.merged(self.tower(torch.from_numpy(rows), grid))
