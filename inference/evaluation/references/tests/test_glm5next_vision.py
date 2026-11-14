"""The `glm5v` vision reference against transformers 5.17.0's `Glm5NextVisionModel` on the same small random weights,
written in the converter's tensor names and layout (llama.cpp v0.9.0 `glm5v`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

from magnitude_references.gguf import Value, Writer
from magnitude_references.families import open_projector

DEPTH, HIDDEN, HEADS, FF, OUT, MERGER_FF, PATCH, MERGE, LIMIT = 2, 32, 2, 24, 16, 20, 4, 2, 0.8
# An RMS epsilon unlike the merger LayerNorm's fixed 1e-5, so a reference that used it there would disagree.
RMS_EPSILON = 1e-3


def write_tiny_projector(path: Path, seed: int = 0) -> dict[str, np.ndarray]:
    """A random projector file; returns its tensors by name (NumPy order)."""
    rng = np.random.default_rng(seed)
    u32, f32 = (lambda v: Value(V.UINT32, v)), (lambda v: Value(V.FLOAT32, v))
    metadata = {
        "general.architecture": Value(V.STRING, "clip"), "clip.projector_type": Value(V.STRING, "glm5v"),
        "clip.vision.block_count": u32(DEPTH), "clip.vision.attention.head_count": u32(HEADS),
        "clip.vision.patch_size": u32(PATCH), "clip.vision.spatial_merge_size": u32(MERGE),
        "clip.vision.attention.layer_norm_epsilon": f32(RMS_EPSILON), "clip.vision.swiglu_clamp": f32(LIMIT),
        "clip.vision.image_mean": Value(V.ARRAY, [0.5, 0.4, 0.3], V.FLOAT32),
        "clip.vision.image_std": Value(V.ARRAY, [0.25, 0.3, 0.35], V.FLOAT32),
    }
    tensors: dict[str, np.ndarray] = {}

    def weight(name: str, *shape: int, scale: float = 1.0, mean: float = 0.0) -> None:
        tensors[name] = (mean + rng.standard_normal(shape) * scale / np.sqrt(shape[-1])).astype(np.float32)

    for frame in ("", ".1"):
        weight(f"v.patch_embd.weight{frame}", HIDDEN, 3, PATCH, PATCH)
    weight("v.patch_embd.bias", HIDDEN, scale=0.1)
    for b in range(DEPTH):
        p = f"v.blk.{b}."
        for norm in ("ln1", "ln2"):
            weight(p + norm + ".weight", HIDDEN, scale=0.1, mean=1.0)
        weight(p + "attn_qkv.weight", 3 * HIDDEN, HIDDEN)
        weight(p + "attn_qkv.bias", 3 * HIDDEN, scale=0.1)
        for norm in ("attn_q_norm", "attn_k_norm"):
            weight(p + norm + ".weight", HIDDEN // HEADS, scale=0.1, mean=1.0)
        weight(p + "attn_out.weight", HIDDEN, HIDDEN)
        weight(p + "attn_out.bias", HIDDEN, scale=0.1)
        for part, shape in (("gate", (FF, HIDDEN)), ("up", (FF, HIDDEN)), ("down", (HIDDEN, FF))):
            weight(p + f"ffn_{part}.weight", *shape, scale=3.0)
            weight(p + f"ffn_{part}.bias", shape[0], scale=0.3)
    weight("v.post_ln.weight", HIDDEN, scale=0.1, mean=1.0)
    weight("mm.patch_merger.weight", OUT, HIDDEN, MERGE, MERGE)
    weight("mm.patch_merger.bias", OUT, scale=0.1)
    weight("mm.model.fc.weight", OUT, OUT)
    weight("mm.post_norm.weight", OUT, scale=0.1, mean=1.0)
    weight("mm.post_norm.bias", OUT, scale=0.1)
    for part, shape in (("gate", (MERGER_FF, OUT)), ("up", (MERGER_FF, OUT)), ("down", (OUT, MERGER_FF))):
        weight(f"mm.{part}.weight", *shape, scale=3.0)
    writer = Writer(path, metadata)
    for name, values in tensors.items():
        writer.declare(name, values.shape[::-1], Q.F32)
    writer.begin()
    for name, values in tensors.items():
        writer.write(name, values)
    writer.close()
    return tensors


def test_vision_reference_matches_transformers(tmp_path):
    pytest.importorskip("transformers", reason="the references' `check` extra")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    tensors = write_tiny_projector(tmp_path / "mmproj.gguf")
    reference = open_projector(tmp_path / "mmproj.gguf")
    config = Glm5NextVisionConfig(
        depth=DEPTH, hidden_size=HIDDEN, num_heads=HEADS, intermediate_size=FF, out_hidden_size=OUT,
        projection_intermediate_size=MERGER_FF, patch_size=PATCH, spatial_merge_size=MERGE, swiglu_limit=LIMIT,
        rms_norm_eps=RMS_EPSILON, rope_parameters={"rope_type": "axial", "rope_theta": 10_000.0})
    model = Glm5NextVisionModel(config).float().eval()
    t = {name: torch.from_numpy(value) for name, value in tensors.items()}
    state = {
        "patch_embed.proj.weight": torch.stack([t["v.patch_embd.weight"], t["v.patch_embd.weight.1"]], dim=2),
        "patch_embed.proj.bias": t["v.patch_embd.bias"], "post_layernorm.weight": t["v.post_ln.weight"],
        "downsample.weight": t["mm.patch_merger.weight"], "downsample.bias": t["mm.patch_merger.bias"],
        "merger.proj.weight": t["mm.model.fc.weight"], "merger.post_projection_norm.weight": t["mm.post_norm.weight"],
        "merger.post_projection_norm.bias": t["mm.post_norm.bias"],
    }
    for part in ("gate", "up", "down"):
        state[f"merger.{part}_proj.weight"] = t[f"mm.{part}.weight"]
    for b in range(DEPTH):
        p, h = f"v.blk.{b}.", f"blocks.{b}."
        for ours, theirs in (("ln1.weight", "norm1.weight"), ("ln2.weight", "norm2.weight"),
                             ("attn_qkv.weight", "attn.qkv.weight"), ("attn_qkv.bias", "attn.qkv.bias"),
                             ("attn_out.weight", "attn.proj.weight"), ("attn_out.bias", "attn.proj.bias"),
                             ("attn_q_norm.weight", "attn.q_norm.weight"), ("attn_k_norm.weight", "attn.k_norm.weight")):
            state[h + theirs] = t[p + ours]
        for part in ("gate", "up", "down"):
            state[f"{h}mlp.{part}_proj.weight"] = t[f"{p}ffn_{part}.weight"]
            state[f"{h}mlp.{part}_proj.bias"] = t[f"{p}ffn_{part}.bias"]
    model.load_state_dict(state, strict=True)

    height, width = 4 * PATCH * MERGE, 3 * PATCH * MERGE
    image = np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)
    from magnitude_references.families.glm5next_vision import patches
    rows, (gh, gw) = patches(image, reference.mean, reference.std, PATCH, MERGE)
    with torch.no_grad():
        expected = model(torch.from_numpy(rows), torch.tensor([[1, gh, gw]])).pooler_output
    ours = reference.features(image)
    assert ours.shape == expected.shape
    assert (ours - expected).abs().max() <= 1e-4 * expected.abs().max()


def test_a_projector_opens_by_its_type_and_an_unknown_type_is_refused(tmp_path):
    from magnitude_references.families.glm5next_vision import Glm5NextVisionReference
    write_tiny_projector(tmp_path / "mmproj.gguf")
    assert isinstance(open_projector(tmp_path / "mmproj.gguf"), Glm5NextVisionReference)
    other = Writer(tmp_path / "other.gguf", {"general.architecture": Value(V.STRING, "clip"),
                                             "clip.projector_type": Value(V.STRING, "qwen3vl")})
    other.begin()
    other.close()
    with pytest.raises(ValueError, match="no vision reference for projector type 'qwen3vl'"):
        open_projector(tmp_path / "other.gguf")
