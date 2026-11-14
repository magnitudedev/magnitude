"""GLM-5.3-Flash conversion of a tiny checkpoint in NVIDIA's ModelOpt form (`nvidia/GLM-5.3-Flash-FP4-Local`): NVFP4
routed experts with compressed block scales, MXFP8 and FP8_PB_WO grids side by side in one file, the W4A16 MTP layer
kept; byte-exact, sliceable, its NVFP4 blocks the ggml layout of the source's values, refusing what it can't store
exactly."""

from __future__ import annotations

import math
from pathlib import Path

import glm5next_checkpoint as ckpt
import numpy as np
import pytest
import torch
from conversion.base import TextModel, gguf

from magnitude_conversion import check, csf, glm5next

URL = "https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"
T = gguf.GGMLQuantizationType
LM = "model.language_model"


@pytest.fixture(autouse=True)
def glm4_pre_tokenizer(monkeypatch):
    # the tiny tokenizer can't hash to a known pre-tokenizer; the release's hashes to glm4 (upstream's comment)
    monkeypatch.setattr(TextModel, "get_vocab_base_pre", lambda self, tokenizer: "glm4")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    return ckpt.write(tmp_path_factory.mktemp("glm") / "checkpoint", form="modelopt")


def _convert(checkpoint: Path, out: Path, layers: list[int] | None = None) -> Path:
    with torch.inference_mode():
        glm5next.build("text", checkpoint, out, source_repo_url=URL, layers=layers).write()
    return out


def _check(checkpoint: Path, out: Path, layers: list[int] | None = None) -> dict[str, int]:
    with torch.inference_mode():
        return check.check(glm5next.build("text", checkpoint, out, source_repo_url=URL, layers=layers), out)


@pytest.fixture(scope="module")
def converted(checkpoint, tmp_path_factory) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TextModel, "get_vocab_base_pre", lambda self, tokenizer: "glm4")
        return _convert(checkpoint, tmp_path_factory.mktemp("out") / "glm.gguf")


def _tensors(path: Path) -> dict[str, gguf.ReaderTensor]:
    return {t.name: t for t in gguf.GGUFReader(path).tensors}


def _form(t: gguf.ReaderTensor) -> tuple[T, tuple[int, ...]]:
    return t.tensor_type, tuple(int(n) for n in reversed(t.shape))


def test_every_tensor_is_byte_exact(checkpoint, converted):
    checked = _check(checkpoint, converted)
    assert set(checked) == {"NVFP4", "F8_E4M3", "I8", "F32", "BF16"}
    t = _tensors(converted)
    # routed experts: stacked NVFP4, one second-level and one activation scale per expert
    assert _form(t["blk.2.ffn_down_exps.weight"]) == (T.NVFP4, (4, 256, 128))
    assert _form(t["blk.2.ffn_down_exps.scale"]) == (T.F32, (4,))
    assert _form(t["blk.2.ffn_gate_exps.input_scale"]) == (T.F32, (4,))
    # the MTP layer's experts are W4A16: no activation scale
    assert _form(t["blk.4.ffn_up_exps.weight"]) == (T.NVFP4, (4, 128, 256))
    assert "blk.4.ffn_up_exps.input_scale" not in t
    # MXFP8 beside its I8 UE8M0 grid per 1x32, FP8_PB_WO beside its F32 grid per 128x128, in one file
    assert _form(t["blk.0.ffn_up.weight"]) == (T.F8_E4M3, (256, 256))
    assert _form(t["blk.0.ffn_up.scale"]) == (T.I8, (256, 8))
    assert _form(t["blk.4.attn_q_a.scale"]) == (T.I8, (128, 8))
    assert _form(t["blk.3.attn_q_b.scale"]) == (T.F32, (2, 1))
    assert _form(t["blk.2.ffn_up_shexp.scale"]) == (T.F32, (1, 2))
    assert _form(t["output.scale"]) == (T.I8, (ckpt.VOCAB, 8))
    # the MTP layer's MXFP8 kv_b, split as upstream splits it: v_b MXFP8 verbatim, k_b exactly in BF16
    assert _form(t["blk.4.attn_v_b.weight"]) == (T.F8_E4M3, (2, 128, 128))
    assert _form(t["blk.4.attn_v_b.scale"]) == (T.I8, (2, 128, 4))
    assert _form(t["blk.4.attn_k_b.weight"]) == (T.BF16, (2, 128, 128))
    assert "blk.4.attn_k_b.scale" not in t
    reader = gguf.GGUFReader(converted)
    fields = {name: f.contents() for name, f in reader.fields.items()}
    assert fields["general.file_type"] == gguf.LlamaFileType.MOSTLY_NVFP4
    assert "magnitude.scale_encoding" not in fields
    # upstream's key for weights calibrated without FP4 activations: the MTP layer's experts
    prec_a4 = dict(zip(fields["general.tensor_extra.name"], fields["general.tensor_extra.prec_a4"]))
    assert prec_a4 == {f"blk.4.ffn_{p}_exps.weight": False for p in ("down", "gate", "up")}


def _e2m1(code: np.ndarray) -> np.ndarray:
    magnitude = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)[code & 7]
    return np.where(code & 8, -magnitude, magnitude)


def test_nvfp4_blocks_hold_the_sources_values(checkpoint, converted):
    """ggml's own NVFP4 decoding of the file equals each source code times its E4M3 block scale (the second-level
    scale is applied beside it): the layout is ggml's, independently of the converter and the check."""
    source = ckpt.tensors("modelopt")[0]
    t = _tensors(converted)["blk.2.ffn_gate_exps.weight"]
    stored = gguf.quants.dequantize(np.asarray(t.data), T.NVFP4).reshape(4, ckpt.MOE_FF, ckpt.HIDDEN)
    for e in range(4):
        name = f"{LM}.layers.2.mlp.experts.{e}.gate_proj"
        packed = source[f"{name}.weight"].numpy()
        codes = np.stack((packed & 0xF, packed >> 4), -1).reshape(ckpt.MOE_FF, ckpt.HIDDEN)
        streams = {s: source[csf.stream_name(f"{name}.weight_scale", s)].numpy() for s in csf.STREAMS}
        scales = torch.from_numpy(csf.decode(name, ckpt.MOE_FF, ckpt.HIDDEN // 16, streams))
        values = _e2m1(codes) * np.repeat(scales.view(torch.float8_e4m3fn).float().numpy(), 16, axis=1)
        assert np.array_equal(stored[e], values), e


def test_k_b_holds_the_sources_products(converted):
    """The MTP layer's k_b: each value, as BF16 holds it, is the source code times 2^(b - 127), exactly."""
    source = ckpt.tensors("modelopt")[0]
    name = f"{LM}.layers.4.self_attn.kv_b_proj"
    codes = source[f"{name}.weight"].float().reshape(ckpt.HEADS, ckpt.NOPE + ckpt.V_HEAD, ckpt.KV_LORA)
    grid = source[f"{name}.weight_scale"].reshape(ckpt.HEADS, ckpt.NOPE + ckpt.V_HEAD, ckpt.KV_LORA // 32)
    products = codes * torch.pow(2.0, grid.float().repeat_interleave(32, -1) - 127)
    t = _tensors(converted)["blk.4.attn_k_b.weight"]
    stored = torch.from_numpy(np.asarray(t.data).copy()).view(torch.bfloat16).float()
    assert torch.equal(stored.reshape(ckpt.HEADS, ckpt.KV_LORA, ckpt.NOPE), products[:, :ckpt.NOPE].transpose(1, 2))


@pytest.mark.parametrize("tensor", ["blk.2.ffn_down_exps.weight", "blk.4.ffn_gate_exps.scale",
                                    "blk.3.ffn_up_exps.input_scale", "blk.0.ffn_up.scale"])
def test_the_check_finds_one_changed_byte(checkpoint, converted, tmp_path, tensor):
    changed = tmp_path / "changed.gguf"
    changed.write_bytes(converted.read_bytes())
    t = _tensors(changed)[tensor]
    with open(changed, "r+b") as f:
        f.seek(int(t.data_offset) + 1)
        byte = f.read(1)[0]
        f.seek(-1, 1)
        f.write(bytes([byte ^ 0x01]))
    with pytest.raises(ValueError, match=f"{tensor}: bytes differ"):
        _check(checkpoint, changed)


def test_tensors_are_layer_major(converted):
    layers = [int(t.name.split(".")[1]) if t.name.startswith("blk.") else -1
              for t in gguf.GGUFReader(converted).tensors]
    assert layers == sorted(layers)


def test_one_routed_layer_converts_like_the_model(checkpoint, converted, tmp_path):
    out = _convert(checkpoint, tmp_path / "slice.gguf", layers=[3])
    assert set(_check(checkpoint, out, layers=[3])) == {"NVFP4", "F8_E4M3", "I8", "F32", "BF16"}
    whole = _tensors(converted)
    for name, t in _tensors(out).items():
        assert np.array_equal(np.asarray(t.data), np.asarray(whole[name.replace("blk.0.", "blk.3.")].data)), name


def _drop(name: str):
    def edit(t: dict[str, torch.Tensor]) -> None:
        del t[name]
    return edit


def _set_byte(name: str, index: int, value: int):
    def edit(t: dict[str, torch.Tensor]) -> None:
        t[name] = t[name].clone()
        t[name].view(-1).view(torch.uint8)[index] = value
    return edit


def _set(name: str, value: float):
    def edit(t: dict[str, torch.Tensor]) -> None:
        t[name] = torch.tensor(value)
    return edit


def _truncate(name: str):
    def edit(t: dict[str, torch.Tensor]) -> None:
        t[name] = t[name][:-1].clone()
    return edit


def _below_bf16(t: dict[str, torch.Tensor]) -> None:
    """The MTP layer's kv_b: its smallest subnormal code under grid byte 1, 2^-9 x 2^-126, below BF16's range."""
    name = f"{LM}.layers.4.self_attn.kv_b_proj"
    t[f"{name}.weight"] = t[f"{name}.weight"].clone()
    t[f"{name}.weight"].view(torch.uint8)[0, 0] = 0x01
    t[f"{name}.weight_scale"] = t[f"{name}.weight_scale"].clone()
    t[f"{name}.weight_scale"][0, 0] = 1


EXPERT = f"{LM}.layers.2.mlp.experts.1.up_proj"
MTP_EXPERT = f"{LM}.layers.4.mlp.experts.0.down_proj"


@pytest.mark.parametrize("edit_tensors, edit_config, refusal", [
    (_drop(f"{EXPERT}.weight_scale.csf32_offsets"), None, "weight_scale.csf32_offsets is missing"),
    (_drop(f"{EXPERT}.input_scale"), None, r"layers\.2\.mlp\.experts\.1\.up_proj\.input_scale is missing"),
    (_drop(f"{MTP_EXPERT}.weight_scale_2"), None, r"layers\.4\.mlp\.experts\.0\.down_proj\.weight_scale_2 is missing"),
    (_truncate(f"{EXPERT}.weight_scale.csf32_codes"), None, r"csf32_codes is uint8 \[1023\]"),
    (_set(f"{EXPERT}.weight_scale_2", math.inf), None, "inf, not a positive finite F32 value"),
    (_set(f"{EXPERT}.input_scale", 0.0), None, "input_scale: 0.0, not a positive finite F32 value"),
    (_set(f"{MTP_EXPERT}.input_scale", 1.0), None, "W4A16, yet the checkpoint stores"),
    (_set_byte(f"{LM}.layers.0.mlp.up_proj.weight_scale", 0, 250), None, "UE8M0 byte 250 is above 246"),
    (_below_bf16, None, r"blk\.4\.attn_k_b\.weight: 0\.001953125 x 2\^-126 = .* is not a BF16 value"),
    (None, lambda c: c["quantization_config"]["quantized_layers"].update(
        {f"{LM}.layers.1.mlp.experts": ckpt.ENTRIES["nvfp4"]}), "names .*layers.1.mlp.experts, which has no weight"),
    (None, lambda c: c["quantization_config"]["quantized_layers"].update(
        {f"{LM}.layers.0.mlp.up_proj": {"quant_algo": "INT4_AWQ"}}), "no rule for quant_algo 'INT4_AWQ'"),
    (None, lambda c: c["quantization_config"]["quantized_layers"].pop(f"{LM}.layers.0.mlp.up_proj"),
     r"layers\.0\.mlp\.up_proj\.weight: torch\.float8_e4m3fn, which no declared format stores"),
], ids=["missing-stream", "missing-input-scale", "missing-second-level-scale", "stream-size", "inf-second-level-scale",
        "zero-input-scale", "w4a16-with-input-scale", "ue8m0-out-of-range", "k-b-below-bf16", "entry-without-weights",
        "unknown-algorithm", "undeclared-fp8-weight"])
def test_refusals(tmp_path, edit_tensors, edit_config, refusal):
    checkpoint = ckpt.write(tmp_path / "checkpoint", form="modelopt", edit_config=edit_config,
                            edit_tensors=edit_tensors)
    with pytest.raises(ValueError, match=refusal):
        _convert(checkpoint, tmp_path / "refused.gguf")
    assert not (tmp_path / "refused.gguf").exists()


def _exception_out_of_its_block(t: dict[str, torch.Tensor]) -> None:
    name = f"{EXPERT}.weight_scale.csf32_exceptions"
    t[name] = t[name].clone()
    t[name][0] = 64 | 0x40 << 8


def _negative_block_scale(t: dict[str, torch.Tensor]) -> None:
    name = f"{MTP_EXPERT}.weight_scale"
    t[name] = t[name].clone()
    t[name].view(torch.uint8)[3, 2] = 0xB8


@pytest.mark.parametrize("edit_tensors, refusal", [
    (_exception_out_of_its_block, "csf32 exception lies outside its block"),
    (_negative_block_scale, "block scale byte 0xb8 is not a positive E4M3 value"),
], ids=["csf-exception-out-of-range", "negative-block-scale"])
def test_scale_contents_are_refused_when_read(tmp_path, edit_tensors, refusal):
    """Block scales are read when written: the conversion and the check refuse them there, the conversion leaving no
    file."""
    checkpoint = ckpt.write(tmp_path / "checkpoint", form="modelopt", edit_tensors=edit_tensors)
    with pytest.raises(ValueError, match=refusal):
        _convert(checkpoint, tmp_path / "refused.gguf")
    assert not (tmp_path / "refused.gguf").exists()
