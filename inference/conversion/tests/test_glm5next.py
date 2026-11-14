"""GLM-5.3-Flash conversion on a tiny checkpoint in the release's form: byte-exact, sliceable, refusing what it
can't store exactly, and named as upstream's `glm5-next`."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import glm5next_checkpoint as ckpt
import numpy as np
import pytest
import torch
from conversion.base import TextModel, gguf
from conversion.glm import Glm5NextModel

from magnitude_conversion import check, glm5next
from magnitude_conversion.native import MIN_F32_SCALE

URL = "https://huggingface.co/zai-org/GLM-5.3-Flash/tree/46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
T = gguf.GGMLQuantizationType
PINNED_NEGATIVE_EXP = "56a39732947afbc156aeae7684d8c74ca4ec2121ce9afba178088a789ca35d6b"
PINNED_SSM_A = "e52f3488c8402224479be1e25aee87818f05e57ffb52f89248f6eca2bdbf3a96"


@pytest.fixture(autouse=True)
def glm4_pre_tokenizer(monkeypatch):
    # the tiny tokenizer can't hash to a known pre-tokenizer; the release's hashes to glm4 (upstream's comment)
    monkeypatch.setattr(TextModel, "get_vocab_base_pre", lambda self, tokenizer: "glm4")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    return ckpt.write(tmp_path_factory.mktemp("glm") / "checkpoint")


def _convert(checkpoint: Path, out: Path, layers: list[int] | None = None) -> Path:
    with torch.inference_mode():
        glm5next.build("text", checkpoint, out, source_repo_url=URL, layers=layers).write()
    return out


def _check(checkpoint: Path, out: Path, layers: list[int] | None = None) -> dict[str, int]:
    with torch.inference_mode():
        return check.check(glm5next.build("text", checkpoint, out, source_repo_url=URL, layers=layers), out)


def _fields(path: Path) -> dict[str, object]:
    reader = gguf.GGUFReader(path)
    return {name: field.contents() for name, field in reader.fields.items() if not name.startswith("GGUF.")}


@pytest.fixture(scope="module")
def converted(checkpoint, tmp_path_factory) -> Path:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TextModel, "get_vocab_base_pre", lambda self, tokenizer: "glm4")
        return _convert(checkpoint, tmp_path_factory.mktemp("out") / "glm.gguf")


def test_every_tensor_is_byte_exact(checkpoint, converted):
    checked = _check(checkpoint, converted)
    assert set(checked) == {"F8_E4M3", "F32", "BF16"}
    reader = gguf.GGUFReader(converted)
    types = {t.name: t.tensor_type for t in reader.tensors}
    shapes = {t.name: tuple(int(n) for n in reversed(t.shape)) for t in reader.tensors}
    # routed experts stacked beside their stacked F32 tile grid; FP8 linears beside theirs
    assert (types["blk.2.ffn_down_exps.weight"], shapes["blk.2.ffn_down_exps.weight"]) == (T.F8_E4M3, (4, 256, 128))
    assert (types["blk.2.ffn_down_exps.scale"], shapes["blk.2.ffn_down_exps.scale"]) == (T.F32, (4, 2, 1))
    assert (types["blk.3.attn_q_b.scale"], shapes["blk.3.attn_q_b.scale"]) == (T.F32, (2, 1))
    # BF16 stays BF16 where upstream widens it: router, convolutions, indexer heads, mHC and pool-gate weights, norms
    for name in ("blk.2.ffn_gate_inp.weight", "blk.0.ssm_conv1d_q.weight", "blk.3.indexer.proj.weight",
                 "blk.0.hc_attn_fn.weight", "blk.3.indexer_compressor_ape.weight", "blk.0.attn_norm.weight",
                 "blk.3.attn_k_b.weight"):
        assert types[name] == T.BF16, name
    for name in ("blk.0.ssm_a", "blk.0.hc_attn_base.weight", "blk.2.exp_probs_b.bias"):
        assert types[name] == T.F32, name
    fields = _fields(converted)
    # the MTP layer is the last block, in upstream's nextn layout
    assert fields["glm5-next.nextn_predict_layers"] == 1 and fields["glm5-next.block_count"] == 5
    assert any(name.startswith("blk.4.") for name in types)
    assert fields["general.alignment"] == 4096
    assert fields["general.source.repo_url"] == URL
    assert fields["general.file_type"] == gguf.LlamaFileType.MOSTLY_F8_E4M3


@pytest.mark.parametrize("tensor", ["blk.2.ffn_gate_exps.scale", "blk.0.ssm_a", "blk.3.attn_k_b.weight"])
def test_the_check_finds_one_changed_byte(checkpoint, converted, tmp_path, tensor):
    changed = tmp_path / "changed.gguf"
    changed.write_bytes(converted.read_bytes())
    (t,) = [t for t in gguf.GGUFReader(changed).tensors if t.name == tensor]
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


def test_a_slice_converts_like_a_model(checkpoint, tmp_path):
    out = _convert(checkpoint, tmp_path / "slice.gguf", layers=[0, 3])
    checked = _check(checkpoint, out, layers=[0, 3])
    assert set(checked) == {"F8_E4M3", "F32", "BF16"}
    fields = _fields(out)
    assert fields["glm5-next.block_count"] == 2 and "glm5-next.nextn_predict_layers" not in fields  # no MTP layer
    assert fields["glm5-next.leading_dense_block_count"] == 1
    assert fields["glm5-next.attention.head_count_kv"] == [0, 1]
    assert fields["glm5-next.attention.indexer.types"] == [True, True]
    names = {t.name for t in gguf.GGUFReader(out).tensors}
    # source layer 3 (DSA, MoE) is layer 1; source layer 0 (KDA, dense) stays layer 0
    assert {"blk.1.attn_k_b.weight", "blk.1.ffn_up_exps.scale", "blk.0.ssm_a", "blk.0.ffn_up.scale"} <= names
    assert not any(name.startswith("blk.2.") for name in names)
    # a slice's bytes are the whole conversion's, layer for layer
    whole = {t.name: np.asarray(t.data) for t in gguf.GGUFReader(_convert(checkpoint, tmp_path / "w.gguf")).tensors}
    for t in gguf.GGUFReader(out).tensors:
        source = t.name.replace("blk.1.", "blk.3.")
        assert np.array_equal(np.asarray(t.data), whole[source]), t.name


def _values(tensor) -> np.ndarray:
    """A tensor's stored values in F32, numpy order."""
    raw = np.asarray(tensor.data)
    if tensor.tensor_type == T.BF16:
        return gguf.quants.dequantize(raw, T.BF16)
    if tensor.tensor_type == T.F8_E4M3:
        return torch.from_numpy(raw.view(np.uint8).copy()).view(torch.float8_e4m3fn).float().numpy()
    return raw


def test_upstreams_f32_slice_holds_the_slices_values(checkpoint, tmp_path):
    ours = _convert(checkpoint, tmp_path / "slice.gguf", layers=[0, 3])
    with torch.inference_mode():
        glm5next.upstream_f32(checkpoint, tmp_path / "f32.gguf", layers=[0, 3]).write()
    native = {t.name: t for t in gguf.GGUFReader(ours).tensors}
    wide = {t.name: t for t in gguf.GGUFReader(tmp_path / "f32.gguf").tensors}
    assert set(wide) == {name for name in native if not name.endswith(".scale")}
    for name, tensor in wide.items():
        assert tensor.tensor_type == T.F32, name
        expected = _values(native[name])
        if native[name].tensor_type == T.F8_E4M3:  # each code times its tile's scale, one F32 product
            scale = _values(native[name.removesuffix(".weight") + ".scale"])
            expected = expected * scale.repeat(128, axis=-2).repeat(128, axis=-1)
        assert np.array_equal(_values(tensor).reshape(expected.shape), expected), name


def test_a_slice_keeps_a_shared_indexers_source():
    tc = ckpt.config()["text_config"]
    tc["indexer_types"] = ["full", "full", "full", "shared"]
    with pytest.raises(ValueError, match="reuses layer 2"):
        glm5next.Glm5NextSlice([0, 3], 4).remap_config(tc)
    assert glm5next.Glm5NextSlice([2, 3], 4).remap_config(tc)["indexer_types"] == ["full", "shared"]


def _set_scale(name: str, value: float):
    def edit(t: dict[str, torch.Tensor]) -> None:
        t[name] = t[name].clone()
        t[name][0, 0] = value
    return edit


def _drop_layer_1(t: dict[str, torch.Tensor]) -> None:
    for name in [n for n in t if ".layers.1." in n]:
        del t[name]


def _partial_tile(t: dict[str, torch.Tensor]) -> None:
    ckpt._fp8(t, "model.language_model.layers.0.mlp.up_proj", ckpt.DENSE_FF, 192)


@pytest.mark.parametrize("edit_tensors, edit_config, refusal", [
    (_partial_tile, None, r"not whole \(128, 128\) blocks"),
    (_drop_layer_1, None, r"layers \[1\] have no tensors"),
    (_set_scale("model.language_model.layers.3.self_attn.o_proj.weight_scale_inv", math.inf), None, "not finite"),
    (_set_scale("model.language_model.layers.2.mlp.experts.1.up_proj.weight_scale_inv", math.nan), None,
     "not finite"),
    (_set_scale("model.language_model.layers.0.mlp.down_proj.weight_scale_inv", MIN_F32_SCALE / 2), None,
     "below 2\\^-117"),
    (None, lambda c: c["text_config"].update(layer_norm_eps=1e-5), "upstream's conversion writes 1e-06"),
    (None, lambda c: c["text_config"].update(mlp_layer_types=["dense", "sparse", "dense", "sparse"]),
     "first_k_dense_replace"),
], ids=["partial-tile", "missing-layer", "inf-scale", "nan-expert-scale", "subnormal-product-scale",
        "layer-norm-eps", "dense-layers-not-leading"])
def test_refusals(tmp_path, edit_tensors, edit_config, refusal):
    checkpoint = ckpt.write(tmp_path / "checkpoint", edit_config=edit_config, edit_tensors=edit_tensors)
    with pytest.raises(ValueError, match=refusal):
        _convert(checkpoint, tmp_path / "refused.gguf")
    assert not (tmp_path / "refused.gguf").exists()


def test_the_smallest_scale_and_zero_are_kept(tmp_path):
    name = "model.language_model.layers.0.mlp.down_proj.weight_scale_inv"

    def edit(t: dict[str, torch.Tensor]) -> None:
        t[name] = t[name].clone()
        t[name][0, 0], t[name][0, 1] = MIN_F32_SCALE, 0.0
    checkpoint = ckpt.write(tmp_path / "checkpoint", edit_tensors=edit)
    _check(checkpoint, _convert(checkpoint, tmp_path / "glm.gguf"))


class _Upstream(Glm5NextModel):
    """Upstream's converter as it stands: the text model and its MTP layer."""
    model_arch = gguf.MODEL_ARCH.GLM5_NEXT


def test_names_and_keys_are_upstreams(checkpoint, converted, tmp_path):
    upstream_out = tmp_path / "upstream.gguf"
    with torch.inference_mode(), glm5next._tokenizer_from_json(checkpoint):
        _Upstream(checkpoint, gguf.LlamaFileType.MOSTLY_BF16, upstream_out).write()
    ours, theirs = gguf.GGUFReader(converted), gguf.GGUFReader(upstream_out)
    our_names = {t.name for t in ours.tensors}
    assert {n for n in our_names if not n.endswith(".scale")} == {t.name for t in theirs.tensors}
    assert {n.removesuffix(".scale") + ".weight" for n in our_names if n.endswith(".scale")} <= our_names
    ours_kv, theirs_kv = _fields(converted), _fields(upstream_out)
    added = {"general.alignment", "general.source.repo_url"}
    assert set(ours_kv) == set(theirs_kv) | added
    differ = {k for k in theirs_kv if ours_kv[k] != theirs_kv[k]}
    assert differ == {"general.file_type"}  # FP8 kept rather than widened to BF16
    assert theirs_kv["general.architecture"] == "glm5-next"


def test_negative_exp_is_the_correctly_rounded_f32_on_every_host():
    from magnitude_conversion.exact import negative_exp_f32
    x = np.array([0.0, 1.0, -1.0, 0.5, 2.0, -3.25, 4.0, 0.001, 2.7725887], dtype=np.float32)
    got = negative_exp_f32(x)
    assert got.dtype == np.float32 and got.shape == x.shape
    # -exp(1) = -2.718281828459045…: the F32 nearest is 0xC02DF854 (negated)
    assert got[1].view(np.uint32) == 0xC02DF854
    assert got[0] == -1.0
    assert hashlib.sha256(got.tobytes()).hexdigest() == PINNED_NEGATIVE_EXP


def test_a_converted_ssm_a_has_the_same_bytes_on_every_host(converted):
    reader = gguf.GGUFReader(converted)
    digest = hashlib.sha256()
    for tensor in sorted((t for t in reader.tensors if t.name.endswith(".ssm_a")), key=lambda t: t.name):
        digest.update(np.asarray(tensor.data).tobytes())
    assert digest.hexdigest() == PINNED_SSM_A


def test_two_converters_in_one_process_do_not_share_their_main_layer_count():
    # upstream's filter tells the MTP layer by the converter's `_n_main_layers`; one converter's count must not
    # reach another's, nor the class's
    from magnitude_conversion.glm5next import Glm5NextText
    short, long = object.__new__(Glm5NextText), object.__new__(Glm5NextText)
    short._n_main_layers, long._n_main_layers = 3, 5
    for converter in (short, long):
        converter.mtp_only = True  # keeps only the MTP layer's tensors
    item = ("model.layers.4.mlp.gate_proj.weight", lambda: torch.zeros(1))
    assert short.filter_tensors(item) is not None, "layer 4 is past three main layers"
    assert long.filter_tensors(item) is None, "layer 4 is a main layer of five"
    assert Glm5NextText._n_main_layers is None
