"""The `glm5-next` reference on a small random file in the converter's form (upstream llama.cpp names, FP8 weights as
E4M3 codes under F32 tile scales): the F32 tile grid, the row-by-row definition, every tensor read, and transformers
5.17.0's `glm5_next` model on the same weights."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

from conftest import assert_rows_hold, storage_step
from magnitude_references.checks import check_hf
from magnitude_references.families.glm5next import Glm5NextReference
from magnitude_references.gguf import Package, Value, Writer, quantize
from magnitude_references.model import ROW_TILE, Weights
from magnitude_references.perturbation import Perturbation

CPU = torch.device("cpu")
VOCAB, HIDDEN, HEADS, KDA_DIM, CONV, STREAMS, TILE = 64, 64, 2, 16, 4, 4, 16
Q_RANK, KV_RANK, KEY_DIM, VALUE_DIM, INDEX_HEADS, INDEX_DIM, KPOOL, INDEX_TOP = 32, 32, 16, 16, 8, 16, 4, 8
EXPERTS, USED, EXPERT_FFN, DENSE_FFN = 4, 2, 32, 48
# KDA + dense, MLA + dense, KDA + routed, MLA + routed
LINEAR, DENSE = [True, False, True, False], 2
LENGTH = 24  # 6 complete pools against INDEX_TOP / KPOOL = 2 selected: the top pools bind from row 11 on
assert LENGTH // KPOOL > INDEX_TOP // KPOOL
# GLM-5.3-Flash's own selection: 512 pools of KPOOL = 4 rows, so the top pools bind from row 2051 on. A prompt of
# GLM_LENGTH rows crosses row and key tiles both before and after they bind.
GLM_TOP = 2048
GLM_LENGTH = GLM_TOP + 2 * ROW_TILE + 3


def fp8(rng, shape: tuple[int, ...], gain: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 codes and F32 tile scales (amax / 448 per TILE x TILE tile, one grid per stacked matrix) of a random
    matrix or stack: the checkpoint's form."""
    *lead, rows, cols = shape
    values = torch.from_numpy(rng.standard_normal(shape) * gain / math.sqrt(cols)).to(torch.float32)
    tiles = values.view(*lead, rows // TILE, TILE, cols // TILE, TILE)
    scales = tiles.abs().amax((-3, -1)) / 448
    codes = (tiles / scales[..., :, None, :, None]).to(torch.float8_e4m3fn).view(shape)
    return codes.view(torch.uint8).numpy(), scales.numpy()


def write_tiny_glm5next(path: Path, seed: int = 0, index_top: int = INDEX_TOP, mtp: int = 0) -> None:
    """`mtp` trailing blocks are declared as a full file's MTP head is (block_count counts them), holding no tensor:
    the text model's reference never reads them."""
    rng = np.random.default_rng(seed)
    layers = len(LINEAR)
    u32, f32 = (lambda v: Value(V.UINT32, v)), (lambda v: Value(V.FLOAT32, v))
    array = lambda kind, values: Value(V.ARRAY, list(values), kind)  # noqa: E731
    keys = {
        "block_count": u32(layers + mtp), "context_length": u32(4096), "embedding_length": u32(HIDDEN),
        "feed_forward_length": u32(DENSE_FFN), "attention.head_count": u32(HEADS),
        "attention.head_count_kv": array(V.UINT32, [0 if linear else 1 for linear in LINEAR] + [1] * mtp),
        "attention.layer_norm_rms_epsilon": f32(1e-5), "attention.layer_norm_epsilon": f32(1e-6),
        "vocab_size": u32(VOCAB), "ssm.conv_kernel": u32(CONV), "kda.head_dim": u32(KDA_DIM),
        "kda.gate_lower_bound": f32(-5.0), "attention.q_lora_rank": u32(Q_RANK), "attention.kv_lora_rank": u32(KV_RANK),
        "rope.dimension_count": u32(0), "attention.key_length": u32(KV_RANK), "attention.value_length": u32(KV_RANK),
        "attention.key_length_mla": u32(KEY_DIM), "attention.value_length_mla": u32(VALUE_DIM),
        "attention.indexer.head_count": u32(INDEX_HEADS), "attention.indexer.key_length": u32(INDEX_DIM),
        "attention.indexer.top_k": u32(index_top), "attention.indexer.kpool": u32(KPOOL),
        "attention.indexer.kpool_select_tail": Value(V.BOOL, True),
        "hyper_connection.count": u32(STREAMS), "hyper_connection.sinkhorn_iterations": u32(20),
        "hyper_connection.epsilon": f32(1e-6), "leading_dense_block_count": u32(DENSE),
        "expert_count": u32(EXPERTS), "expert_used_count": u32(USED), "expert_group_count": u32(1),
        "expert_group_used_count": u32(1), "expert_feed_forward_length": u32(EXPERT_FFN),
        "expert_shared_count": u32(1), "expert_weights_scale": f32(2.5), "expert_weights_norm": Value(V.BOOL, True),
        "expert_gating_func": u32(2), "swiglu_clamp_exp": array(V.FLOAT32, [1.0] * layers),
        "swiglu_clamp_shexp": array(V.FLOAT32, [1.0] * layers),
        **({"nextn_predict_layers": u32(mtp)} if mtp else {}),
    }
    metadata = {"general.architecture": Value(V.STRING, "glm5-next")}
    metadata |= {f"glm5-next.{key}": value for key, value in keys.items()}
    tensors: list[tuple[str, np.ndarray, Q, tuple[int, ...]]] = []  # name, stored bytes, type, NumPy shape

    def store(name: str, values: np.ndarray, kind: Q = Q.BF16) -> None:
        values = values.astype(np.float32)
        tensors.append((name, values if kind == Q.F32 else quantize(values, kind), kind, values.shape))

    def normal(name: str, shape: tuple[int, ...], kind: Q = Q.BF16, gain: float = 1.0, mean: float = 0.0) -> None:
        store(name, mean + rng.standard_normal(shape) * gain / math.sqrt(shape[-1] if len(shape) > 1 else 1), kind)

    def scaled(name: str, shape: tuple[int, ...], gain: float = 1.0) -> None:
        codes, scales = fp8(rng, shape, gain)
        tensors.extend([(name + ".weight", codes, Q.F8_E4M3, codes.shape), (name + ".scale", scales, Q.F32, scales.shape)])

    def swiglu(prefix: str, suffix: str, width: int, lead: tuple[int, ...] = ()) -> None:
        for part, shape in (("gate", (width, HIDDEN)), ("up", (width, HIDDEN)), ("down", (HIDDEN, width))):
            scaled(f"{prefix}ffn_{part}{suffix}", lead + shape, 0.5 if part == "down" else 1.0)

    normal("token_embd.weight", (VOCAB, HIDDEN), gain=math.sqrt(HIDDEN))
    normal("output_norm.weight", (HIDDEN,), gain=0.1, mean=1.0)
    normal("output.weight", (VOCAB, HIDDEN), gain=4.0)
    mixes, inner = (2 + STREAMS) * STREAMS, HEADS * KDA_DIM
    for layer, linear in enumerate(LINEAR):
        p = f"blk.{layer}."
        for site in ("attn", "ffn"):
            normal(p + f"hc_{site}_fn.weight", (mixes, STREAMS * HIDDEN))
            normal(p + f"hc_{site}_base.weight", (mixes,), Q.F32, 0.5)
            normal(p + f"hc_{site}_scale.weight", (3,), Q.F32, 0.5, 1.0)
        for norm in ("attn_norm", "ffn_norm"):
            normal(p + norm + ".weight", (HIDDEN,), gain=0.1, mean=1.0)
        if linear:
            for part in "qkv":
                normal(p + f"attn_{part}.weight", (inner, HIDDEN))
                normal(p + f"ssm_conv1d_{part}.weight", (1, inner, 1, CONV), gain=0.5)
            normal(p + "attn_output.weight", (HIDDEN, inner), gain=0.5)
            normal(p + "ssm_beta.weight", (HEADS, HIDDEN))
            normal(p + "ssm_f_a.weight", (KDA_DIM, HIDDEN))
            normal(p + "ssm_f_b.weight", (inner, KDA_DIM))
            normal(p + "ssm_g_a.weight", (KDA_DIM, HIDDEN))
            normal(p + "ssm_g_b.weight", (inner, KDA_DIM))
            normal(p + "ssm_norm.weight", (KDA_DIM,), gain=0.1, mean=1.0)
            normal(p + "ssm_dt.bias", (inner,), Q.F32, mean=-3.0)  # decays from about -0.2 to -5 per row
            store(p + "ssm_a", -np.exp(rng.standard_normal(HEADS).astype(np.float32) * 0.5), Q.F32)
        else:
            scaled(p + "attn_q_a", (Q_RANK, HIDDEN))
            normal(p + "attn_q_a_norm.weight", (Q_RANK,), gain=0.1, mean=1.0)
            scaled(p + "attn_q_b", (HEADS * KEY_DIM, Q_RANK), gain=2.0)
            scaled(p + "attn_kv_a_mqa", (KV_RANK, HIDDEN))
            normal(p + "attn_kv_a_norm.weight", (KV_RANK,), gain=0.1, mean=1.0)
            normal(p + "attn_k_b.weight", (HEADS, KV_RANK, KEY_DIM), gain=2.0)
            normal(p + "attn_v_b.weight", (HEADS, VALUE_DIM, KV_RANK))
            scaled(p + "attn_output", (HIDDEN, HEADS * VALUE_DIM), gain=0.5)
            normal(p + "indexer.attn_q_b.weight", (INDEX_HEADS * INDEX_DIM, Q_RANK))
            normal(p + "indexer.attn_k.weight", (INDEX_DIM, HIDDEN))
            normal(p + "indexer.k_norm.weight", (INDEX_DIM,), gain=0.1, mean=1.0)
            normal(p + "indexer.k_norm.bias", (INDEX_DIM,), gain=0.1)
            normal(p + "indexer.proj.weight", (INDEX_HEADS, HIDDEN))
            normal(p + "indexer_compressor_gate.weight", (INDEX_DIM, HIDDEN))
            normal(p + "indexer_compressor_ape.weight", (KPOOL, INDEX_DIM))
        if layer < DENSE:
            swiglu(p, "", DENSE_FFN)
        else:
            normal(p + "ffn_gate_inp.weight", (EXPERTS, HIDDEN))
            normal(p + "exp_probs_b.bias", (EXPERTS,), Q.F32, 0.1)
            swiglu(p, "_exps", EXPERT_FFN, (EXPERTS,))
            swiglu(p, "_shexp", EXPERT_FFN)
    writer = Writer(path, metadata)
    for name, _, kind, shape in tensors:
        writer.declare(name, tuple(reversed(shape)), kind)
    writer.begin()
    for name, data, _, _ in tensors:
        writer.write(name, data)
    writer.close()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("glm5next") / "tiny.gguf"
    write_tiny_glm5next(path)
    return path


@pytest.fixture(scope="module")
def glm_selection(tmp_path_factory) -> Path:
    """The tiny file's widths under GLM-5.3-Flash's own top_k and kpool."""
    path = tmp_path_factory.mktemp("glm5next") / "selection.gguf"
    write_tiny_glm5next(path, index_top=GLM_TOP)
    return path


def reference(path: Path, cache: bool = True, perturbation: Perturbation | None = None,
              layers: int | None = None) -> Glm5NextReference:
    package = Package.local(path)
    return Glm5NextReference(package, Weights(package, CPU, cache, perturbation), layers)


def tokens(seed: int, count: int = 2, length: int = LENGTH) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).integers(0, VOCAB, (count, length)))


def test_f32_tile_scales_are_one_product(tiny: Path) -> None:
    """Each value is its E4M3 code times its tile's F32 scale, rounded once to float32 (the product of a 4-bit and
    a 24-bit significand is exact in float64), for matrices, rows of a matrix, stacked experts and each expert."""
    package = Package.local(tiny)
    weights = Weights(package, CPU, False)
    inexact = 0
    for name in ("blk.1.attn_q_a", "blk.0.ffn_down", "blk.2.ffn_up_exps"):
        info = package.info(name + ".weight")
        codes = torch.from_numpy(np.array(package.raw(name + ".weight"))).view(torch.float8_e4m3fn).double()
        scales = torch.from_numpy(np.array(package.raw(name + ".scale"))).view(torch.float32).double()
        scales = scales.view(package.info(name + ".scale").array_shape)
        exact = codes.view(info.array_shape) * scales.repeat_interleave(TILE, -2).repeat_interleave(TILE, -1)
        value = weights(name + ".weight")
        assert torch.equal(value, exact.float()), name
        inexact += int((value.double() != exact).sum())
        if len(info.shape) == 3:
            for index in range(EXPERTS):
                assert torch.equal(weights.expert(name + ".weight", index), value[index]), (name, index)
        else:
            rows = torch.tensor([[17, 0], [17, 31]])
            assert torch.equal(weights.rows(name + ".weight", rows), value[rows]), name
    assert inexact > 0  # the products are correctly rounded, not all exact


def test_a_non_finite_tile_scale_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "inf.gguf"
    writer = Writer(path, {"general.architecture": Value(V.STRING, "glm5-next")})
    writer.declare("x.weight", (TILE, TILE), Q.F8_E4M3)
    writer.declare("x.scale", (1, 1), Q.F32)
    writer.begin()
    writer.write("x.weight", np.zeros((TILE, TILE), np.uint8))
    writer.write("x.scale", np.array([[np.inf]], np.float32))
    writer.close()
    with pytest.raises(ValueError, match="not finite"):
        Weights(Package.local(path), CPU, False)("x.weight")


def test_a_perturbed_tile_grid_moves_as_f32(tiny: Path) -> None:
    """An F32 tile grid is a stored F32 tensor: the perturbation moves its elements by at most one F32 step, by their
    identity, so a stacked expert's matrix reads the same alone and in its stack."""
    package = Package.local(tiny)
    moved = Weights(package, CPU, False, Perturbation(5))
    name = "blk.3.ffn_gate_exps"
    stack = moved(name + ".weight")
    assert all(torch.equal(moved.expert(name + ".weight", index), stack[index]) for index in range(EXPERTS))
    grids = [grid for grid in package.names() if grid.endswith(".scale")]
    before = np.concatenate([np.array(package.raw(grid)).view(np.int32).reshape(-1) for grid in grids])
    after = np.concatenate([moved.stored(grid).view(np.int32).reshape(-1) for grid in grids])  # positive: ordinals
    assert np.abs(after - before).max() == 1


@pytest.mark.parametrize("file, length, prefixes", [("tiny", LENGTH, (1, 5, 9, 16)),
                                                   ("glm_selection", GLM_LENGTH,
                                                    (1, 77, ROW_TILE, ROW_TILE + 1, GLM_TOP + 4))])
def test_every_row_depends_only_on_its_prefix(file: str, length: int, prefixes: tuple[int, ...], request) -> None:
    """Chunked prefill and verify compute the same rows as a whole prompt, the indexer's selection included: row t
    of any prefix holds row t (`assert_rows_hold`, bit for bit where the platform's rows are exact), for a row alone,
    inside a row tile and across tiles, before and after the top pools bind."""
    path, prompt = request.getfixturevalue(file), tokens(1, 2, length)
    with torch.no_grad():
        whole = reference(path).forward(prompt).numpy()
        step = storage_step(whole, reference(path, perturbation=Perturbation(1)).forward(prompt).numpy())
        for prefix in prefixes:
            assert_rows_hold(reference(path).forward(prompt[:, :prefix]).numpy(), whole[:, :prefix],
                             step.reshape(2, length)[:, :prefix].reshape(-1))


def test_a_full_file_runs_its_text_blocks_and_never_its_mtp_head(tiny: Path, tmp_path: Path) -> None:
    """A checkpoint's trailing `nextn_predict_layers` blocks are the drafter's: the reference computes the text
    model's rows exactly as the file without them does."""
    full = tmp_path / "full.gguf"
    write_tiny_glm5next(full, mtp=1)
    prompt = tokens(1, 2, LENGTH)
    with torch.no_grad():
        assert np.array_equal(reference(full).forward(prompt).numpy(), reference(tiny).forward(prompt).numpy())
    with pytest.raises(ValueError, match="only 4 main layers"):
        reference(full, layers=5)


def test_the_forward_reads_every_tensor(tiny: Path) -> None:
    model = reference(tiny, cache=False)
    with torch.no_grad():
        model.forward(tokens(2, 1))
    assert model.weights.unread() == []


def test_matches_transformers(tiny: Path) -> None:
    pytest.importorskip("transformers", reason="the references' `check` extra")
    report = check_hf(reference(tiny), 2, LENGTH, 20261109)
    assert report["pass"], report
