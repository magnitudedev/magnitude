"""The `deepseek41` reference on a small synthetic file in the converter's form: the FP8 block scales,
the trained roundings against torch's own E4M3 cast and the E2M1 grid, the row-by-row definition, and
DeepSeek's released code (network: the pinned `inference/` files)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V
from sympy import nextprime

from magnitude_references.checks import check_hf
from magnitude_references.families.deepseek41 import (E2M1, E4M3, DeepSeek41Reference, Roundings, power_of_two_ceil,
                                                      round_codes)
from magnitude_references.gguf import Package, Value, Writer, quantize
from magnitude_references.model import SCALE_ENCODING, Weights

CPU = torch.device("cpu")
VOCAB, HIDDEN, HEADS, DIM, ROPE, Q_RANK, GROUPS, O_RANK = 64, 64, 2, 64, 16, 32, 2, 32
INDEX_HEADS, INDEX_DIM, INDEX_TOP, EXPERTS, USED, FFN, STREAMS, WINDOW = 8, 32, 3, 4, 2, 32, 4, 4
# SWA, Engram + SWA, ratio-2 source, reuse (Engram), ratio-1 source and candidates, reuse, restricted index, reuse
RATIOS, KV_SOURCES, INDEX_SOURCES, CANDIDATE = [0, 0, 2, 2, 1, 1, 1, 1], [2, 4], [2, 4, 6], 4
ENGRAM_LAYERS, NGRAM, NGRAM_HEADS, ENGRAM_DIM, ENGRAM_BASE, COMPRESSED_VOCAB = [1, 3], 3, 2, 32, 11, 40


def fp8(rng, rows: int, cols: int, block: tuple[int, int], gain: float = 1.0):
    """E4M3 codes [rows, cols] and UE8M0 block exponents (as I8) of a random matrix, the checkpoint's form."""
    values = torch.from_numpy(rng.standard_normal((rows, cols)) * gain / math.sqrt(cols)).to(torch.float32)
    blocks = values.view(rows // block[0], block[0], cols // block[1], block[1])
    exponents = torch.ceil(torch.log2(blocks.abs().amax((1, 3)) / 448)).to(torch.int32)
    codes = (blocks / torch.exp2(exponents.float())[:, None, :, None]).to(torch.float8_e4m3fn).view(rows, cols)
    return codes.view(torch.uint8).numpy(), (exponents + 127).to(torch.uint8).numpy().view(np.int8)


def hash_constants():
    """Engram primes (fresh primes from the base, layer-major), absolute offsets and multipliers (engram.py)."""
    sizes, seen = [], set()
    for _ in ENGRAM_LAYERS:
        for _ in range(NGRAM - 1):
            current = ENGRAM_BASE - 1
            for _ in range(NGRAM_HEADS):
                current = nextprime(current)
                while current in seen:
                    current = nextprime(current)
                seen.add(current)
                sizes.append(current)
    bound = np.iinfo(np.int64).max // COMPRESSED_VOCAB // 2
    multipliers = [int(m) * 2 + 1 for layer in ENGRAM_LAYERS
                   for m in np.random.default_rng(10007 * layer).integers(0, bound, NGRAM, dtype=np.int64)]
    return sizes, [sum(sizes[:i]) for i in range(len(sizes))], multipliers


def write_tiny_deepseek41(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    layers = len(RATIOS)
    sizes, offsets, multipliers = hash_constants()
    first_seen: dict[int, int] = {}
    every = rng.permutation(np.concatenate((np.arange(COMPRESSED_VOCAB), rng.integers(0, COMPRESSED_VOCAB, VOCAB - COMPRESSED_VOCAB))))
    token_map = [first_seen.setdefault(int(t), len(first_seen)) for t in every]
    u32, f32 = (lambda v: Value(V.UINT32, v)), (lambda v: Value(V.FLOAT32, v))
    array = lambda kind, values: Value(V.ARRAY, list(values), kind)  # noqa: E731
    keys = {
        "block_count": u32(layers), "embedding_length": u32(HIDDEN), "attention.head_count": u32(HEADS),
        "attention.key_length": u32(DIM), "rope.dimension_count": u32(ROPE),
        "attention.layer_norm_rms_epsilon": f32(1e-20), "attention.sliding_window": u32(WINDOW),
        "attention.output_group_count": u32(GROUPS), "attention.output_lora_rank": u32(O_RANK),
        "attention.compress_ratios": array(V.UINT32, RATIOS + [0]), "rope.freq_base": f32(10000.0),
        "attention.compress_rope_freq_base": f32(160000.0), "rope.scaling.factor": f32(16.0),
        "rope.scaling.original_context_length": u32(64), "rope.scaling.yarn_beta_fast": f32(32.0),
        "rope.scaling.yarn_beta_slow": f32(1.0), "hyper_connection.count": u32(STREAMS),
        "hyper_connection.sinkhorn_iterations": u32(20), "hyper_connection.epsilon": f32(1e-6),
        "expert_count": u32(EXPERTS), "expert_used_count": u32(USED), "expert_weights_scale": f32(1.5),
        "expert_weights_norm": Value(V.BOOL, True), "expert_gating_func": u32(4),
        "swiglu_clamp_exp": array(V.FLOAT32, [0.5] * layers), "swiglu_clamp_shexp": array(V.FLOAT32, [0.5] * layers),
        "attention.indexer.head_count": u32(INDEX_HEADS), "attention.indexer.key_length": u32(INDEX_DIM),
        "attention.indexer.top_k": u32(INDEX_TOP), "attention.kv_source_layers": array(V.UINT32, KV_SOURCES),
        "attention.indexer.source_layers": array(V.UINT32, INDEX_SOURCES),
        "attention.indexer.candidate_source_layer": Value(V.INT32, CANDIDATE),
        "attention.indexer.candidate_top_k_blocks": u32(2), "attention.indexer.candidate_block_size": u32(2),
        "ple.layers": array(V.UINT32, ENGRAM_LAYERS), "ple.ngram_size": u32(NGRAM),
        "ple.heads_per_ngram": u32(NGRAM_HEADS), "embedding_length_per_layer_input": u32(ENGRAM_DIM),
        "ple.layer_multipliers": array(V.UINT64, multipliers), "ple.head_offsets": array(V.UINT64, offsets),
        "ple.head_vocab_sizes": array(V.UINT64, sizes), "engram.pad_token_id": u32(2),
        "engram.token_map": array(V.INT32, token_map),
    }
    metadata = {"general.architecture": Value(V.STRING, "deepseek41"), SCALE_ENCODING: Value(V.STRING, "ue8m0")}
    metadata |= {f"deepseek41.{key}": value for key, value in keys.items()}
    tensors: list[tuple[str, np.ndarray, Q, tuple[int, ...]]] = []  # name, stored bytes, type, NumPy shape

    def normal(name: str, shape: tuple[int, ...], kind: Q, gain: float = 1.0, mean: float = 0.0) -> None:
        values = (mean + rng.standard_normal(shape) * gain / math.sqrt(shape[-1] if len(shape) > 1 else 1)).astype(np.float32)
        tensors.append((name, values if kind == Q.F32 else quantize(values, kind), kind, shape))

    def scaled(name: str, rows: int, cols: int, block=(32, 32), gain: float = 1.0) -> None:
        codes, exponents = fp8(rng, rows, cols, block, gain)
        tensors.extend([(name + ".weight", codes, Q.F8_E4M3, codes.shape), (name + ".scale", exponents, Q.I8, exponents.shape)])

    normal("token_embd.weight", (VOCAB, HIDDEN), Q.BF16, math.sqrt(HIDDEN))
    normal("output_norm.weight", (HIDDEN,), Q.BF16, 0.1, 1.0)
    normal("output.weight", (VOCAB, HIDDEN), Q.BF16, 4.0)
    scaled("per_layer_token_embd", sum(sizes), ENGRAM_DIM, (1, 32), math.sqrt(ENGRAM_DIM))
    mixes = (2 + STREAMS) * STREAMS
    for layer in range(layers):
        p = f"blk.{layer}."
        for site in ("attn", "ffn"):
            normal(p + f"hc_{site}_fn.weight", (mixes, STREAMS * HIDDEN), Q.F32)
            normal(p + f"hc_{site}_base.weight", (mixes,), Q.F32, 0.5)
            normal(p + f"hc_{site}_scale.weight", (3,), Q.F32, 0.5, 1.0)
        for norm, width in (("attn_norm", HIDDEN), ("ffn_norm", HIDDEN), ("attn_q_a_norm", Q_RANK), ("attn_kv_a_norm", DIM)):
            normal(p + norm + ".weight", (width,), Q.BF16, 0.1, 1.0)
        normal(p + "attn_sinks.weight", (HEADS,), Q.F32)
        scaled(p + "attn_q_a", Q_RANK, HIDDEN)
        scaled(p + "attn_q_b", HEADS * DIM, Q_RANK)
        scaled(p + "attn_kv", DIM, HIDDEN)
        scaled(p + "attn_output_a", GROUPS * O_RANK, HEADS * DIM // GROUPS)
        scaled(p + "attn_output_b", HIDDEN, GROUPS * O_RANK, gain=0.5)
        if layer in KV_SOURCES:
            for part in ("kv", "gate") if RATIOS[layer] > 1 else ("kv",):
                normal(p + f"attn_compressor_{part}.weight", (DIM, HIDDEN), Q.BF16)
            normal(p + "attn_compressor_norm.weight", (DIM,), Q.BF16, 0.1, 1.0)
            normal(p + "indexer.attn_k.weight", (INDEX_DIM, DIM), Q.BF16)
            normal(p + "indexer.k_norm.weight", (INDEX_DIM,), Q.BF16, 0.1, 1.0)
        if layer in INDEX_SOURCES:
            scaled(p + "indexer.attn_q_b", INDEX_HEADS * INDEX_DIM, Q_RANK)
            normal(p + "indexer.proj.weight", (INDEX_HEADS, HIDDEN), Q.BF16)
        if layer in ENGRAM_LAYERS:
            scaled(p + "engram_kv", (STREAMS + 1) * HIDDEN, (NGRAM - 1) * NGRAM_HEADS * ENGRAM_DIM)
            normal(p + "engram_gate.weight", (STREAMS, HIDDEN), Q.F32, math.sqrt(HIDDEN))
        normal(p + "ffn_gate_inp.weight", (EXPERTS, HIDDEN), Q.BF16)
        normal(p + "exp_probs_b.bias", (EXPERTS,), Q.F32, 0.1)
        for name, rows, cols in (("gate", FFN, HIDDEN), ("up", FFN, HIDDEN), ("down", HIDDEN, FFN)):
            values = (rng.standard_normal((EXPERTS, rows, cols)) / math.sqrt(cols)).astype(np.float32)
            tensors.append((p + f"ffn_{name}_exps.weight", quantize(values, Q.MXFP4), Q.MXFP4, values.shape))
            scaled(p + f"ffn_{name}_shexp", rows, cols, gain=0.5 if name == "down" else 1.0)
    writer = Writer(path, metadata)
    for name, _, kind, shape in tensors:
        writer.declare(name, tuple(reversed(shape)), kind)
    writer.begin()
    for name, data, _, _ in tensors:
        writer.write(name, data)
    writer.close()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("deepseek41") / "tiny.gguf"
    write_tiny_deepseek41(path)
    return path


def reference(path: Path, roundings: Roundings = Roundings()) -> DeepSeek41Reference:
    package = Package.local(path)
    return DeepSeek41Reference(package, Weights(package, CPU, True), None, roundings)


def test_fp8_block_scales_are_exact(tiny: Path) -> None:
    package = Package.local(tiny)
    weights = Weights(package, CPU, False)
    for name, block in (("blk.0.attn_q_b", (32, 32)), ("per_layer_token_embd", (1, 32))):
        codes = torch.from_numpy(np.array(package.raw(name + ".weight"))).view(torch.float8_e4m3fn).float()
        codes = codes.view(package.info(name + ".weight").array_shape)
        exponents = torch.from_numpy(np.array(package.raw(name + ".scale")).astype(np.int32) - 127)
        exponents = exponents.view(package.info(name + ".scale").array_shape)
        factors = torch.exp2(exponents.float()).repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
        assert torch.equal(weights(name + ".weight"), codes * factors), name
    rows = torch.tensor([[5, 0], [5, 40]])
    assert torch.equal(weights.rows("per_layer_token_embd.weight", rows), weights("per_layer_token_embd.weight")[rows])


def test_roundings_match_the_formats() -> None:
    generator = torch.Generator().manual_seed(0)
    values = torch.cat((torch.randn(4096, generator=generator) * 64, torch.randn(4096, generator=generator) * 2 ** -8,
                        torch.tensor([0.0, 448.0, -448.0, 2 ** -9, 2 ** -10, 3 * 2 ** -11, 17.0, 19.0, -9.5])))
    values = values.clamp(-448, 448)
    assert torch.equal(round_codes(values, E4M3), values.to(torch.float8_e4m3fn).float())
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    points = torch.cat((torch.linspace(-6, 6, 1201), (grid[1:] + grid[:-1]) / 2, -(grid[1:] + grid[:-1]) / 2))
    distance = (points.abs()[:, None] - grid[None, :]).abs()
    nearest = distance == distance.min(-1, keepdim=True).values
    # ties go to the code with an even mantissa: indices 0, 2, 4, 6 of the grid
    even = nearest & (torch.arange(8) % 2 == 0)
    choice = torch.where(even.any(-1, keepdim=True), even, nearest).float().argmax(-1)
    assert torch.equal(round_codes(points, E2M1), grid[choice] * points.sign())
    samples = torch.tensor([2.0 ** -126, 0.3, 0.5, 1.0, 1.5, 448.0, 1000.0])
    assert torch.equal(power_of_two_ceil(samples), torch.exp2(torch.ceil(torch.log2(samples))))


@pytest.mark.parametrize("roundings", [Roundings(), Roundings.trained()], ids=["exact", "trained"])
def test_every_row_depends_only_on_its_prefix(tiny: Path, roundings: Roundings) -> None:
    """Chunked prefill and verify compute the same rows as a whole prompt: row t of any prefix equals row t."""
    model = reference(tiny, roundings)
    tokens = torch.from_numpy(np.random.default_rng(1).integers(0, VOCAB, (2, 24)))
    with torch.no_grad():
        whole = model.forward(tokens)
        for length in (1, 5, 9, 16):
            torch.testing.assert_close(model.forward(tokens[:, :length]), whole[:, :length], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("roundings", [Roundings(), Roundings.trained()], ids=["exact", "trained"])
def test_matches_deepseek_code(tiny: Path, roundings: Roundings) -> None:
    report = check_hf(reference(tiny, roundings), 2, 24, 20261106)
    assert report["pass"], report


def test_hash_constants_are_deepseeks(tiny: Path) -> None:
    from magnitude_references import deepseek41_released
    model = reference(tiny)
    module = deepseek41_released.load(model)
    engram = module.NgramHashState.__init__.__globals__
    args = deepseek41_released.arguments(model, module, torch.zeros(1, 8, dtype=torch.int64))
    layout = module.EngramLayout.from_args(args)
    assert torch.equal(torch.tensor(layout.primes).flatten(1), model.bucket_sizes)
    assert torch.equal(engram["compute_hash_multipliers"](layout.layer_ids, NGRAM, COMPRESSED_VOCAB), model.multipliers)
