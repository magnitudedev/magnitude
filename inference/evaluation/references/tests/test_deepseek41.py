"""The `deepseek41` reference on a small synthetic file in the converter's form: the FP8 block scales,
the trained roundings against torch's own E4M3 cast and the E2M1 grid, the row-by-row definition, and
DeepSeek's released code (network: the pinned `inference/` files)."""
from __future__ import annotations

from dataclasses import replace
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
from magnitude_references.perturbation import Perturbation

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


def reference(path: Path, roundings: Roundings = Roundings(), perturbation: Perturbation | None = None,
              cache: bool = True) -> DeepSeek41Reference:
    package = Package.local(path)
    return DeepSeek41Reference(package, Weights(package, CPU, cache, perturbation), None, roundings)


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


@pytest.mark.parametrize("roundings, perturbation", [(Roundings(), None), (Roundings.trained(), None),
                                                      (Roundings.trained(), Perturbation(3))],
                         ids=["exact", "trained", "perturbed"])
def test_every_row_depends_only_on_its_prefix(tiny: Path, roundings: Roundings, perturbation: Perturbation | None) -> None:
    """Chunked prefill and verify compute the same rows as a whole prompt: row t of any prefix equals row t. Perturbed,
    so a stored row moves by its own identity alone, however many rows the store holds when it is formed."""
    model = reference(tiny, roundings, perturbation)
    tokens = torch.from_numpy(np.random.default_rng(1).integers(0, VOCAB, (2, 24)))
    with torch.no_grad():
        whole = model.forward(tokens)
        for length in (1, 5, 9, 16):
            torch.testing.assert_close(model.forward(tokens[:, :length]), whole[:, :length], rtol=1e-5, atol=1e-5)


# DeepSeek's code runs widened to F32 here, where its BF16 holds (`.to(dtype)`) are identities.
@pytest.mark.parametrize("roundings", [Roundings(), replace(Roundings.trained(), held=False)],
                         ids=["exact", "trained-unheld"])
def test_matches_deepseek_code(tiny: Path, roundings: Roundings) -> None:
    report = check_hf(reference(tiny, roundings), 2, 24, 20261106)
    assert report["pass"], report


def test_holds_match_deepseek_code_in_its_dtypes(tiny: Path) -> None:
    """`Roundings.held` against DeepSeek's code in its own dtypes (BF16 activations, F32 where model.py says). At
    every layer, fed its BF16 input as model.py computed it, the reference's window keys are the rows model.py's
    window stores, exactly. At each layer that compresses and indexes its own rows, fed also its low-rank query,
    the reference's compressed rows and index keys are model.py's codes and its selection is model.py's top-k,
    exactly."""
    from magnitude_references import deepseek41_released as released
    from magnitude_references.families.deepseek41 import Shared
    model = reference(tiny, Roundings.trained())
    tokens = torch.from_numpy(np.random.default_rng(2).integers(0, VOCAB, (1, 24)))
    module, transformer = released.build(model, tokens, native=True)
    seen, windows = {}, {}
    forward, window = module.Indexer.forward, module.Attention._window_kv

    def indexer_forward(self, x, qr, latent, start_pos, offset):
        idxs = forward(self, x, qr, latent, start_pos, offset)
        if self.owns_k:  # `latent` is rotated and rounded in place after this returns
            seen[self] = (x.float(), qr.float(), latent, self.k_cache, idxs, offset)
        return idxs

    def window_kv(self, x, freqs_cis, start_pos):
        kv, idxs = window(self, x, freqs_cis, start_pos)
        windows[self.layer_id] = (x.float(), kv.float())
        return kv, idxs

    module.Indexer.forward, module.Attention._window_kv = indexer_forward, window_kv
    with torch.no_grad():
        transformer(tokens, 0)
    positions = torch.arange(tokens.shape[1])
    rotaries = model.rotaries(positions)
    for layer, (x, kv) in windows.items():
        with torch.no_grad():
            assert torch.equal(model.window_keys(layer, x, rotaries[model.ratios[layer] > 0]), kv), layer
    assert len(windows) == model.block_count
    rotary = rotaries[True]
    checked = 0
    for layer, block in enumerate(transformer.layers):
        indexer = block.attn.indexer
        if indexer not in seen:
            continue
        x, qr, rows, keys, idxs, offset = seen[indexer]
        shared = Shared()
        with torch.no_grad():
            model.compress(layer, x, rotary, shared)
            model.select(layer, x, qr, rotary, positions, shared)
        assert torch.equal(shared.rows, rows.float()), layer
        assert torch.equal(shared.index_keys, keys[:, :rows.shape[1]].float()), layer
        if not 0 <= model.candidate < layer:  # a candidate reader's selection depends on an earlier layer's
            picked = [sorted(int(i) - offset for i in row if i >= 0) for row in idxs[0]]
            assert picked == [row.nonzero().flatten().tolist() for row in shared.selected[0]], layer
        checked += 1
    assert checked > 0


def test_activations_match_deepseek_code_in_its_dtypes(tiny: Path) -> None:
    """`Roundings.activations` (a measurement switch) against DeepSeek's code in its own dtypes, its FP8 / FP4 GEMMs
    over their cast inputs: every block and Engram layer, fed its BF16 streams as model.py computed them, gives
    model.py's streams, exactly."""
    from magnitude_references import deepseek41_released as released
    from magnitude_references.families.deepseek41 import Shared
    model = reference(tiny, replace(Roundings.trained(), activations=True))
    tokens = torch.from_numpy(np.random.default_rng(3).integers(0, VOCAB, (1, 24)))
    module, transformer = released.build(model, tokens, native=True)
    blocks, engrams = {}, {}
    block_forward, engram_forward = module.Block.forward, module.Engram.forward

    def block(self, x, start_pos, pre_mix, image_mask, *rest):
        out, pre = block_forward(self, x, start_pos, pre_mix, image_mask, *rest)
        blocks[self.layer_id] = (x.float(), pre_mix, out.float(), pre)
        return out, pre

    def engram(self, x, hash_ids, token_mask=None):
        out = engram_forward(self, x, hash_ids, token_mask)
        engrams[self.layer_id] = (x.float(), out.float())
        return out

    module.Block.forward, module.Engram.forward = block, engram
    with torch.no_grad():
        transformer(tokens, 0)
    positions = torch.arange(tokens.shape[1])
    rotaries, shared, rows = model.rotaries(positions), Shared(), model.ngram_rows(tokens)
    with torch.no_grad():
        for layer, (x, out) in engrams.items():
            assert torch.equal(model.engram(layer, x, rows[:, :, model.engram_layers.index(layer)]), out), layer
        for layer in range(model.block_count):
            x, pre_mix, out, pre = blocks[layer]
            streams, ours = model.block(layer, x, pre_mix, rotaries, positions, shared)
            assert torch.equal(ours, pre), layer
            assert torch.equal(streams, out), layer
    assert engrams and len(blocks) == model.block_count


def test_activations_variant_has_its_own_identity(tiny: Path) -> None:
    """The `activations` variant opens the trained roundings with the switch on, under an implementation identity
    (the reference cache's key) of its own; an unknown variant is refused."""
    from magnitude_references import digest
    from magnitude_references.families import open_reference
    assert open_reference(tiny, CPU, True, variant="activations").roundings == replace(Roundings.trained(), activations=True)
    assert open_reference(tiny, CPU, True).roundings == Roundings.trained()
    assert digest.implementation("activations") != digest.implementation()
    assert len({digest.implementation(), digest.implementation("perturbed:1"), digest.implementation("perturbed:2")}) == 3
    perturbed = open_reference(tiny, CPU, True, variant="perturbed:1")
    assert perturbed.weights.perturbation == Perturbation(1) and perturbed.roundings == Roundings.trained()
    with pytest.raises(ValueError, match="no reference variant"):
        open_reference(tiny, CPU, True, variant="unknown")


def test_a_perturbed_value_moves_once_however_it_is_read(tiny: Path) -> None:
    """A stored value moves by one draw, the same in every read of it: a whole tensor, one expert's matrix of it,
    table rows and their tile scales, a second streamed read. About 1 in 12 stored codes moves, each to an
    adjacent code."""
    package = Package.local(tiny)
    plain, moved = Weights(package, CPU, False), Weights(package, CPU, False, Perturbation(11))
    name = "blk.0.ffn_up_exps.weight"
    assert all(torch.equal(moved(name)[index], moved.expert(name, index)) for index in range(EXPERTS))
    rows = torch.tensor([[5, 0], [5, 40]])
    table = "per_layer_token_embd.weight"
    assert torch.equal(moved.rows(table, rows), moved(table)[rows])
    for tensor in (name, table, "blk.0.attn_q_b.weight", "token_embd.weight"):
        assert torch.equal(moved(tensor), moved(tensor)), tensor
        assert not torch.equal(moved(tensor), plain(tensor)), tensor
    def ordinals(codes: np.ndarray) -> np.ndarray:  # BF16 codes in value order
        codes = codes.reshape(-1).view(np.uint16).astype(np.int64)
        return np.where(codes & 0x8000, -(codes & 0x7FFF), codes)

    before, after = ordinals(np.array(package.raw("token_embd.weight"))), ordinals(moved.stored("token_embd.weight"))
    assert 0.06 < np.count_nonzero(after != before) / before.size < 0.11
    assert np.abs(after - before).max() == 1


def test_a_perturbed_reference_moves_stored_values_only(tiny: Path, monkeypatch) -> None:
    """Every plane the perturbation draws for is a stored one: a tensor of the file (an I8 tensor only as a weight's
    scale) or a store of trained-rounding rows. The n-gram table rows, an index computed from the token ids, are the
    unperturbed model's."""
    planes: set[str] = set()
    steps = Perturbation.steps

    def recorded(self, plane, index):
        planes.add(plane)
        return steps(self, plane, index)

    monkeypatch.setattr(Perturbation, "steps", recorded)
    model = reference(tiny, Roundings.trained(), Perturbation(2))
    tokens = torch.from_numpy(np.random.default_rng(4).integers(0, VOCAB, (1, 16)))
    with torch.no_grad():
        model.forward(tokens)
    assert torch.equal(model.ngram_rows(tokens), reference(tiny, Roundings.trained()).ngram_rows(tokens))
    package, stores = model.package, set()
    for plane in planes:
        name = plane.removesuffix("/scales")
        if package.has(name):
            assert package.info(name).type != Q.I8 or name.endswith(".scale"), plane
        else:
            layer, _, store = name.removeprefix("blk.").partition(".")
            assert layer.isdecimal(), plane
            stores.add(store)
    assert stores == {"window", "indexer.keys", "compressor.rows"}
    assert planes >= {"blk.0.ffn_up_exps.weight", "blk.0.ffn_up_exps.weight/scales", "blk.0.attn_q_b.scale",
                      "blk.2.window/scales", "blk.2.compressor.rows/scales"}


def test_a_perturbed_reference_built_twice_is_the_same(tiny: Path) -> None:
    """One seed's reference is one function of the file: two streamed builds give the same bytes, which differ from
    the unperturbed reference's and from another seed's."""
    tokens = torch.from_numpy(np.random.default_rng(5).integers(0, VOCAB, (2, 20)))
    with torch.no_grad():
        first, second, other, plain = (reference(tiny, Roundings.trained(), perturbation, cache=False).forward(tokens)
                                       for perturbation in (Perturbation(9), Perturbation(9), Perturbation(10), None))
    assert first.numpy().tobytes() == second.numpy().tobytes()
    assert not torch.equal(first, plain) and not torch.equal(first, other)


def test_a_perturbed_reference_needs_the_stored_roundings(tiny: Path) -> None:
    with pytest.raises(ValueError, match="needs every stored rounding"):
        reference(tiny, Roundings(), Perturbation(1))


def test_hash_constants_are_deepseeks(tiny: Path) -> None:
    from magnitude_references import deepseek41_released
    model = reference(tiny)
    module = deepseek41_released.load(model)
    engram = module.NgramHashState.__init__.__globals__
    args = deepseek41_released.arguments(model, module, torch.zeros(1, 8, dtype=torch.int64))
    layout = module.EngramLayout.from_args(args)
    assert torch.equal(torch.tensor(layout.primes).flatten(1), model.bucket_sizes)
    assert torch.equal(engram["compute_hash_multipliers"](layout.layer_ids, NGRAM, COMPRESSED_VOCAB), model.multipliers)
