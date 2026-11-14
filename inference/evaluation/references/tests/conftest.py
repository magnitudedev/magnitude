from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gguf.constants import GGMLQuantizationType as Q, GGUFValueType as V

from magnitude_references import digest
from magnitude_rig.host.identity import sha256
from magnitude_references.gguf import Value, Writer, quantize
from magnitude_references.quality import compare_rows

VOCAB, HIDDEN, HEADS, KV_HEADS, LAYERS, FFN = 64, 32, 4, 2, 2, 48
ROWS_EXACT = digest.platform("cpu")["rows_exact"]


def storage_step(rows: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Each row's KL from logits `rows` [..., V] of a model's reference to `moved`, the same rows of its
    storage-resolution-perturbed reference (`perturbed:<seed>`): the storage-resolution bar's unit at each row."""
    unit, _ = compare_rows(*(np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1]) for x in (rows, moved)))
    return unit


def assert_rows_hold(rows: np.ndarray, whole: np.ndarray, step: np.ndarray) -> None:
    """Logits `rows` [..., V] a chunked or batched run gave are the rows `whole` one run gives: bit for bit where the
    platform makes a row's bits its prefix's alone (`digest.ROWS_EXACT`). Elsewhere each row's KL to its whole row is
    at most `step`, the KL one storage step moves that row by (`storage_step`).

    The bound is the bar's own unit: `kl.resolution` admits a build up to α² (64) times the perturbed reference's
    KL, so a reference whose rows move by at most one unit moves no verdict by more than 1/64 of the bar. It holds
    x86's near-exact rows, whose mechanisms (MKL rounds a GEMM row by its place in the call and the call's row count,
    torch's vectorised silu and softplus an element by the tensor's length) all stay far below one storage step of
    a model stored as models are (the fixtures store their tensors in their models' formats, the tiny llama's
    matrices in BF16)."""
    rows, whole = (np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1]) for x in (rows, whole))
    if ROWS_EXACT:
        assert np.array_equal(rows, whole)
        return
    drift, _ = compare_rows(whole, rows)
    assert (drift <= step).all(), f"rows moved by {drift.max():.3g}, beyond one storage step ({step[drift.argmax()]:.3g})"


def write_tiny_llama(path: Path, seed: int = 0, sharp: float = 0.0) -> None:
    """A two-layer `llama` GGUF of the MiniCPM5 shape with random weights. `sharp` > 0 adds a scaled
    identity-shifted readout so the model predicts token t+1 after t (a model in a sane regime). Its matrices are
    stored in BF16, as a model's are, so one storage step (`storage_step`) is a real format's resolution: in F32 it
    would be one float32 step, the scale of the rounding it bounds."""
    rng = np.random.default_rng(seed)
    metadata = {"general.architecture": Value(V.STRING, "llama")}
    for key, kind, value in [("block_count", V.UINT32, LAYERS), ("context_length", V.UINT32, 4096),
                             ("embedding_length", V.UINT32, HIDDEN), ("feed_forward_length", V.UINT32, FFN),
                             ("attention.head_count", V.UINT32, HEADS), ("attention.head_count_kv", V.UINT32, KV_HEADS),
                             ("rope.freq_base", V.FLOAT32, 10000.0),
                             ("attention.layer_norm_rms_epsilon", V.FLOAT32, 1e-6)]:
        metadata[f"llama.{key}"] = Value(kind, value)
    dim = HIDDEN // HEADS
    tensors: list[tuple[str, np.ndarray, Q]] = []

    def matrix(name: str, rows: int, cols: int, scale: float = 1.0) -> None:
        tensors.append((name, (rng.standard_normal((rows, cols)) * scale / np.sqrt(cols)).astype(np.float32), Q.BF16))

    embedding = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float32)
    tensors.append(("token_embd.weight", embedding, Q.BF16))
    tensors.append(("output_norm.weight", np.ones(HIDDEN, np.float32), Q.F32))
    output = (rng.standard_normal((VOCAB, HIDDEN)) / np.sqrt(HIDDEN)).astype(np.float32)
    if sharp:
        # Readout row t+1 aligned with the (normalized) embedding of t: the residual stream carries the
        # input token, so the model predicts the next token id.
        output += sharp * np.roll(embedding, 1, axis=0) / np.linalg.norm(embedding, axis=1, keepdims=True).mean()
    tensors.append(("output.weight", output, Q.BF16))
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        tensors.append((p + "attn_norm.weight", np.ones(HIDDEN, np.float32), Q.F32))
        tensors.append((p + "ffn_norm.weight", np.ones(HIDDEN, np.float32), Q.F32))
        matrix(p + "attn_q.weight", HEADS * dim, HIDDEN)
        matrix(p + "attn_k.weight", KV_HEADS * dim, HIDDEN)
        matrix(p + "attn_v.weight", KV_HEADS * dim, HIDDEN, 0.1)
        matrix(p + "attn_output.weight", HIDDEN, HEADS * dim, 0.1)
        matrix(p + "ffn_gate.weight", FFN, HIDDEN)
        matrix(p + "ffn_up.weight", FFN, HIDDEN)
        matrix(p + "ffn_down.weight", HIDDEN, FFN, 0.1)
    writer = Writer(path, metadata)
    for name, values, kind in tensors:
        writer.declare(name, tuple(reversed(values.shape)), kind)
    writer.begin()
    for name, values, kind in tensors:
        writer.write(name, quantize(values, kind))
    writer.close()


def write_rendered(directory: Path, model: Path, fixtures: list[tuple[list[int], list[list[int]]]],
                   model_id: str = "tiny:gguf:f32") -> tuple[Path, Path]:
    target_sha256 = sha256(model)
    lines = [{"fixture_id": f"test/{index}", "fixture_set_digest": "0" * 64, "model": model_id,
              "target_sha256": target_sha256, "template_sha256": "1" * 64, "context_target": None,
              "tokens": tokens, "scored_spans": spans, "generation_prefix_tokens": None}
             for index, (tokens, spans) in enumerate(fixtures)]
    rendered = directory / "rendered.jsonl"
    rendered.write_text("".join(json.dumps(line) + "\n" for line in lines))
    artifacts = directory / "artifacts.json"
    artifacts.write_text(json.dumps({"model": model_id, "target": str(model), "target_sha256": target_sha256,
                                     "drafter": None, "projector": None}))
    return rendered, artifacts


@pytest.fixture
def tiny_llama(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.gguf"
    write_tiny_llama(path)
    return path


@pytest.fixture
def sharp_llama(tmp_path: Path) -> Path:
    path = tmp_path / "sharp.gguf"
    write_tiny_llama(path, sharp=12.0)
    return path
