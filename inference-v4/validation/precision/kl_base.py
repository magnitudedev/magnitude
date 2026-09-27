#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Read llama.cpp `--kl-divergence-base` files and compute D4 metrics.

uv run inference-v4/validation/precision/kl_base.py info BASE.bin
uv run inference-v4/validation/precision/kl_base.py compare BASE.bin CANDIDATE.{bin,npy} [--json OUT]
uv run inference-v4/validation/precision/kl_base.py self-test [--real BASE.bin --corpus TEXT --model GGUF]

The byte format and metric definitions are specified in README.md next to this
file. Metrics follow llama.cpp `tools/perplexity/perplexity.cpp` (b8680 through
b10998): KL(base || candidate) per evaluated position, summing only vocabulary
entries whose decoded base log-probability exceeds -16; "same top" compares the
first argmax of the candidate logits with the first argmax of the decoded base
row. A candidate is either another base file (decoded log-probabilities stand
in for logits) or a float32 .npy array of raw logits shaped
[n_chunk, n_eval, n_vocab] aligned with the base rows.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

MAGIC = b"_logits_"
HEADER_BYTES = 20
BASE_LOG_PROB_CUTOFF = np.float32(-16.0)


def row_width(n_vocab: int) -> int:
    """uint16 elements per row: 4 (two float32 fields) + n_vocab rounded up to even."""
    return 2 * ((n_vocab + 1) // 2) + 4


def evaluated_rows(n_ctx: int) -> int:
    return n_ctx - 1 - n_ctx // 2


class BaseFile:
    """A memory-mapped llama.cpp KL-divergence base file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header = stream.read(HEADER_BYTES)
        if len(header) != HEADER_BYTES or header[:8] != MAGIC:
            raise ValueError(f"{self.path}: not a llama.cpp log-probability file")
        self.n_ctx = int(np.frombuffer(header, "<u4", 1, 8)[0])
        self.n_vocab, self.n_chunk = (int(value) for value in np.frombuffer(header, "<i4", 2, 12))
        self.first = self.n_ctx // 2
        self.n_eval = evaluated_rows(self.n_ctx)
        self.nv = row_width(self.n_vocab)
        self.rows_offset = HEADER_BYTES + 4 * self.n_ctx * self.n_chunk
        expected = self.rows_offset + 2 * self.nv * self.n_eval * self.n_chunk
        actual = self.path.stat().st_size
        if actual != expected:
            raise ValueError(f"{self.path}: size {actual} does not match header layout ({expected} bytes)")
        self.tokens = np.memmap(self.path, "<i4", "r", HEADER_BYTES, (self.n_chunk, self.n_ctx))
        self.rows = np.memmap(self.path, "<u2", "r", self.rows_offset, (self.n_chunk, self.n_eval, self.nv))

    def targets(self, chunk: int) -> np.ndarray:
        """Token predicted by each evaluated row: tokens[chunk, first + 1 + r]."""
        return np.asarray(self.tokens[chunk, self.first + 1 :], dtype=np.int64)

    def fields(self, chunk: int) -> tuple[np.ndarray, np.ndarray]:
        """(scale, min_log_prob) float32 per row of a chunk."""
        head = np.ascontiguousarray(self.rows[chunk, :, :4]).view("<f4")
        return head[:, 0], head[:, 1]

    def log_probs(self, chunk: int) -> np.ndarray:
        """Decoded base log-probabilities [n_eval, n_vocab] float32: scale*q + min_log_prob."""
        scale, minimum = self.fields(chunk)
        codes = np.asarray(self.rows[chunk, :, 4 : 4 + self.n_vocab], dtype=np.float32)
        return codes * scale[:, None] + minimum[:, None]

    def describe(self) -> dict:
        return {
            "path": str(self.path), "n_ctx": self.n_ctx, "n_vocab": self.n_vocab, "n_chunk": self.n_chunk,
            "first_evaluated_position": self.first, "evaluated_rows_per_chunk": self.n_eval,
            "row_uint16": self.nv, "row_bytes": 2 * self.nv, "rows_offset": self.rows_offset,
            "bytes": self.path.stat().st_size,
        }


def encode_row(logits: np.ndarray) -> np.ndarray:
    """Encode one row of float32 logits exactly as llama.cpp's log_softmax(..., uint16_t *, ...)."""
    logits = np.asarray(logits, dtype=np.float32)
    n_vocab = logits.shape[0]
    row = np.zeros(row_width(n_vocab), dtype=np.uint16)
    max_logit = logits.max()
    min_logit = np.maximum(logits.min(), max_logit - np.float32(16))
    sum_exp = float(np.exp(logits - max_logit, dtype=np.float32).astype(np.float64).sum())
    log_sum_exp = np.float32(math.log(sum_exp))
    min_log_prob = np.float32(min_logit - max_logit - log_sum_exp)
    scale = np.float32((max_logit - min_logit) / np.float32(65535))
    row[:4].view("<f4")[:] = (scale, min_log_prob)
    if scale != 0:
        inv_scale = np.float32(1) / scale
        codes = np.rint(inv_scale * (logits - min_logit)).astype(np.int64)
        row[4 : 4 + n_vocab] = np.where(logits > min_logit, codes, 0)
    return row


def write_base(path: Path, tokens: np.ndarray, logits: np.ndarray) -> None:
    """Write a base file from tokens [n_chunk, n_ctx] and logits [n_chunk, n_eval, n_vocab]."""
    n_chunk, n_ctx = tokens.shape
    _, n_eval, n_vocab = logits.shape
    if logits.shape[:2] != (n_chunk, evaluated_rows(n_ctx)):
        raise ValueError("logits must be [n_chunk, n_ctx - 1 - n_ctx/2, n_vocab]")
    with Path(path).open("wb") as stream:
        stream.write(MAGIC)
        stream.write(np.array([n_ctx, n_vocab, n_chunk], dtype="<i4").tobytes())
        stream.write(np.ascontiguousarray(tokens, dtype="<i4").tobytes())
        for chunk in range(n_chunk):
            for row in range(n_eval):
                stream.write(encode_row(logits[chunk, row]).tobytes())


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of float32 logits, in float64."""
    logits = logits.astype(np.float64)
    maximum = logits.max(axis=-1, keepdims=True)
    return logits - maximum - np.log(np.exp(logits - maximum).sum(axis=-1, keepdims=True))


MARGINS = (0.01, 0.05, 0.1, 0.2, 0.5)
"""Reference top-two log-probability gaps (nats) at which margin-conditioned same-top is reported."""


def top_gap(base: np.ndarray) -> np.ndarray:
    """Per row: base log-probability of the top token minus that of the runner-up (nats, >= 0)."""
    top_two = np.partition(base, -2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


class Accumulator:
    """Per-position metrics with llama.cpp's summary formulas."""

    def __init__(self):
        self.kld, self.same_top, self.nll, self.nll_base, self.p_diff, self.gap = [], [], [], [], [], []

    def add(self, base: np.ndarray, targets: np.ndarray, log_softmax: np.ndarray) -> None:
        """base: decoded base rows; log_softmax: candidate log-probabilities (same shape)."""
        base = base.astype(np.float64)
        candidate = log_softmax
        included = base > BASE_LOG_PROB_CUTOFF
        terms = np.where(included, np.exp(base) * (base - log_softmax), 0.0)
        rows = np.arange(base.shape[0])
        nll = -log_softmax[rows, targets]
        nll_base = -base[rows, targets]
        self.kld.append(terms.sum(axis=1))
        self.same_top.append(candidate.argmax(axis=1) == base.argmax(axis=1))
        self.nll.append(nll)
        self.nll_base.append(nll_base)
        self.p_diff.append(np.exp(-nll) - np.exp(-nll_base))
        self.gap.append(top_gap(base))

    def summary(self) -> dict:
        kld = np.concatenate(self.kld)
        same = np.concatenate(self.same_top)
        gap = np.concatenate(self.gap)
        nll = np.concatenate(self.nll)
        nll_base = np.concatenate(self.nll_base)
        p_diff = np.concatenate(self.p_diff)
        count = kld.size

        def mean_and_uncertainty(values: np.ndarray) -> tuple[float, float]:
            mean = float(values.mean())
            variance = float((values * values).mean() - mean * mean)
            return mean, math.sqrt(variance / (count - 1)) if variance > 0 and count > 10 else 0.0

        kl_mean, kl_uncertainty = mean_and_uncertainty(kld)
        same_top = float(same.mean())
        log_ppl, _ = mean_and_uncertainty(nll)
        log_ppl_base, _ = mean_and_uncertainty(nll_base)
        return {
            "positions": count,
            "mean_kld": kl_mean, "mean_kld_uncertainty": kl_uncertainty,
            "max_kld": float(kld.max()), "median_kld": float(np.median(kld)),
            "kld_percentiles": {str(p): float(np.percentile(kld, p)) for p in (90, 95, 99, 99.9)},
            "same_top": same_top, "same_top_count": int(same.sum()),
            "same_top_uncertainty": math.sqrt(same_top * (1 - same_top) / (count - 1)) if count > 1 else 0.0,
            "ppl": math.exp(log_ppl), "ppl_base": math.exp(log_ppl_base),
            "ln_ppl_ratio": log_ppl - log_ppl_base,
            "rms_p_diff": math.sqrt(float((p_diff * p_diff).mean())), "max_abs_p_diff": float(np.abs(p_diff).max()),
            # Same-top over positions whose reference top-two gap exceeds each margin (near-tie
            # positions excluded), and the reference gaps at the positions whose top token flipped.
            "same_top_above_margin": {
                str(margin): {"positions": int((gap > margin).sum()),
                              "same_top": float(same[gap > margin].mean()),
                              "flips": int((~same[gap > margin]).sum())}
                for margin in MARGINS
            },
            "flip_gap_percentiles": ({str(p): float(np.percentile(gap[~same], p)) for p in (50, 90, 99, 100)}
                                     if (~same).any() else {}),
        }


def compare(base: BaseFile, candidate) -> dict:
    """Compare a base file with another BaseFile or a [n_chunk, n_eval, n_vocab] logits array."""
    if isinstance(candidate, BaseFile):
        # A candidate may cover more chunks than the base (e.g. a subset F32 reference); only the
        # base's chunks are compared.
        if (candidate.n_ctx, candidate.n_vocab) != (base.n_ctx, base.n_vocab) or candidate.n_chunk < base.n_chunk:
            raise ValueError("candidate base file has a different (n_ctx, n_vocab) or fewer chunks")
        if not np.array_equal(candidate.tokens[: base.n_chunk], base.tokens):
            raise ValueError("candidate base file was evaluated on different tokens")
        # Decoded rows are already log-probabilities. Renormalizing them would count the floor
        # value (row max - 16) once per floored token, ~n_vocab * e^-16 ≈ 0.03 of spurious mass.
        rows = lambda chunk: candidate.log_probs(chunk).astype(np.float64)
    else:
        if candidate.shape != (base.n_chunk, base.n_eval, base.n_vocab):
            raise ValueError(f"candidate logits shape {candidate.shape} != {(base.n_chunk, base.n_eval, base.n_vocab)}")
        rows = lambda chunk: log_softmax(np.asarray(candidate[chunk], dtype=np.float32))
    accumulator = Accumulator()
    for chunk in range(base.n_chunk):
        accumulator.add(base.log_probs(chunk), base.targets(chunk), rows(chunk))
    return accumulator.summary()


def open_candidate(path: Path):
    return np.load(path, mmap_mode="r") if path.suffix == ".npy" else BaseFile(path)


# ---------------------------------------------------------------------------
# Self-test


def synthetic_logits(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    logits = rng.normal(0.0, 3.0, shape).astype(np.float32)
    logits[..., 7] += 12  # a confident token, so rows resemble real LM output
    return logits


def direct_metrics(logits_base: np.ndarray, targets: np.ndarray, logits: np.ndarray) -> tuple[float, float, float]:
    """Unquantized KL / same-top / base NLL, straight from definitions."""
    def log_softmax(x):
        x = x.astype(np.float64)
        m = x.max(axis=-1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

    p, q = log_softmax(logits_base), log_softmax(logits)
    kl = (np.exp(p) * (p - q)).sum(axis=-1).mean()
    same = (logits_base.argmax(-1) == logits.argmax(-1)).mean()
    # The base encoding floors log-probabilities at (row max - 16); llama.cpp's NLL(base) sees the floor.
    floored = np.maximum(p, p.max(axis=-1, keepdims=True) - 16)
    nll = -np.take_along_axis(floored, targets[..., None], -1).mean()
    return float(kl), float(same), float(nll)


def self_test_synthetic() -> None:
    rng = np.random.default_rng(0)
    n_chunk, n_ctx, n_vocab = 3, 12, 1001  # odd vocab exercises the padding element
    n_eval = evaluated_rows(n_ctx)
    tokens = rng.integers(0, n_vocab, (n_chunk, n_ctx), dtype=np.int32)
    logits_base = synthetic_logits(rng, (n_chunk, n_eval, n_vocab))
    logits = logits_base + rng.normal(0.0, 0.05, logits_base.shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "base.bin"
        write_base(path, tokens, logits_base)
        base = BaseFile(path)
        assert (base.n_ctx, base.n_vocab, base.n_chunk, base.first, base.n_eval, base.nv) == (12, 1001, 3, 6, 5, 1006)
        assert path.stat().st_size == 20 + 4 * 12 * 3 + 2 * 1006 * 5 * 3
        assert np.array_equal(base.tokens, tokens)
        assert np.array_equal(base.targets(1), tokens[1, 7:])
        for chunk in range(n_chunk):
            scale, _ = base.fields(chunk)
            decoded = base.log_probs(chunk)
            reference = logits_base[chunk].astype(np.float64)
            reference = reference - reference.max(-1, keepdims=True)
            reference -= np.log(np.exp(reference).sum(-1, keepdims=True))
            live = reference > reference.max(-1, keepdims=True) - 16
            error = np.abs(decoded - reference)[live]
            assert error.max() <= scale.max() * 0.5 + 1e-5, error.max()
        kl_direct, same_direct, nll_direct = direct_metrics(logits_base, tokens[:, n_ctx // 2 + 1 :], logits)
        summary = compare(base, logits)
        assert abs(summary["mean_kld"] - kl_direct) < 2e-4, (summary["mean_kld"], kl_direct)
        assert summary["same_top"] == same_direct, (summary["same_top"], same_direct)
        assert abs(math.log(summary["ppl_base"]) - nll_direct) < 1e-3
        identical = compare(base, logits_base)
        assert identical["mean_kld"] < 1e-6 and identical["same_top"] == 1.0, identical
        candidate_path = Path(directory) / "candidate.bin"
        write_base(candidate_path, tokens, logits)
        via_file = compare(base, BaseFile(candidate_path))
        assert abs(via_file["mean_kld"] - summary["mean_kld"]) < 1e-4
        itself = compare(base, BaseFile(path))
        assert itself["mean_kld"] == 0.0 and itself["ppl"] == itself["ppl_base"] and itself["same_top"] == 1.0
        np.save(Path(directory) / "candidate.npy", logits)
        via_npy = compare(base, open_candidate(Path(directory) / "candidate.npy"))
        assert via_npy == summary
    print(f"synthetic: ok (mean KL {summary['mean_kld']:.6f} vs direct {kl_direct:.6f}, same top {summary['same_top']:.3f})")


def self_test_real(path: Path, corpus: Path, model: Path, tokenizer: str) -> None:
    """Checks on a file produced by llama-perplexity: layout, tokens, row normalization."""
    base = BaseFile(path)
    tokenized = subprocess.run(
        [tokenizer, "-m", str(model), "-f", str(corpus), "--ids", "--log-disable"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    ids = np.array(json.loads(tokenized), dtype=np.int32)
    assert np.array_equal(base.tokens.reshape(-1), ids[: base.n_chunk * base.n_ctx]), "token array mismatch"
    worst = 0.0
    for chunk in range(base.n_chunk):
        # Code 0 is the floor (row max - 16 and below); the coded entries carry the probability mass.
        _, floor = base.fields(chunk)
        decoded = base.log_probs(chunk).astype(np.float64)
        coded = decoded > floor[:, None]
        worst = max(worst, float(np.abs(np.log(np.where(coded, np.exp(decoded), 0.0).sum(axis=1))).max()))
    assert worst < 5e-3, worst
    nll = np.concatenate([-base.log_probs(c)[np.arange(base.n_eval), base.targets(c)] for c in range(base.n_chunk)])
    print(f"real: ok ({base.n_chunk} chunks, tokens match llama-tokenize, max |logsumexp(row)| {worst:.2e}, "
          f"PPL(base) from file {math.exp(float(nll.mean())):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info")
    info.add_argument("base", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("base", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--json", type=Path)
    test = commands.add_parser("self-test")
    test.add_argument("--real", type=Path)
    test.add_argument("--corpus", type=Path)
    test.add_argument("--model", type=Path)
    test.add_argument("--tokenizer", default="llama-tokenize")
    options = parser.parse_args()
    if options.command == "info":
        print(json.dumps(BaseFile(options.base).describe(), indent=2))
    elif options.command == "compare":
        summary = {"base": str(options.base), "candidate": str(options.candidate),
                   **compare(BaseFile(options.base), open_candidate(options.candidate))}
        text = json.dumps(summary, indent=2)
        print(text)
        if options.json:
            options.json.parent.mkdir(parents=True, exist_ok=True)
            options.json.write_text(text + "\n")
    else:
        self_test_synthetic()
        if options.real:
            if options.corpus is None or options.model is None:
                sys.exit("--real requires --corpus and --model")
            self_test_real(options.real, options.corpus, options.model, options.tokenizer)


if __name__ == "__main__":
    main()
