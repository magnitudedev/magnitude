"""Quality: per-position KL divergence and top-1 agreement between two logits outputs, their
contract summary, and the noise-aware judgement of a candidate against a base.

`KL(P || Q)` takes P as the truth: the reference when judging an engine, the base when judging
the candidate against it. Both outputs must be over the same rendered input, and the tested
output's rows, matched by fixture and position, must be distinct and among the truth's (contract
"Logits output"). An output whose header names no `sample` scored every row, so its rows must be
exactly the truth's; one that names its sample scored the rows that sample chose, which its
producer, the one owner of the selection rule, lists.

The judgement is paired: the base and the candidate are measured against the same reference at
the same positions, so the noise of a metric's change is estimated by resampling positions
(a paired bootstrap). A change counts only when its 95% interval excludes zero and it moves
the metric by more than the relative bar.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from magnitude_references.contract import CHUNK_ROWS, LOGITS, read_logits

TRIM = 0.001  # trimmed KL drops the top 0.1% of positions
RESAMPLES = 1000
SEED = 20261009
WORST = 5
METRICS = ("mean_kl", "trimmed_kl", "p99_kl", "max_kl", "top1_agreement")
Verdict = Literal["improved", "regressed", "within noise"]


class Incomparable(ValueError):
    """Two logits outputs are not over the same rendered input, or their rows do not correspond."""


@dataclass(frozen=True)
class Positions:
    """Per-position KL and top-1 agreement of one pair, aligned with the logits rows."""

    rows: tuple[dict, ...]  # {"fixture_id", "position"}
    kl: np.ndarray
    agree: np.ndarray

    def at(self, rows: tuple[dict, ...]) -> "Positions":
        """These values at `rows`, which must all be among this pair's rows."""
        index = matched(self.rows, rows)
        return Positions(rows, self.kl[index], self.agree[index])


def matched(rows: tuple[dict, ...] | list[dict], subset: tuple[dict, ...] | list[dict]) -> np.ndarray:
    """The index into `rows` of each row of `subset`, matched by fixture and position."""
    where = {(row["fixture_id"], row["position"]): index for index, row in enumerate(rows)}
    missing = [row for row in subset if (row["fixture_id"], row["position"]) not in where]
    if missing:
        raise Incomparable(f"{len(missing)} rows are not among the rows compared against: first {missing[0]}")
    return np.fromiter(
        (where[row["fixture_id"], row["position"]] for row in subset), dtype=np.intp, count=len(subset)
    )


def _log_softmax(values: np.ndarray) -> np.ndarray:
    peak = values.max(axis=1, keepdims=True)
    shifted = values - peak
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def compare_rows(
    truth: np.ndarray, test: np.ndarray, truth_rows: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """KL(truth || test) and argmax agreement per row of `test`, a [rows, vocabulary] logits
    array, against row `truth_rows[i]` of `truth` (its row `i` when `truth_rows` is None)."""
    if truth_rows is None:
        truth_rows = np.arange(test.shape[0])
    kl = np.empty(test.shape[0])
    agree = np.empty(test.shape[0], dtype=bool)
    for start in range(0, test.shape[0], CHUNK_ROWS):
        p = np.asarray(truth[truth_rows[start : start + CHUNK_ROWS]], dtype=np.float64)
        q = np.asarray(test[start : start + CHUNK_ROWS], dtype=np.float64)
        for side, values in (("truth", p), ("test", q)):
            finite = np.isfinite(values).all(axis=1)
            if not finite.all():
                rows = start + np.flatnonzero(~finite)
                raise ValueError(
                    f"non-finite {side} logits in {len(rows)} of rows {start}..{start + len(p)}: first {rows[:8].tolist()}"
                )
        log_p, log_q = _log_softmax(p), _log_softmax(q)
        # Rounding can leave an exact-zero divergence a hair below zero.
        kl[start : start + len(p)] = np.maximum((np.exp(log_p) * (log_p - log_q)).sum(axis=1), 0.0)
        agree[start : start + len(p)] = p.argmax(axis=1) == q.argmax(axis=1)
    return kl, agree


def entropy(logits: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """The softmax's entropy in nats of rows `rows` of `logits` [rows, vocabulary] (all of them when None)."""
    if rows is None:
        rows = np.arange(logits.shape[0])
    out = np.empty(len(rows))
    for start in range(0, len(rows), CHUNK_ROWS):
        log_p = _log_softmax(np.asarray(logits[rows[start : start + CHUNK_ROWS]], dtype=np.float64))
        out[start : start + len(log_p)] = -(np.exp(log_p) * log_p).sum(axis=1)
    return out


def compare_outputs(truth: Path, test: Path) -> Positions:
    """Per-position comparison of the logits outputs in two directories (`logits.json/.f32`)."""
    truth_header, truth_values = read_logits(truth, LOGITS)
    test_header, test_values = read_logits(test, LOGITS)
    for key in ("rendered_digest", "vocabulary"):
        if truth_header[key] != test_header[key]:
            raise Incomparable(
                f"{truth} ({truth_header['producer']}) and {test} ({test_header['producer']}) "
                f"differ in {key}"
            )
    _distinct(truth, truth_header["rows"])
    rows = tuple(test_header["rows"])
    _distinct(test, rows)
    index = matched(truth_header["rows"], rows)
    if test_header.get("sample") is None and len(rows) != len(truth_header["rows"]):
        raise Incomparable(
            f"{test} names no sample but scores {len(rows)} of the {len(truth_header['rows'])} rows of {truth}"
        )
    kl, agree = compare_rows(truth_values, test_values, index)
    return Positions(rows, kl, agree)


def _distinct(directory: Path, rows: tuple[dict, ...] | list[dict]) -> None:
    keys = {(row["fixture_id"], row["position"]) for row in rows}
    if len(keys) != len(rows):
        raise Incomparable(f"{directory} repeats {len(rows) - len(keys)} of its rows")


def _sorted_summary(kl_sorted: np.ndarray, agree_mean: float) -> dict:
    n = len(kl_sorted)
    kept = kl_sorted[: n - int(n * TRIM)]
    return {
        "mean_kl": float(kl_sorted.mean()),
        "trimmed_kl": float(kept.mean()),
        "p99_kl": float(np.percentile(kl_sorted, 99)),
        "max_kl": float(kl_sorted[-1]),
        "top1_agreement": agree_mean,
    }


def summarize(kl: np.ndarray, agree: np.ndarray) -> dict:
    """The contract `quality` measurement of per-position values."""
    if len(kl) == 0:
        raise ValueError("no scored positions")
    ordered = np.sort(kl)
    return {
        **_sorted_summary(ordered, float(agree.mean())),
        "positions": len(kl),
        "median_kl": float(np.median(ordered)),
        "p90_kl": float(np.percentile(ordered, 90)),
        "mean_kl_without_10_worst": float(ordered[:-10].mean()) if len(ordered) > 10 else None,
        "positions_above_0_5_nats": int(np.count_nonzero(ordered > 0.5)),
    }


def worst(positions: Positions, count: int = WORST) -> list[dict]:
    """The positions of largest divergence: diagnostic evidence, never a measurement."""
    order = np.argsort(positions.kl)[::-1][:count]
    return [
        {
            **positions.rows[index],
            "kl": float(positions.kl[index]),
            "agree": bool(positions.agree[index]),
        }
        for index in order
    ]


@dataclass(frozen=True)
class MetricChange:
    metric: str
    base: float
    candidate: float
    delta: float
    relative: float
    interval: tuple[float, float]  # 95% bootstrap interval of the delta
    verdict: Verdict

    def to_json(self) -> dict:
        return {
            "metric": self.metric,
            "base": self.base,
            "candidate": self.candidate,
            "delta": self.delta,
            "relative": self.relative,
            "interval": list(self.interval),
            "verdict": self.verdict,
        }


def judge(base: Positions, candidate: Positions, bar: float) -> list[MetricChange]:
    """Each metric of the candidate against the base, both measured against the same truth, at
    the candidate's positions (all of which the base must have measured)."""
    base = base.at(candidate.rows)
    n = len(base.kl)
    base_summary, candidate_summary = (
        summarize(base.kl, base.agree),
        summarize(candidate.kl, candidate.agree),
    )
    generator = np.random.default_rng(SEED)
    deltas = {metric: np.empty(RESAMPLES) for metric in METRICS}
    for resample in range(RESAMPLES):
        chosen = generator.integers(0, n, n)
        before = _sorted_summary(np.sort(base.kl[chosen]), float(base.agree[chosen].mean()))
        after = _sorted_summary(
            np.sort(candidate.kl[chosen]), float(candidate.agree[chosen].mean())
        )
        for metric in METRICS:
            deltas[metric][resample] = after[metric] - before[metric]
    changes = []
    for metric in METRICS:
        before, after = base_summary[metric], candidate_summary[metric]
        delta = after - before
        relative = delta / before if before else (0.0 if delta == 0 else float("inf"))
        low, high = (float(value) for value in np.percentile(deltas[metric], [2.5, 97.5]))
        significant = low > 0 or high < 0
        better = delta > 0 if metric == "top1_agreement" else delta < 0
        verdict: Verdict = "within noise"
        if significant and abs(relative) > bar:
            verdict = "improved" if better else "regressed"
        changes.append(MetricChange(metric, before, after, delta, relative, (low, high), verdict))
    return changes
