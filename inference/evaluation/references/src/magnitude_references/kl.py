"""`magnitude-references kl`: a build's logits against the cached reference of their rendered
input (built when missing), and, given the base build's logits on the same tokens, the paired
judgement of the build against the base and whether the two builds' outputs are identical.

The reference is built on the CPU and admitted when it is built (`cache`), then passes its gate: a
model's sanity gate, or, for a layer slice (`mode_of`), none beyond its admission. In either mode it
was computed where a row's bits are its prefix's alone (`cache.obtain`): a build's rows come from
chunked prefill and single-row decode, and are judged against rows computed in one batch.

A single build has no bar against the reference itself: QUALITY_BAR is relative, and the precision
system admits error against the engine's own default (`inference/docs/precision.md`, which states
no model-level ceiling). Its bar is one of two yardsticks. An independent producer's logits on the
same tokens (another engine, or the reference under another engine's storage): the build is judged
against it as against a base, and is within the yardstick when no metric regressed.

Or the precision system's own storage-resolution bar (`inference/docs/precision.md`, "Storage resolution
supplies the yardstick"; DeepSeek V4.1's D11 as amended 2026-11-10): the reference built again over its stored
values moved by one step of their formats, once per draw (the `perturbed:<seed>` variant, `perturbation.py`, each
cached by its own identity beside the reference). Each row's bar is α² times the largest KL to the reference over
the draws: α² because a local ratio r ≤ α costs r² at model level by the square law, and the largest because one
draw's KL can fall far below the perturbation's own spread at a row. A row is judgeable while its bar is below
ln V - H, the KL of a uniform guess from the reference's distribution there (entropy H): a bar at or past it would
admit the guess. Unjudgeable rows are reported apart with their top-1 agreement. The build is within the bar when it
has judgeable rows and every one is within its own bar; a zero bar admits no nonzero KL. Each row over is reported
with its ratio KL / bar and whether its top-1 flipped.

The perturbation's KL at a row is heavy-tailed: on near-certain rows one draw's can sit three decades past the
others', so a build that is itself pure perturbation noise can be over its bar at a few such rows, and fails: an
open defect of the bar (`inference/docs/precision.md`, "Open defect: the per-row storage-resolution bar rejects
noise").

Writes one JSON summary. The builds' logits outputs are deleted once judged (they are gigabytes);
references stay in the host's cache, and the yardstick is the caller's.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from magnitude_rig.host.identity import recorded_sha256

from magnitude_references import quality, rendering
from magnitude_references.cache import Reference, check, obtain
from magnitude_references.contract import LOGITS, read_artifacts, read_logits

QUALITY_BAR = 0.02  # relative change a metric must exceed, beyond its paired 95% interval
ALPHA = 8  # the engine's standard α (precision.md)
QUANTILES = {"median": 0.5, "p90": 0.9, "p99": 0.99, "max": 1.0}  # of the ratios KL / bar reported
Mode = Literal["model", "slice"]


def mode_of(artifacts: Path) -> Mode:
    """`slice` when the target of `artifacts` is a layer slice, whose provenance record beside it (`<target>.json`,
    written by `catalog-tool slice`, or by its `slice-record` for another tool's slice; `identity.recorded_sha256`)
    names the target's SHA-256; `model` otherwise. A record that names no SHA-256 is refused."""
    acquired = read_artifacts(artifacts)
    return "slice" if recorded_sha256(acquired.target) == acquired.target_sha256 else "model"


def obtain_all(rendered: Path, artifacts: Path, build_identity: Path, cache: Path, logs: Path,
               variant: str | None = None, perturbed: tuple[int, ...] = ()) -> tuple[Reference, list[Reference]]:
    """The cached reference of `rendered` and the storage-resolution bar's draw of each seed in `perturbed`, each
    built and admitted when missing (`obtain`), their build logs in `logs`: the references a judgement reads, and
    all that depends only on the model's stored values and the rendered tokens."""
    reference = obtain(rendered, artifacts, build_identity, cache, logs / "reference.log", variant)
    draws = [obtain(rendered, artifacts, build_identity, cache, logs / f"reference-perturbed-{seed}.log",
                    f"perturbed:{seed}") for seed in perturbed]
    return reference, draws


def run(logits: Path, artifacts: Path, build_identity: Path, cache: Path, output: Path,
        base: Path | None, own_rendering: Path | None, yardstick: Path | None,
        variant: str | None = None, perturbed: tuple[int, ...] = ()) -> dict:
    """`perturbed`: the seeds of the storage-resolution bar's draws to judge the build by (module docstring)."""
    mode = mode_of(artifacts)
    if perturbed and variant is not None:
        raise ValueError(f"the storage-resolution bar perturbs the model's numerics, not the {variant!r} variant")
    measured = base if base is not None else logits
    rendered = measured / rendering.RENDERED
    output.parent.mkdir(parents=True, exist_ok=True)
    # The bar's references are obtained before anything is judged, so a failure to build one keeps the build's logits.
    reference, draws = obtain_all(rendered, artifacts, build_identity, cache, output.parent, variant, perturbed)
    if mode == "model":
        for judging in (reference, *draws):
            check(judging)
    summary: dict = {"reference": reference.to_json(), "gate": {"mode": mode}}
    try:
        positions = quality.compare_outputs(reference.directory, logits)
        summary["build"] = {**quality.summarize(positions.kl, positions.agree),
                            "worst": quality.worst(positions)}
        if base is not None:
            before = quality.compare_outputs(reference.directory, base)
            summary["base"] = {**quality.summarize(before.kl, before.agree), "worst": quality.worst(before)}
            summary["changes"] = [change.to_json() for change in quality.judge(before, positions, QUALITY_BAR)]
            between = quality.compare_outputs(base, logits)
            builds = quality.summarize(between.kl, between.agree)
            summary["between_builds"] = builds
            summary["builds_identical"] = builds["max_kl"] == 0.0 and builds["top1_agreement"] == 1.0
        if yardstick is not None:
            independent = quality.compare_outputs(reference.directory, yardstick)
            summary["yardstick"] = {**quality.summarize(independent.kl, independent.agree),
                                    "worst": quality.worst(independent)}
            changes = quality.judge(independent, positions, QUALITY_BAR)
            summary["yardstick_changes"] = [change.to_json() for change in changes]
            summary["within_yardstick"] = all(change.verdict != "regressed" for change in changes)
        if draws:
            summary["resolution"] = resolution(reference, draws, positions)
        if own_rendering is not None:
            summary["rendering_differences"] = rendering.compare_files(rendered, own_rendering)
    finally:
        for directory in (logits, base):
            for values in directory.glob("*.f32") if directory is not None else ():
                values.unlink()
    output.write_text(json.dumps(summary, indent=1))
    return summary


def resolution(reference: Reference, draws: list[Reference], build: quality.Positions) -> dict:
    """The build's rows against the storage-resolution bar of the perturbed references `draws` (module docstring)."""
    perturbed = [quality.compare_outputs(reference.directory, draw.directory).at(build.rows) for draw in draws]
    header, values = read_logits(reference.directory, LOGITS)
    entropy = quality.entropy(values, quality.matched(header["rows"], build.rows))
    return {"draws": [{"reference": draw.to_json(), **quality.summarize(noise.kl, noise.agree),
                       "worst": quality.worst(noise)} for draw, noise in zip(draws, perturbed)],
            **within_resolution(build.rows, entropy, header["vocabulary"], np.stack([noise.kl for noise in perturbed]),
                                build.kl, build.agree)}


def bars(entropy: np.ndarray, vocabulary: int, perturbed_kl: np.ndarray) -> np.ndarray:
    """Each row's storage-resolution bar, α² times the largest of the draws' KL there (`perturbed_kl`, [draws, rows]),
    NaN where it is not below ln V - H, the row unjudgeable (module docstring)."""
    bar = ALPHA ** 2 * perturbed_kl.max(axis=0)
    return np.where(bar < math.log(vocabulary) - entropy, bar, np.nan)


def ratios(kl: np.ndarray, bar: np.ndarray) -> np.ndarray:
    """KL / bar per row: a zero bar admits only a zero KL, and any other is infinitely over it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(bar == 0, np.where(kl == 0, 0.0, np.inf), kl / bar)


def within_resolution(rows: Sequence[dict], entropy: np.ndarray, vocabulary: int, perturbed_kl: np.ndarray,
                      kl: np.ndarray, agree: np.ndarray) -> dict:
    """A build's per-row KL and top-1 agreement against the storage-resolution bar of the draws' KL `perturbed_kl`
    [draws, rows], the other arrays aligned by row: within when it has judgeable rows and each is within its own bar
    (`bars`). `over` lists the judgeable rows past their bar, each by its identity among `rows`, with its ratio
    KL / bar and whether its top-1 flipped. The ratios' quantiles are order statistics (the upper one where a quantile
    falls between two), so an infinite ratio never interpolates. `worst` is the judgeable row of largest ratio."""
    bar = bars(entropy, vocabulary, perturbed_kl)
    judgeable = ~np.isnan(bar)
    judged = np.flatnonzero(judgeable)
    ratio = ratios(kl[judged], bar[judged])
    over = [{**rows[row], "ratio": float(r), "top1_flipped": not bool(agree[row])}
            for row, r in zip(judged, ratio) if r > 1]
    return {"alpha": ALPHA, "vocabulary": vocabulary,
            "rows": {"judgeable": len(judged), "unjudgeable": int((~judgeable).sum())},
            "ratio": {name: float(np.quantile(ratio, q, method="higher")) for name, q in QUANTILES.items()}
            if len(judged) else None,
            "over": over,
            "unjudgeable_top1_agreement": float(agree[~judgeable].mean()) if not judgeable.all() else None,
            "worst": rows[int(judged[ratio.argmax()])] if len(judged) else None,
            "within": bool(len(judged) and not over)}
