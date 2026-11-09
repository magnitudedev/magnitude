"""`magnitude-references kl`: a build's logits against the cached reference of their rendered
input (built when missing), and, given the base build's logits on the same tokens, the paired
judgement of the build against the base and whether the two builds' outputs are identical.

The reference is built on the CPU and admitted when it is built (`cache`), then passes its gate: a
model's sanity gate, or, for a layer slice (`mode_of`), none beyond its admission.

A single build has no bar against the reference itself: QUALITY_BAR is relative, and the precision
system admits error against the engine's own default (`inference/docs/precision.md`, which states
no model-level ceiling). Its bar is one of two yardsticks. An independent producer's logits on the
same tokens (another engine, or the reference under another engine's storage): the build is judged
against it as against a base, and is within the yardstick when no metric regressed.

Or the precision system's own storage-resolution bar (`inference/docs/precision.md`, "Storage resolution
supplies the yardstick"; DeepSeek V4.1's D11): the reference built once more over its stored values moved
by one step of their formats (the `perturbed:<seed>` variant, `perturbation.py`, cached by its own
identity beside the reference). Its KL to the reference, at the build's rows, is the bar's distribution. The
build is within it when its median, p90 and p99 KL are each at most α² times the perturbed reference's, and
its worst row at most α² times the perturbed reference's worst row: α² because a local ratio r ≤ α costs
r² at model level by the square law. A zero bar admits no nonzero KL.

Writes one JSON summary. The builds' logits outputs are deleted once judged (they are gigabytes);
references stay in the host's cache, and the yardstick is the caller's.
"""

import json
from pathlib import Path
from typing import Literal

from magnitude_references import quality, rendering
from magnitude_references.cache import Reference, check, obtain
from magnitude_references.contract import read_artifacts

QUALITY_BAR = 0.02  # relative change a metric must exceed, beyond its paired 95% interval
ALPHA = 8  # the engine's standard α (precision.md)
RESOLUTION = ("median_kl", "p90_kl", "p99_kl", "max_kl")  # the quantiles the storage-resolution bar bounds
Mode = Literal["model", "slice"]


def mode_of(artifacts: Path) -> Mode:
    """`slice` when the target of `artifacts` is a `catalog-tool slice` output, whose provenance record
    beside it (`<target>.json`) names the target's SHA-256; `model` otherwise."""
    acquired = read_artifacts(artifacts)
    provenance = acquired.target.with_suffix(".json")
    sliced = provenance.is_file() and json.loads(provenance.read_text()).get("sha256") == acquired.target_sha256
    return "slice" if sliced else "model"


def run(logits: Path, artifacts: Path, build_identity: Path, cache: Path, output: Path,
        base: Path | None, own_rendering: Path | None, yardstick: Path | None,
        variant: str | None = None, perturbed: int | None = None) -> dict:
    """`perturbed`: the seed of the storage-resolution bar to judge the build by (module docstring)."""
    mode = mode_of(artifacts)
    if perturbed is not None and variant is not None:
        raise ValueError(f"the storage-resolution bar perturbs the model's numerics, not the {variant!r} variant")
    measured = base if base is not None else logits
    rendered = measured / rendering.RENDERED
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = obtain(rendered, artifacts, build_identity, cache, output.with_name("reference.log"), variant)
    # The bar's reference is obtained before anything is judged, so a failure to build it keeps the build's logits.
    noise = None if perturbed is None else obtain(rendered, artifacts, build_identity, cache,
                                                  output.with_name("reference-perturbed.log"), f"perturbed:{perturbed}")
    if mode == "model":
        check(reference)
        if noise is not None:
            check(noise)
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
        if noise is not None:
            summary["resolution"] = resolution(quality.compare_outputs(reference.directory, noise.directory)
                                               .at(positions.rows), summary["build"], noise)
        if own_rendering is not None:
            summary["rendering_differences"] = rendering.compare_files(rendered, own_rendering)
    finally:
        for directory in (logits, base):
            for values in directory.glob("*.f32") if directory is not None else ():
                values.unlink()
    output.write_text(json.dumps(summary, indent=1))
    return summary


def resolution(noise: quality.Positions, build: dict, reference: Reference) -> dict:
    """The build's summary against the storage-resolution bar: the perturbed `reference`'s KL `noise` at the
    build's rows (module docstring)."""
    measured = {**quality.summarize(noise.kl, noise.agree), "worst": quality.worst(noise)}
    bounds = {metric: {"build": build[metric], "perturbed": measured[metric], "limit": ALPHA ** 2 * measured[metric],
                       "within": build[metric] <= ALPHA ** 2 * measured[metric]} for metric in RESOLUTION}
    return {"reference": reference.to_json(), "alpha": ALPHA, "perturbed": measured, "bounds": bounds,
            "within": all(bound["within"] for bound in bounds.values())}
