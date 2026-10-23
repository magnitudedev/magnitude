"""`magnitude-references kl`: a build's logits against the cached reference of their rendered
input (built when missing), and, given the base build's logits on the same tokens, the paired
judgement of the build against the base and whether the two builds' outputs are identical.

Writes one JSON summary. The logits outputs are deleted once judged (they are gigabytes);
references stay in the host's cache.
"""

import json
from pathlib import Path

from magnitude_references import quality, rendering
from magnitude_references.cache import check, obtain

QUALITY_BAR = 0.02  # relative change a metric must exceed, beyond its paired 95% interval


def run(logits: Path, artifacts: Path, build_identity: Path, cache: Path, output: Path,
        base: Path | None, own_rendering: Path | None, keep_logits: bool) -> dict:
    measured = base if base is not None else logits
    rendered = measured / rendering.RENDERED
    reference = check(obtain(rendered, artifacts, build_identity, cache, output.with_name("reference.log")))
    summary: dict = {"reference": reference.to_json()}
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
        if own_rendering is not None:
            summary["rendering_differences"] = rendering.compare_files(rendered, own_rendering)
    finally:
        if not keep_logits:
            for directory in (logits, base):
                for values in directory.glob("*.f32") if directory is not None else ():
                    values.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=1))
    return summary
