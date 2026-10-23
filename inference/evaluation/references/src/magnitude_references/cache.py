"""References for rendered fixture sets, built on demand by `magnitude-references run` and cached
on the host by their identity: model, rendered digest and reference implementation.

A cached reference is reused whatever its sanity result (the result is deterministic), and a
reference whose sanity gate failed is refused for comparison.
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from magnitude_references.contract import read_rendered
from magnitude_references.digest import implementation
from magnitude_references.sanity import Gate

RECORD = "reference.json"


class ReferenceRefused(Exception):
    """The reference failed its sanity gate and must not judge anything."""


class ReferenceFailed(Exception):
    """`magnitude-references run` produced no reference."""


@dataclass(frozen=True)
class Reference:
    directory: Path
    record: dict
    cached: bool

    def to_json(self) -> dict:
        return {
            "directory": str(self.directory),
            "cached": self.cached,
            "family": self.record["family"],
            "implementation_digest": self.record["implementation_digest"],
            "sanity": self.record["sanity"],
        }


def location(cache: Path, model: str, rendered_digest: str, implementation_digest: str) -> Path:
    return (
        cache / model.replace(":", "_").replace("/", "_") / rendered_digest / implementation_digest
    )


def check(reference: Reference) -> Reference:
    """The reference itself when its sanity gate passed; refused otherwise."""
    sanity = reference.record["sanity"]
    if not sanity["passed"]:
        raise ReferenceRefused(
            f"reference {reference.directory} failed its sanity gate (perplexity "
            f"{sanity['perplexity']:.3g}, top-1 {sanity['top1_accuracy']:.3f}; failing fixtures "
            f"{', '.join(Gate.failing(sanity))}): it must not be used for comparison"
        )
    return reference


def obtain(
    rendered: Path, artifacts: Path, build_identity: Path, cache: Path, log: Path
) -> Reference:
    """The cached reference of `rendered` for the model of `artifacts`, built when missing.
    Its sanity gate is not yet checked (`check`)."""
    rendered_set = read_rendered(rendered)
    implementation_digest = implementation()
    directory = location(cache, rendered_set.model, rendered_set.digest, implementation_digest)
    cached = (directory / RECORD).is_file()
    if not cached:
        partial = directory.with_name(f"{directory.name}.partial-{os.getpid()}")
        partial.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(Path(sys.executable).with_name("magnitude-references")),
            "run",
            "--rendered",
            str(rendered),
            "--artifacts",
            str(artifacts),
            "--build-identity",
            str(build_identity),
            "--output",
            str(partial),
        ]
        with log.open("w") as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        # Exit 1 with a record is a failed sanity gate: still a complete, cacheable reference.
        if not (partial / RECORD).is_file():
            shutil.rmtree(partial, ignore_errors=True)
            raise ReferenceFailed(
                f"magnitude-references run exited {code} without a reference; see {log}"
            )
        partial.rename(directory)
    record = json.loads((directory / RECORD).read_text())
    expected = {
        "model": rendered_set.model,
        "rendered_digest": rendered_set.digest,
        "implementation_digest": implementation_digest,
    }
    for name, value in expected.items():
        if record[name] != value:
            raise ReferenceFailed(
                f"cached reference {directory} records {name} {record[name]!r}, not {value!r}"
            )
    return Reference(directory, record, cached)
