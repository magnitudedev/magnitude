"""References for rendered fixture sets, built on demand by `magnitude-references run` and cached
on the host by their identity: model, rendered digest and reference implementation.

A reference is built on the CPU alone and admitted to the cache when its logits are finite
(`admit`). The CPU reference is deterministic, so a second CPU build would repeat its bytes and
prove nothing. An accelerator build is not used: it could be admitted only by a CPU build of the
same reference, which takes the time of building on the CPU, and it may differ from that build by
more than rounding wherever the reference makes a discrete choice (a routed expert, a selected row
or proposal token, a trained code: `Reference.continuous`). A near tie that another summation order
flips moves every later row of the fixture by far more than any rounding bound.
A build that is refused (non-finite logits) is named, and its directory is kept for inspection (the
rig reclaims it as it does cached references); an admitted build is cached with its admission in its
record. A build that fails, or is interrupted, is stopped and removed.

A cached reference is reused whatever its sanity result (the result is deterministic). Before it
judges anything it passes one of two gates:
- a model's sanity gate (`check`): a reference whose dense model is not in a sane regime on the
  input is refused;
- a layer slice, told by the provenance record catalog-tool wrote beside it (`kl.mode_of`), has no
  gate beyond its admission: it always fails the
  sanity gate (one layer and the head is not a language model, so its perplexity says nothing).
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from magnitude_references.contract import CHUNK_ROWS, LOGITS, read_logits, read_rendered
from magnitude_references.digest import implementation
from magnitude_references.sanity import Gate

RECORD = "reference.json"


class ReferenceRefused(Exception):
    """The reference failed its gate and must not judge anything."""


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
            "admission": self.record["admission"],
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


def build(rendered: Path, artifacts: Path, build_identity: Path, directory: Path, log: Path,
          variant: str | None) -> None:
    """`magnitude-references run` of `rendered` on the CPU in `variant` (None: the model's numerics)
    into the new `directory`, which then holds a complete reference not yet admitted; a build that
    fails or is interrupted is stopped and removed."""
    directory.parent.mkdir(parents=True, exist_ok=True)
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
        str(directory),
        "--device",
        "cpu",
        *(["--variant", variant] if variant is not None else []),
    ]
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    try:
        code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        shutil.rmtree(directory, ignore_errors=True)
        raise
    # Exit 1 with a record is a failed sanity gate: still a complete, cacheable reference.
    if not (directory / RECORD).is_file():
        shutil.rmtree(directory, ignore_errors=True)
        raise ReferenceFailed(f"magnitude-references run exited {code} without a reference; see {log}")


def admit(directory: Path) -> dict:
    """The admission of the CPU build in `directory` when its logits are finite (module docstring);
    refused otherwise, naming the build."""
    _, values = read_logits(directory, LOGITS)
    if all(np.isfinite(values[start:start + CHUNK_ROWS]).all() for start in range(0, len(values), CHUNK_ROWS)):
        return {"device": "cpu", "agreement": "CPU-deterministic"}
    raise ReferenceRefused(f"reference {directory} (built on cpu) has non-finite logits; it is kept, not admitted")


def obtain(
    rendered: Path, artifacts: Path, build_identity: Path, cache: Path, log: Path, variant: str | None = None
) -> Reference:
    """The cached reference of `rendered` for the model of `artifacts`, in `variant` (None: the model's
    numerics), built and admitted when missing. Its gate is not yet checked (`check`, unless a slice)."""
    rendered_set = read_rendered(rendered)
    implementation_digest = implementation(variant)
    directory = location(cache, rendered_set.model, rendered_set.digest, implementation_digest)
    # Only a reference admitted as a CPU build is cached: a directory there with any other record is
    # replaced.
    stored = json.loads((directory / RECORD).read_text()) if (directory / RECORD).is_file() else {}
    admission = stored.get("admission", {})
    cached = admission.get("device") == "cpu"
    if not cached:
        partial = directory.with_name(f"{directory.name}.partial-{os.getpid()}")
        build(rendered, artifacts, build_identity, partial, log, variant)
        admission = admit(partial)
        record = json.loads((partial / RECORD).read_text())
        (partial / RECORD).write_text(json.dumps({**record, "admission": admission}, indent=1) + "\n")
        if directory.exists():
            shutil.rmtree(directory)
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
