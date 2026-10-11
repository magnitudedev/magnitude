"""Build a reference: the true next-token distributions predicting every scored token of a rendered
fixture set (the row at position i-1 for scored token i), with the sanity gate.

Input is only harness `render` output plus the catalog acquisition object, so the reference sees
exactly the tokens our engine serves; it never tokenizes. Output (one directory):
- `logits.f32` / `logits.json`: the contract logits output, producer `reference:<family>`;
- `reference.json`: the reference record with the sanity gate;
- `result.json`: the `evaluation-result/1` record (tool `reference`).

Fixtures run in batches of at most `batch_tokens` padded tokens (right-padding is exact: every family
is causal). With `cache_weights=False` each batch streams the model layer by layer, dequantizing every
layer's weights when it runs and dropping them after, so memory holds one layer plus activations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

from magnitude_references import digest, record
from magnitude_references.contract import (LogitsWriter, RenderedFixture, check_match, read_artifacts,
                                           read_rendered)
from magnitude_references.families import family_of, open_reference
from magnitude_references.gguf import Package
from magnitude_references.model import F32, Reference
from magnitude_references.performance import BATCH_TOKENS, HEAD_ROWS
from magnitude_references.sanity import Bounds, Gate

LOGITS = "logits"
RESULTS = Path(__file__).resolve().parents[3] / "results"  # inference/evaluation/results


@dataclass(frozen=True)
class Options:
    device: torch.device
    bounds: Bounds = Bounds()
    batch_tokens: int = BATCH_TOKENS
    cache_weights: bool = False
    build: dict | None = None  # the contract `build` object, recorded verbatim; None: the local tree's (record.build)


def batches(fixtures: tuple[RenderedFixture, ...], budget: int) -> list[list[RenderedFixture]]:
    """Consecutive fixtures whose padded size (count x longest) stays within `budget`; a longer fixture
    runs alone."""
    groups: list[list[RenderedFixture]] = []
    for fixture in fixtures:
        if groups:
            group = groups[-1]
            longest = max(len(fixture.tokens), *(len(f.tokens) for f in group))
            if (len(group) + 1) * longest <= budget:
                group.append(fixture)
                continue
        groups.append([fixture])
    return groups


def run_batch(reference: Reference, group: list[RenderedFixture], writer: LogitsWriter, sanity: Gate) -> None:
    device = reference.weights.device
    length = max(len(fixture.tokens) for fixture in group)
    tokens = torch.zeros(len(group), length, dtype=torch.int64)
    for row, fixture in enumerate(group):
        tokens[row, :len(fixture.tokens)] = torch.tensor(fixture.tokens)
    with torch.no_grad():
        hidden, _ = reference.hidden(tokens.to(device))
        for row, fixture in enumerate(group):
            positions = fixture.positions()
            selected = hidden[row, torch.tensor(positions, device=device)]
            parts = [reference.logits(selected[start:start + HEAD_ROWS]).to("cpu", F32)
                     for start in range(0, len(positions), HEAD_ROWS)]
            logits = torch.cat(parts)
            writer.write(fixture.fixture_id, positions, logits.numpy())
            sanity.add(fixture.fixture_id, logits, torch.tensor(fixture.targets(), dtype=torch.int64))


def build_reference(rendered_path: Path, artifacts_path: Path, output: Path, options: Options) -> dict:
    """Write the reference for a rendered fixture set into `output`; returns the reference record.
    The record's `sanity.passed` says whether the reference may be used."""
    started = record.utc_now()
    rendered = read_rendered(rendered_path)
    artifacts = read_artifacts(artifacts_path)
    check_match(rendered, artifacts)
    package = Package.local(artifacts.target)
    family = family_of(package)
    reference = open_reference(package, options.device, options.cache_weights)
    for fixture in rendered.fixtures:
        if max(fixture.tokens) >= reference.vocab_size:
            raise ValueError(f"{fixture.fixture_id}: token beyond the model's {reference.vocab_size} vocabulary")
    output.mkdir(parents=True, exist_ok=True)
    writer = LogitsWriter(output, LOGITS, f"reference:{family}", artifacts.model, rendered.digest, reference.vocab_size)
    sanity = Gate()
    groups = batches(rendered.fixtures, options.batch_tokens)
    for index, group in enumerate(groups, 1):
        run_batch(reference, group, writer, sanity)
        print(f"batch {index}/{len(groups)}: {len(group)} fixtures, {sanity.count} scored rows", file=sys.stderr, flush=True)
    writer.close()
    reference_record = {
        "model": artifacts.model, "rendered_digest": rendered.digest, "family": family,
        "implementation_digest": digest.implementation(), "dtype": "float32", "sanity": sanity.result(options.bounds),
    }
    (output / "reference.json").write_text(json.dumps(reference_record, indent=1) + "\n")
    result = {
        "schema": "evaluation-result/1", "tool": "reference", "started_at": started, "finished_at": record.utc_now(),
        "build": options.build if options.build is not None else record.build(), "model": artifacts.model, "artifacts": artifacts.record,
        "fixture_set": {"path": None, "digest": rendered.fixture_set_digest}, "rendered_digest": rendered.digest,
        "host": record.host(options.device), "command": sys.argv, "measurements": {}, "traces": {},
    }
    (output / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return reference_record


def default_output(rendered_digest: str) -> Path:
    return RESULTS / f"{record.utc_stamp()}-reference-{rendered_digest[:8]}"
