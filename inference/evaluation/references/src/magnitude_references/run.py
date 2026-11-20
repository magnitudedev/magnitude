"""Build a reference: the true next-token distributions predicting every scored token of a template's
rendered sessions (the row at position i-1 for scored token i), with the sanity gate.

Input is only harness `render` output plus the catalog acquisition object, so the reference sees
exactly the tokens our engine serves; it never tokenizes. Output (one directory):
- `logits.f32` / `logits.json`: the contract logits output, producer `reference:<family>`;
- `reference.json`: the reference record with the sanity gate;
- `result.json`: the `evaluation-result/1` record (tool `reference`).

Sessions run in batches of at most `batch_tokens` padded tokens (right-padding is exact: every family
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
from magnitude_rig import store
from magnitude_references.contract import (LOGITS, LogitsWriter, RenderedSession, check_match, read_artifacts,
                                           read_rendered)
from magnitude_references.checks import device_of
from magnitude_references.families import family_of, open_reference, reference_of
from magnitude_references.gguf import Package
from magnitude_references.model import F32, Reference
from magnitude_references.performance import BATCH_TOKENS, HEAD_ROWS
from magnitude_references.sanity import Bounds, Gate


@dataclass(frozen=True)
class Options:
    device: torch.device | None  # None: `build_device`
    bounds: Bounds = Bounds()
    batch_tokens: int = BATCH_TOKENS
    cache_weights: bool = False
    build: dict | None = None  # the contract `build` object, recorded verbatim; None: the local tree's (record.build)
    variant: str | None = None  # a measurement variant (`Reference.variants`); None: the model's numerics


def batches(sessions: tuple[RenderedSession, ...], budget: int) -> list[list[RenderedSession]]:
    """Consecutive sessions whose padded size (count x longest) stays within `budget`; a longer session
    runs alone."""
    groups: list[list[RenderedSession]] = []
    for session in sessions:
        if groups:
            group = groups[-1]
            longest = max(len(session.tokens), *(len(s.tokens) for s in group))
            if (len(group) + 1) * longest <= budget:
                group.append(session)
                continue
        groups.append([session])
    return groups


def run_batch(reference: Reference, group: list[RenderedSession], writer: LogitsWriter, sanity: Gate) -> None:
    device = reference.weights.device
    length = max(len(session.tokens) for session in group)
    tokens = torch.zeros(len(group), length, dtype=torch.int64)
    for row, session in enumerate(group):
        tokens[row, :len(session.tokens)] = torch.tensor(session.tokens)
    with torch.no_grad():
        hidden, _ = reference.hidden(tokens.to(device))
        for row, session in enumerate(group):
            positions = session.positions()
            selected = hidden[row, torch.tensor(positions, device=device)]
            parts = [reference.logits(selected[start:start + HEAD_ROWS]).to("cpu", F32)
                     for start in range(0, len(positions), HEAD_ROWS)]
            logits = torch.cat(parts)
            writer.write(session, positions, logits.numpy())
            sanity.add(session.template, session.lane, logits, torch.tensor(session.targets(), dtype=torch.int64))


def build_device(package: Package) -> torch.device:
    """The device a reference of `package` builds on by default: the host's default, or the CPU for
    a family whose reference rounds to trained codes (`Reference.continuous`)."""
    return device_of("auto" if reference_of(package).continuous else "cpu")


def build_reference(rendered_path: Path, artifacts_path: Path, output: Path, options: Options) -> dict:
    """Write the reference for a template's rendered sessions into `output`; returns the reference record.
    The record's `sanity.passed` says whether the reference may be used."""
    started = record.utc_now()
    rendered = read_rendered(rendered_path)
    artifacts = read_artifacts(artifacts_path)
    check_match(rendered, artifacts)
    package = Package.local(artifacts.target)
    family = family_of(package)
    device = options.device or build_device(package)
    if not reference_of(package).continuous and device.type != "cpu":
        raise ValueError(f"the {family} reference rounds to trained codes, so it builds on the CPU alone, not {device}")
    reference = open_reference(package, device, options.cache_weights, variant=options.variant)
    for session in rendered.sessions:
        if max(session.tokens) >= reference.vocab_size:
            raise ValueError(f"{session.name}: token beyond the model's {reference.vocab_size} vocabulary")
    output.mkdir(parents=True, exist_ok=True)
    writer = LogitsWriter(output, LOGITS, f"reference:{family}", artifacts.model, rendered.digest, reference.vocab_size)
    sanity = Gate()
    groups = batches(rendered.sessions, options.batch_tokens)
    for index, group in enumerate(groups, 1):
        run_batch(reference, group, writer, sanity)
        print(f"batch {index}/{len(groups)}: {len(group)} sessions, {sanity.count} scored rows", file=sys.stderr, flush=True)
    writer.close()
    reference_record = {
        "model": artifacts.model, "template": rendered.template, "prompt_sha256": rendered.prompt_sha256,
        "rendered_digest": rendered.digest, "family": family,
        "implementation_digest": digest.implementation(reference_of(package), options.variant, device.type),
        "platform": digest.platform(device.type), "dtype": "float32", "sanity": sanity.result(options.bounds),
    }
    (output / "reference.json").write_text(json.dumps(reference_record, indent=1) + "\n")
    result = {
        "schema": "evaluation-result/1", "tool": "reference", "started_at": started, "finished_at": record.utc_now(),
        "build": options.build if options.build is not None else record.build(), "model": artifacts.model, "artifacts": artifacts.record,
        "template": rendered.template, "prompt_sha256": rendered.prompt_sha256, "rendered_digest": rendered.digest,
        "host": record.host(device), "command": sys.argv, "measurements": {}, "traces": {},
    }
    (output / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return reference_record


def default_output(rendered_digest: str) -> Path:
    """A new run directory in the repository's results store (`magnitude_rig.store`)."""
    return store.results() / f"{record.utc_stamp()}-reference-{rendered_digest[:8]}"
