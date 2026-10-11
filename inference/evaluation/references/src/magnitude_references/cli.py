"""magnitude-references: independent float32 references of the model families.

  run            reference logits + sanity gate for a rendered fixture set (exit 1 when the gate fails)
  check-hf       a family reference against transformers on a real GGUF
  check-weights  a family reference's weights against the released checkpoint
  check-draft    the DFlash/DSpark draft reference against the released draft code
  check-families every family reference on synthetic variation models (network: template headers)
  header         a GGUF's metadata and tensor directory
  slice          a truncated real-weight GGUF

Add `--device cuda|mps|cpu` (default auto). Matmuls run in full float32 (TF32 disabled). The check
commands need the `check` extra.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from magnitude_references import slice as slicing
from magnitude_references.checks import check_hf, check_weights, device_of
from magnitude_references.digest import sha256_file
from magnitude_references.families import open_reference
from magnitude_references.gguf import HEADER_CACHE, Package, describe
from magnitude_references.performance import BATCH_TOKENS
from magnitude_references.run import Options, build_reference, default_output
from magnitude_references.sanity import PERPLEXITY_MAX, TOP1_MIN, Bounds, Gate


def emit(report: dict, path: Path | None) -> None:
    text = json.dumps(report, indent=1)
    print(text)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="magnitude-references", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="reference logits and sanity gate for rendered fixtures")
    run.add_argument("--rendered", type=Path, required=True, help="harness `render` output (JSON Lines)")
    run.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition object (JSON)")
    run.add_argument("--output", type=Path, help="run directory (default: evaluation/results/<stamp>-reference-<id>)")
    run.add_argument("--perplexity-max", type=float, default=PERPLEXITY_MAX)
    run.add_argument("--top1-min", type=float, default=TOP1_MIN)
    run.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS, help="padded tokens per streamed forward")
    run.add_argument("--cache-weights", action="store_true", help="keep dequantized weights resident")
    run.add_argument("--build-identity", type=Path,
                     help="contract build object {commit, dirty_digest} (JSON), recorded verbatim in result.json")

    for name in ("check-hf", "check-weights"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--layers", type=int, help="run only the first N layers (then the final norm and head)")
    commands.choices["check-hf"].add_argument("--prompts", type=int, default=3)
    commands.choices["check-hf"].add_argument("--length", type=int, default=48)
    commands.choices["check-hf"].add_argument("--seed", type=int, default=20260927)
    commands.choices["check-weights"].add_argument("--checkpoint", type=Path, required=True)

    draft = commands.add_parser("check-draft", help="draft reference against the released draft code")
    draft.add_argument("--draft", type=Path, required=True)
    draft.add_argument("--target", type=Path, required=True)
    draft.add_argument("--context", type=int, default=24, help="context rows (random features)")
    draft.add_argument("--released-code", choices=["muse", "zlab", "dspark"], required=True)
    draft.add_argument("--source-path", type=Path, help="pinned Z-Lab model.py or directory with DSpark dflash.py/dspark.py")
    draft.add_argument("--source-config", type=Path, help="exact upstream checkpoint config for causal overrides")
    draft.add_argument("--rows", type=int, help="physical block rows; default full trained block")
    draft.add_argument("--seed", type=int, default=20260927)

    families = commands.add_parser("check-families", help="every family reference on synthetic models")
    families.add_argument("--cases", help="comma-separated case names (default: all)")
    families.add_argument("--zlab-source", type=Path,
                          help="Z-Lab dflash/model.py at z-lab/dflash@07ebd93d (families.dflash.ZLAB_CODE) for DFlash2")
    families.add_argument("--list", action="store_true")

    for command in commands.choices.values():
        command.add_argument("--device", default="auto")
        command.add_argument("--json", type=Path, help="also write the report here")

    header = commands.add_parser("header", help="a GGUF's metadata and tensor directory")
    header.add_argument("path")
    header.add_argument("--repository")
    header.add_argument("--revision")

    slicing.add_arguments(commands.add_parser("slice", help="a truncated real-weight GGUF", description=slicing.__doc__,
                                              formatter_class=argparse.RawDescriptionHelpFormatter))
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    options = root.parse_args(argv)
    if options.command == "header":
        package = (Package.remote(options.repository, options.revision, options.path, HEADER_CACHE)
                   if options.repository else Package.local(Path(options.path)))
        print(json.dumps(describe(package), indent=1))
        return 0
    if options.command == "slice":
        slicing.run(root, options)
        return 0
    device = device_of(options.device)
    if options.command == "run":
        output = options.output or default_output(sha256_file(options.rendered))
        build = None
        if options.build_identity:
            build = json.loads(options.build_identity.read_text())
            if set(build) != {"commit", "dirty_digest"}:
                root.error(f"{options.build_identity}: a build identity has exactly commit and dirty_digest")
        reference = build_reference(options.rendered, options.artifacts, output,
                                    Options(device, Bounds(options.perplexity_max, options.top1_min),
                                            options.batch_tokens, options.cache_weights, build))
        emit({"output": str(output), **reference}, options.json)
        if not reference["sanity"]["passed"]:
            print(f"sanity gate failed on {Gate.failing(reference['sanity'])}: this reference must not be used "
                  "for comparison", file=sys.stderr)
            return 1
        return 0
    if options.command in ("check-hf", "check-weights"):
        reference = open_reference(options.model, device, True, options.layers)
        if options.command == "check-hf":
            report = check_hf(reference, options.prompts, options.length, options.seed)
        else:
            report = check_weights(reference, options.checkpoint)
        report = {"architecture": reference.architecture, "model": str(options.model), **report}
        emit({k: v for k, v in report.items() if k != "rows"}, None)
        if options.json:
            options.json.parent.mkdir(parents=True, exist_ok=True)
            options.json.write_text(json.dumps(report, indent=1) + "\n")
        return 0 if report.get("pass", True) else 1
    if options.command == "check-draft":
        from magnitude_references.families.dflash import check_draft, open_draft
        draft = open_draft(options.draft, options.target, device)
        report = {"draft": str(options.draft), "target": str(options.target),
                  **check_draft(draft, options.context, options.seed, options.rows, options.released_code,
                                options.source_path, options.source_config)}
        emit(report, options.json)
        return 0 if report["pass"] else 1
    from magnitude_references.synthetic import check_families, selected
    names = None if options.cases is None else set(options.cases.split(","))
    if options.list:
        targets, drafts = selected(names)
        for case in [*targets, *drafts]:
            print(f"{case.name}: {'; '.join(case.covers)}")
        return 0
    reports = check_families(names, device, options.zlab_source)
    emit({"cases": reports, "pass": all(report["pass"] for report in reports)}, options.json)
    return 0 if all(report["pass"] for report in reports) else 1


def entry() -> None:
    raise SystemExit(main())
