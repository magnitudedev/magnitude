"""magnitude-references: independent float32 references of the model families.

  run            reference logits + sanity gate for a rendered fixture set (exit 1 when the gate fails)
  kl             a build's logits against the cached reference (and against a base build's)

`mrig kl` runs `kl`, which builds a missing reference as `run` does; the references' own checks
against released code and checkpoints are `tests/test_released.py`. A GGUF's header and a
truncated real-weight GGUF are `catalog-tool header` and `catalog-tool slice`. `run` takes
`--device cuda|mps|cpu` (default auto). Matmuls run in full float32 (TF32 disabled).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from magnitude_references.checks import device_of
from magnitude_references.digest import sha256_file
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
    run.add_argument("--device", default="auto")
    run.add_argument("--json", type=Path, help="also write the report here")

    kl = commands.add_parser("kl", help="a build's logits against the cached reference")
    kl.add_argument("--logits", type=Path, required=True, help="the build's harness `logits` run directory")
    kl.add_argument("--base", type=Path, help="the base build's `logits` run directory, on the same tokens")
    kl.add_argument("--own-rendering", type=Path, help="the build's own `render` of the fixtures, to compare")
    kl.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition object (JSON)")
    kl.add_argument("--build-identity", type=Path, required=True, help="identity of the reference tools' tree")
    kl.add_argument("--cache", type=Path, required=True, help="the host's reference cache")
    kl.add_argument("--output", type=Path, required=True, help="the summary (JSON)")
    kl.add_argument("--keep-logits", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    options = root.parse_args(argv)
    if options.command == "kl":
        from magnitude_references import kl
        summary = kl.run(options.logits, options.artifacts, options.build_identity, options.cache,
                         options.output, options.base, options.own_rendering, options.keep_logits)
        print(json.dumps({key: value for key, value in summary.items() if key != "reference"}, indent=1))
        return 0
    output = options.output or default_output(sha256_file(options.rendered))
    build = None
    if options.build_identity:
        build = json.loads(options.build_identity.read_text())
        if set(build) != {"commit", "dirty_digest"}:
            root.error(f"{options.build_identity}: a build identity has exactly commit and dirty_digest")
    reference = build_reference(options.rendered, options.artifacts, output,
                                Options(device_of(options.device), Bounds(options.perplexity_max, options.top1_min),
                                        options.batch_tokens, options.cache_weights, build))
    emit({"output": str(output), **reference}, options.json)
    if not reference["sanity"]["passed"]:
        print(f"sanity gate failed on {Gate.failing(reference['sanity'])}: this reference must not be used "
              "for comparison", file=sys.stderr)
        return 1
    return 0


def entry() -> None:
    raise SystemExit(main())
