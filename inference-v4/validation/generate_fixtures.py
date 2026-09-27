#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Generate test fixtures from V3; optionally run Cargo after generation.

uv run inference-v4/validation/generate_fixtures.py
uv run inference-v4/validation/generate_fixtures.py --test -- -p seismic-engine
Outputs live under ignored validation/results/fixtures; commit generators only.
No model downloads, GPU execution, or benchmark results are required/generated.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "fixtures"
GENERATORS = (
    ("gguf-codec-reference.json", "gguf_codec_reference.py", ()),
    ("qwen-decoder-reference.json", "qwen_decoder_reference.py", ()),
    ("qwen-recurrent-reference.json", "qwen_recurrent_reference.py", ()),
    ("qwen-rotary-reference.json", "qwen_rotary_reference.py", ()),
    ("qwen-routed-reference.json", "qwen_routed_reference.py", ()),
    ("qwen-routed-decoder-reference.json", "qwen_decoder_reference.py", ("--routed",)),
    ("sampling-v3-reference.json", "sampling_reference.py", ()),
    ("qwen-vision-block-reference.json", "qwen_vision_block_reference.py", ()),
    ("qwen-vision-merger-reference.json", "qwen_vision_merger_reference.py", ()),
    ("erf-gelu-reference.json", "qwen_vision_merger_reference.py", ("--erf",)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=ROOT.parents[1] / "inference-v3")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--test", action="store_true", help="Run cargo test with arguments after --")
    parser.add_argument("cargo_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cargo_args and not args.test:
        parser.error("Cargo arguments require --test")
    if args.test and args.output.resolve() != OUTPUT.resolve():
        parser.error("--test requires the default fixture output directory")
    source = args.source.resolve(strict=True)
    if not (source / "src/ops/tensor/ops.py").is_file():
        parser.error(f"{source} is not an inference-v3 source directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # Complete every generator before replacing any existing reference files.
    with tempfile.TemporaryDirectory(prefix=".generating-", dir=args.output) as temp:
        for name, generator, flags in GENERATORS:
            destination = Path(temp) / name
            subprocess.run([
                sys.executable, str(ROOT / generator), "--source", str(source),
                "--output", str(destination), *flags,
            ], check=True)
            record = json.loads(destination.read_text())
            if name == "erf-gelu-reference.json":
                if not isinstance(record, list) or not record:
                    raise RuntimeError(f"{name}: missing erf/GELU samples")
            elif not record.get("source_sha256") or not record.get("generator_sha256"):
                raise RuntimeError(f"{name}: missing reference provenance")
        for name, _, _ in GENERATORS:
            os.replace(Path(temp) / name, args.output / name)
            print(f"Generated {args.output / name}", flush=True)
    if args.test:
        cargo_args = args.cargo_args
        if cargo_args[:1] == ["--"]:
            cargo_args = cargo_args[1:]
        return subprocess.call(["cargo", "test", *cargo_args], cwd=ROOT.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
