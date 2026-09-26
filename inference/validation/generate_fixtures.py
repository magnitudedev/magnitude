#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Generate historical reference fixtures; optionally run Cargo afterward.

uv run inference/validation/generate_fixtures.py
uv run inference/validation/generate_fixtures.py --test -- -p seismic-engine
Outputs live under ignored validation/results/fixtures; commit generators only.
No model downloads, GPU execution, or benchmark results are required/generated.
"""
import argparse
from contextlib import contextmanager, nullcontext
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "fixtures"
REPOSITORY = ROOT.parents[1]
REFERENCE_COMMIT = "1fb31d00c548b2da9b5c496ffc7f7df6155c7dda"
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


@contextmanager
def pinned_source():
    """Materialize the independent V3 oracle without requiring its working tree."""
    archive = subprocess.run(
        ["git", "-C", str(REPOSITORY), "archive", "--format=tar", REFERENCE_COMMIT, "inference-v3/src"],
        check=True, capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="magnitude-v3-reference-") as temporary:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as entries:
            entries.extractall(temporary, filter="data")
        yield Path(temporary) / "inference-v3"


def generate(source, output):
    output.mkdir(parents=True, exist_ok=True)
    # Complete every generator before replacing any existing reference files.
    with tempfile.TemporaryDirectory(prefix=".generating-", dir=output) as temp:
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
            os.replace(Path(temp) / name, output / name)
            print(f"Generated {output / name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help="Override the pinned historical V3 source")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--test", action="store_true", help="Run cargo test with arguments after --")
    parser.add_argument("cargo_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cargo_args and not args.test:
        parser.error("Cargo arguments require --test")
    if args.test and args.output.resolve() != OUTPUT.resolve():
        parser.error("--test requires the default fixture output directory")
    if args.source is None:
        source_context = pinned_source()
    else:
        source_context = nullcontext(args.source.resolve(strict=True))
    with source_context as source:
        if not (source / "src/ops/tensor/ops.py").is_file():
            parser.error(f"{source} is not an inference-v3 source directory")
        generate(source, args.output)
    if args.test:
        cargo_args = args.cargo_args
        if cargo_args[:1] == ["--"]:
            cargo_args = cargo_args[1:]
        return subprocess.call(["cargo", "test", *cargo_args], cwd=ROOT.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
