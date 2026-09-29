#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""D4 precision reference and llama.cpp backend spread.

D4 reference: dequantize once to F32, then run the CPU reference on it (one base
file per corpus category):
  uv run inference-v4/validation/precision/reference.py f32-gguf --model $G --output $G_F32
  uv run inference-v4/validation/precision/reference.py base --model $G_F32 --source-model $G --chunks 171,171,170 --label cpu-f32

CPU forward on the quantized file itself (activations quantized to Q8_K/Q8_0):
  uv run inference-v4/validation/precision/reference.py base --model $G --chunks 171,171,170 --label cpu-q4km

Spread (a GPU backend against a reference directory):
  uv run inference-v4/validation/precision/reference.py spread --model $G --reference results/precision/<ref> --label metal [--save-base]

Outputs go under validation/results/precision/<label>/ (git-ignored). Every
run records the exact commands, llama.cpp version, corpus/model sha256 and the
full logs. See README.md for the file format and metric definitions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "precision"
CATEGORIES = ("prose", "code", "tool_json")
N_CTX = 130  # n_ctx - 1 - n_ctx/2 = 64 evaluated positions per chunk

sys.path.insert(0, str(ROOT))
import kl_base  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def binary(directory: Path | None, name: str) -> str:
    return str(directory / name) if directory else name


def llama_version(directory: Path | None) -> str:
    output = subprocess.run([binary(directory, "llama-perplexity"), "--version"], capture_output=True, text=True)
    return next(line for line in (output.stdout + output.stderr).splitlines() if line.startswith("version:"))


def run_logged(command: list[str], log: Path) -> float:
    print("+", " ".join(command), flush=True)
    started = time.monotonic()
    with log.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise SystemExit(f"command failed ({completed.returncode}); see {log}")
    return elapsed


def chunk_counts(text: str) -> dict[str, int]:
    counts = [int(value) for value in text.split(",")]
    if len(counts) == 1:
        counts *= len(CATEGORIES)
    if len(counts) != len(CATEGORIES):
        raise SystemExit(f"--chunks takes one count or {len(CATEGORIES)} comma-separated counts")
    return dict(zip(CATEGORIES, counts))


def selected_categories(text: str) -> tuple[str, ...]:
    names = tuple(text.split(","))
    unknown = [name for name in names if name not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories {unknown}; choose from {', '.join(CATEGORIES)}")
    return names


def corpus_manifest(corpus: Path) -> dict:
    manifest = json.loads((corpus / "manifest.json").read_text())
    for name in CATEGORIES:
        entry = manifest["categories"][name]
        if sha256_file(corpus / entry["file"]) != entry["sha256"]:
            raise SystemExit(f"{corpus / entry['file']} does not match its manifest; rebuild the corpus")
    return manifest


def host_record(directory: Path | None) -> dict:
    return {"host": socket.gethostname(), "platform": platform.platform(), "llama_cpp": llama_version(directory),
            "binary_dir": str(directory) if directory else "PATH"}


def f32_gguf(options) -> None:
    if options.output.exists():
        raise SystemExit(f"{options.output} exists")
    command = [binary(options.binary_dir, "llama-quantize"), "--allow-requantize",
               str(options.model), str(options.output), "F32"]
    options.output.parent.mkdir(parents=True, exist_ok=True)
    elapsed = run_logged(command, options.output.with_suffix(".quantize.log"))
    record = {"source": str(options.model), "source_sha256": sha256_file(options.model),
              "output": str(options.output), "output_sha256": sha256_file(options.output),
              "bytes": options.output.stat().st_size, "command": command, "seconds": elapsed,
              **host_record(options.binary_dir)}
    options.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def base(options) -> None:
    manifest = corpus_manifest(options.corpus)
    output = RESULTS / options.label
    output.mkdir(parents=True, exist_ok=True)
    counts = chunk_counts(options.chunks)
    selected = selected_categories(options.categories)
    record = {"kind": "reference", "n_ctx": options.n_ctx, "model": str(options.model),
              "model_sha256": sha256_file(options.model),
              "source_model": str(options.source_model or options.model),
              "corpus": manifest, **host_record(options.binary_dir), "categories": {}}
    if options.source_model:
        record["source_model_sha256"] = sha256_file(options.source_model)
    existing_path = output / "reference.json"
    if existing_path.exists():
        # Completing a reference category by category: keep the categories already produced,
        # but only from the same model, corpus and llama.cpp build.
        existing = json.loads(existing_path.read_text())
        for key in ("model_sha256", "corpus", "llama_cpp"):
            if existing[key] != record[key]:
                raise SystemExit(f"{existing_path}: {key} differs from this run; use a new --label")
        record["categories"] = {name: entry for name, entry in existing["categories"].items()
                                if name not in selected}
    for name in selected:
        target = output / f"{name}.bin"
        command = [binary(options.binary_dir, "llama-perplexity"), "-m", str(options.model),
                   "-f", str(options.corpus / f"{name}.txt"), "-c", str(options.n_ctx), "--chunks", str(counts[name]),
                   "--kl-divergence-base", str(target), "-ngl", "0", "-dev", "none", "-fa", "off",
                   "-ctk", options.cache_type, "-ctv", options.cache_type, *options.extra]
        elapsed = run_logged(command, output / f"{name}.log")
        written = kl_base.BaseFile(target)
        if (written.n_ctx, written.n_chunk) != (options.n_ctx, counts[name]):
            raise SystemExit(f"{target}: n_ctx={written.n_ctx}, n_chunk={written.n_chunk}; corpus too short?")
        record["categories"][name] = {"base": target.name, "command": command, "seconds": elapsed,
                                      **written.describe(), "sha256": sha256_file(target)}
        (output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


FINAL = {
    "mean_ppl_q": r"Mean PPL\(Q\)\s*:\s*(\S+)\s*±\s*(\S+)",
    "mean_ppl_base": r"Mean PPL\(base\)\s*:\s*(\S+)\s*±\s*(\S+)",
    "mean_ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*(\S+)\s*±\s*(\S+)",
    "mean_kld": r"Mean\s+KLD:\s*(\S+)\s*±\s*(\S+)",
    "max_kld": r"Maximum KLD:\s*(\S+)",
    "kld_99_9": r"99\.9%\s+KLD:\s*(\S+)",
    "kld_99": r"99\.0%\s+KLD:\s*(\S+)",
    "kld_95": r"95\.0%\s+KLD:\s*(\S+)",
    "median_kld": r"Median\s+KLD:\s*(\S+)",
    "rms_p_diff_percent": r"RMS Δp\s*:\s*(\S+)\s*±\s*(\S+)\s*%",
    "same_top_percent": r"Same top p:\s*(\S+)\s*±\s*(\S+)\s*%",
}


def parse_kl_output(text: str) -> dict:
    parsed = {}
    for key, pattern in FINAL.items():
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"llama-perplexity output lacks {key!r} (fewer than 100 positions?)")
        parsed[key] = float(match.group(1))
        if match.lastindex == 2:
            parsed[f"{key}_uncertainty"] = float(match.group(2))
    return parsed


def spread(options) -> None:
    reference = json.loads((options.reference / "reference.json").read_text())
    output = RESULTS / options.label
    output.mkdir(parents=True, exist_ok=True)
    record = {"kind": "spread", "reference": str(options.reference), "reference_llama_cpp": reference["llama_cpp"],
              "model": str(options.model), "model_sha256": sha256_file(options.model),
              **host_record(options.binary_dir), "categories": {}}
    if record["model_sha256"] != reference["model_sha256"] and record["model_sha256"] != reference.get("source_model_sha256"):
        raise SystemExit("spread model differs from the reference model and its source model")
    selected = selected_categories(options.categories)
    existing_path = output / "spread.json"
    if existing_path.exists():
        # One category per run (e.g. one GPU-lock hold each): keep the label's other categories.
        existing = json.loads(existing_path.read_text())
        for key in ("reference", "model_sha256", "llama_cpp"):
            if existing[key] != record[key]:
                raise SystemExit(f"{existing_path}: {key} differs from this run; use a new --label")
        record["categories"] = {name: entry for name, entry in existing["categories"].items()
                                if name not in selected}
    for name in selected:
        base_path = options.reference / f"{name}.bin"
        command = [binary(options.binary_dir, "llama-perplexity"), "-m", str(options.model), "-c", str(reference["n_ctx"]),
                   "--kl-divergence", "--kl-divergence-base", str(base_path), "-ngl", str(options.ngl),
                   *options.extra]
        log = output / f"{name}.log"
        elapsed = run_logged(command, log)
        parsed = parse_kl_output(log.read_text())
        count = kl_base.BaseFile(base_path).n_chunk * kl_base.BaseFile(base_path).n_eval
        entry = {"command": command, "seconds": elapsed, "positions": count, **parsed}
        if options.save_base:
            own = output / f"{name}.bin"
            own_command = [binary(options.binary_dir, "llama-perplexity"), "-m", str(options.model),
                           "-f", str(options.corpus / f"{name}.txt"), "-c", str(reference["n_ctx"]),
                           "--chunks", str(kl_base.BaseFile(base_path).n_chunk),
                           "--kl-divergence-base", str(own), "-ngl", str(options.ngl), *options.extra]
            run_logged(own_command, output / f"{name}.base.log")
            entry["base"] = {"file": own.name, "command": own_command,
                             "comparator": kl_base.compare(kl_base.BaseFile(base_path), kl_base.BaseFile(own))}
        record["categories"][name] = entry
        (output / "spread.json").write_text(json.dumps(record, indent=2) + "\n")
    if set(record["categories"]) == set(CATEGORIES):
        entries = record["categories"].values()
        positions = sum(entry["positions"] for entry in entries)
        record["overall"] = {
            "positions": positions,
            "mean_kld": sum(entry["mean_kld"] * entry["positions"] for entry in entries) / positions,
            "same_top": sum(round(entry["same_top_percent"] / 100 * entry["positions"]) for entry in entries) / positions,
        }
    (output / "spread.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("f32-gguf", "base", "spread"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--binary-dir", type=Path, help="llama.cpp binary directory (default: PATH)")
        if name == "f32-gguf":
            sub.add_argument("--output", type=Path, required=True)
            continue
        sub.add_argument("--label", required=True)
        sub.add_argument("--corpus", type=Path, default=RESULTS / "corpus")
        sub.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra llama-perplexity arguments")
        sub.add_argument("--categories", default=",".join(CATEGORIES),
                         help="comma-separated categories to (re)produce; others already in the "
                              "label's reference.json / spread.json are kept (default: all)")
        if name == "base":
            sub.add_argument("--chunks", required=True, help="chunks per category: N or prose,code,tool_json")
            sub.add_argument("--source-model", type=Path, help="quantized GGUF an F32 --model was derived from")
            sub.add_argument("--cache-type", default="f32", help="KV cache type for K and V (default f32)")
            sub.add_argument("--n-ctx", type=int, default=N_CTX,
                             help=f"chunk length; the second half is scored (default {N_CTX}, the D4 set; "
                                  "longer chunks check long-context numerics such as quantized KV)")
        else:
            sub.add_argument("--reference", type=Path, required=True)
            sub.add_argument("--ngl", type=int, default=99)
            sub.add_argument("--save-base", action="store_true",
                             help="also write this backend's own base files and compare them with kl_base")
    options = parser.parse_args()
    {"f32-gguf": f32_gguf, "base": base, "spread": spread}[options.command](options)


if __name__ == "__main__":
    main()
