#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "gguf==0.19.0", "torch==2.14.0", "tokenizers", "huggingface-hub"]
# ///
"""D4 base files from the model-definition reference (plan §2.6).

uv run inference/validation/precision/model_base.py --model G.gguf --tokenizer ORG/REPO@REVISION \\
    --label ref-model-<id> --chunks 171,171,170 [--categories code] [--device cuda] [--batch 16] [--cache-weights]

The independent float32 family reference (`validation/<family>_reference.py`, chosen by the GGUF's
`general.architecture`) evaluates the fixed corpus in the D4 chunking and writes llama.cpp
`--kl-divergence-base` files (format in README.md), so V4 (`forward_bench qualify`) and llama.cpp
(`reference.py spread`, including its CPU F32 forward) are both scored against the model definition.

Tokens: the corpus is tokenized with the model's released `tokenizer.json` (the Hugging Face
tokenizer, no special tokens), prefixed with BOS when the model adds BOS, and cut into consecutive
chunks of `n_ctx` tokens. The base file stores those tokens; every consumer reads them from the file,
so V4 and llama.cpp evaluate exactly these tokens. When the vocabulary adds BOS, position 0 of every
chunk is evaluated as BOS (llama-perplexity's rule) while the file keeps the original token.
`add_bos` comes from `tokenizer.ggml.add_bos_token`, or from `--add-bos` where llama.cpp decides
otherwise: without the key it decides from the pre-tokenizer (LFM2 adds BOS), and it forces BOS for
Gemma 4 (the 31B header says false). The record keeps both the GGUF key and the value used.

Large models: without `--cache-weights` every layer's weights are dequantized when the layer runs
and dropped after, over a batch of `--batch` chunks, so memory holds one layer plus the activations.
Progress is appended to `<label>/<category>.rows.partial`; rerunning the same command resumes.
`--chunks` takes the first N chunks per category; `kl_base.py compare` accepts candidates with more
chunks than the base, so a subset base scores full-length candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
import kl_base  # noqa: E402
from model_references import open_reference  # noqa: E402
from reference_cli import device_of  # noqa: E402
from reference_gguf import Package  # noqa: E402

RESULTS = ROOT.parent / "results" / "precision"
CATEGORIES = ("prose", "code", "tool_json")
N_CTX = 130


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(spec: str):
    """`ORG/REPO@REVISION` (tokenizer.json at that revision) or a local tokenizer.json path."""
    from tokenizers import Tokenizer
    if Path(spec).is_file():
        return Tokenizer.from_file(spec), {"file": spec, "sha256": sha256_file(Path(spec))}
    from huggingface_hub import hf_hub_download
    repository, revision = spec.split("@")
    path = hf_hub_download(repository, "tokenizer.json", revision=revision)
    return Tokenizer.from_file(path), {"repository": repository, "revision": revision, "sha256": sha256_file(Path(path))}


def add_bos_of(package: Package, option: str | None) -> bool:
    """`--add-bos` when given (llama.cpp decides from the pre-tokenizer when the key is absent, and
    overrides it for some families, e.g. Gemma 4 always adds BOS), else the GGUF key."""
    key = "tokenizer.ggml.add_bos_token"
    if option is not None:
        return option == "true"
    if key not in package.metadata:
        raise SystemExit(f"the GGUF has no {key}; pass --add-bos true|false (llama.cpp's rule for its pre-tokenizer)")
    return bool(package.metadata[key].value)


def chunk_counts(text: str) -> dict[str, int]:
    counts = [int(value) for value in text.split(",")]
    counts = counts * len(CATEGORIES) if len(counts) == 1 else counts
    if len(counts) != len(CATEGORIES):
        raise SystemExit(f"--chunks takes one count or {len(CATEGORIES)} comma-separated counts")
    return dict(zip(CATEGORIES, counts))


def category(reference, options, name: str, count: int, tokenizer, add_bos: bool, bos: int, output: Path) -> dict:
    text = (options.corpus / f"{name}.txt").read_text()
    stream = ([bos] if add_bos else []) + tokenizer.encode(text, add_special_tokens=False).ids
    if count * options.n_ctx > len(stream):
        raise SystemExit(f"{name}: {len(stream)} tokens hold {len(stream) // options.n_ctx} chunks, {count} requested")
    tokens = np.asarray(stream[:count * options.n_ctx], dtype=np.int32).reshape(count, options.n_ctx)
    first, n_eval = options.n_ctx // 2, kl_base.evaluated_rows(options.n_ctx)
    row_bytes = 2 * kl_base.row_width(reference.vocab_size)
    tokens_path, partial = output / f"{name}.tokens.npy", output / f"{name}.rows.partial"
    if tokens_path.exists():
        if not np.array_equal(np.load(tokens_path), tokens):
            raise SystemExit(f"{tokens_path} differs from this run's tokens; use a new --label")
    else:
        np.save(tokens_path, tokens)
    done = partial.stat().st_size // (n_eval * row_bytes) if partial.exists() else 0
    with partial.open("r+b" if partial.exists() else "wb") as stream_out:
        stream_out.truncate(done * n_eval * row_bytes)
        stream_out.seek(done * n_eval * row_bytes)
        started = time.monotonic()
        for start in range(done, count, options.batch):
            batch = torch.from_numpy(tokens[start:start + options.batch].astype(np.int64)).to(reference.weights.device)
            if add_bos:
                batch[:, 0] = bos
            with torch.no_grad():
                hidden, _ = reference.hidden(batch)
                logits = reference.logits(hidden[:, first:first + n_eval]).to("cpu", torch.float32).numpy()
            for rows in logits:
                stream_out.write(b"".join(kl_base.encode_row(row).tobytes() for row in rows))
            stream_out.flush()
            print(f"{name}: chunks {start + len(batch)}/{count} ({time.monotonic() - started:.0f} s)", flush=True)
    target = output / f"{name}.bin"
    with target.open("wb") as stream_out:
        stream_out.write(kl_base.MAGIC)
        stream_out.write(np.array([options.n_ctx, reference.vocab_size, count], dtype="<i4").tobytes())
        stream_out.write(np.ascontiguousarray(tokens, dtype="<i4").tobytes())
        with partial.open("rb") as rows_in:
            while block := rows_in.read(1 << 24):
                stream_out.write(block)
    written = kl_base.BaseFile(target)
    if (written.n_ctx, written.n_chunk, written.n_vocab) != (options.n_ctx, count, reference.vocab_size):
        raise SystemExit(f"{target}: unexpected layout")
    partial.unlink()
    return {"base": target.name, "tokens": len(stream), **written.describe(), "sha256": sha256_file(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="ORG/REPO@REVISION of the released tokenizer, or tokenizer.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--chunks", required=True, help="chunks per category: N or prose,code,tool_json")
    parser.add_argument("--categories", default=",".join(CATEGORIES))
    parser.add_argument("--corpus", type=Path, default=RESULTS / "corpus")
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16, help="chunks per forward pass")
    parser.add_argument("--cache-weights", action="store_true", help="keep dequantized weights (small models)")
    parser.add_argument("--add-bos", choices=("true", "false"))
    options = parser.parse_args()
    selected = tuple(options.categories.split(","))
    if unknown := [name for name in selected if name not in CATEGORIES]:
        raise SystemExit(f"unknown categories {unknown}")
    manifest = json.loads((options.corpus / "manifest.json").read_text())
    for name in CATEGORIES:
        entry = manifest["categories"][name]
        if sha256_file(options.corpus / entry["file"]) != entry["sha256"]:
            raise SystemExit(f"{options.corpus / entry['file']} does not match its manifest")
    reference = open_reference(options.model, device_of(options.device), options.cache_weights)
    package = reference.package
    add_bos = add_bos_of(package, options.add_bos)
    bos = package.metadata["tokenizer.ggml.bos_token_id"].value
    tokenizer, tokenizer_record = load_tokenizer(options.tokenizer)
    if tokenizer.get_vocab_size(with_added_tokens=True) > reference.vocab_size:
        raise SystemExit("the tokenizer has more tokens than the model's vocabulary")
    output = RESULTS / options.label
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "reference", "producer": "model-definition", "n_ctx": options.n_ctx,
        "model": str(options.model), "model_sha256": sha256_file(options.model),
        "architecture": package.architecture, "reference": type(reference).__module__,
        "torch": torch.__version__, "device": options.device, "tokenizer": tokenizer_record,
        "add_bos": add_bos, "bos": bos, "llama_cpp": "none (model-definition reference)",
        "gguf_add_bos": package.metadata["tokenizer.ggml.add_bos_token"].value
        if "tokenizer.ggml.add_bos_token" in package.metadata else None,
        "host": socket.gethostname(), "platform": platform.platform(), "corpus": manifest, "categories": {},
    }
    existing_path = output / "reference.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        for key in ("model_sha256", "corpus", "tokenizer", "add_bos", "n_ctx"):
            if existing[key] != record[key]:
                raise SystemExit(f"{existing_path}: {key} differs from this run; use a new --label")
        record["categories"] = {name: entry for name, entry in existing["categories"].items() if name not in selected}
    counts = chunk_counts(options.chunks)
    for name in selected:
        started = time.monotonic()
        entry = category(reference, options, name, counts[name], tokenizer, add_bos, bos, output)
        record["categories"][name] = {**entry, "seconds": time.monotonic() - started}
        existing_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
