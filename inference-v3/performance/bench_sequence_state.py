"""Small repeatable prefill/decode/fork benchmark through the public library."""

from __future__ import annotations

import argparse
import cProfile
import faulthandler
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import magnitude
from magnitude import LogitsSelection, ModelRequest, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=int, default=256)
    parser.add_argument("--suffix", type=int, default=8)
    parser.add_argument("--decode", type=int, default=16)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--vocabulary", type=int, default=0)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--step", action="store_true", help="Wait for a newline before each measured sample."
    )
    args = parser.parse_args()
    faulthandler.dump_traceback_later(180, repeat=True)
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    source_root = Path(magnitude.__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        digest.update(("src/" + str(path.relative_to(source_root))).encode())
        digest.update(path.read_bytes())
    result = {
        "model_path": args.model,
        "source_sha256": digest.hexdigest(),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "platform": platform.platform(),
        "tilelang_disable_cache": os.getenv("TILELANG_DISABLE_CACHE", "0"),
        "workload": vars(args) | {"output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with load_model(
        args.model,
        memory_bytes=12 * 1024**3,
        backend="metal",
        context_tokens=1024,
        max_sequences=args.branches + 1,
        batch_tokens=256,
    ) as loaded:
        model = loaded.executor
        result["load_seconds"] = time.perf_counter() - started
        result["compiler"] = model.context.runtime.compiler_identity
        result["runtime"] = model.context.runtime.runtime_identity
        result["artifact"] = str(model.artifact_identity)
        prose = loaded.tokenizer.encode(
            "A small library can run independent requests against shared context. " * args.prefix
        )
        prefix = prose[: args.prefix]
        suffixes = tuple(prose[i : i + args.suffix] for i in range(args.branches))

        def advance(sequence, tokens, read=True):
            options = (
                {"vocabulary": tuple(range(args.vocabulary))} if read and args.vocabulary else {}
            )
            batch = model.prepare(
                (
                    ModelRequest(
                        sequence,
                        tokens,
                        LogitsSelection.LAST if read else LogitsSelection.NONE,
                        **options,
                    ),
                )
            )
            try:
                batch.advances[0].commit()
                return batch.advances[0].read_logits()[0] if read else ()
            finally:
                batch.close()

        def run():
            source = loaded.input(prefix)
            parent = source.open()
            checkpoint = None
            branches = []
            try:
                start = time.perf_counter()
                for i in range(0, len(prefix), 256):
                    advance(parent, prefix[i : i + 256], False)
                prefill = time.perf_counter() - start
                print(f"Prefill: {prefill:.3f}s", flush=True)
                checkpoint = parent.checkpoint()
                start = time.perf_counter()
                for i in range(args.decode):
                    advance(parent, (prose[i],))
                decode = time.perf_counter() - start
                print(f"Decode: {decode:.3f}s", flush=True)
                start = time.perf_counter()
                for _ in range(args.branches):
                    branches.append(checkpoint.fork())
                fork = time.perf_counter() - start
                print(f"Fork: {fork:.3f}s", flush=True)
                options = {"vocabulary": tuple(range(args.vocabulary))} if args.vocabulary else {}
                start = time.perf_counter()
                batch = model.prepare(
                    tuple(
                        ModelRequest(branch, tokens, **options)
                        for branch, tokens in zip(branches, suffixes, strict=True)
                    )
                )
                try:
                    batch.completion.wait()
                    values = [item.read_logits()[0] for item in batch.advances]
                    for item in batch.advances:
                        item.commit()
                finally:
                    batch.close()
                branch = time.perf_counter() - start
                if not all(math.isfinite(value) for row in values for value in row):
                    raise RuntimeError("nonfinite model logits")
                metrics = {
                    "prefill_seconds": prefill,
                    "decode_seconds": decode,
                    "fork_seconds": fork,
                    "branches_seconds": branch,
                    "prefill_tokens_per_second": len(prefix) / prefill,
                    "decode_tokens_per_second": args.decode / decode,
                    "branch_logits_first8": [list(row[:8]) for row in values],
                    "reserved_bytes": model.context.memory.reserved,
                }
                if hasattr(model.states, "occupied_rows"):
                    metrics["occupied_history_rows"] = model.states.occupied_rows
                    metrics["history_capacity_rows"] = model.states.history_capacity
                    metrics["history_allocated_bytes"] = sum(
                        value.allocated_bytes for value in model.states.history
                    )
                return metrics
            finally:
                for branch in branches:
                    branch.close()
                if checkpoint is not None:
                    checkpoint.close()
                parent.close()
                source.close()

        warmup_started = time.perf_counter()
        result["warmup"] = run()
        result["warmup_seconds"] = time.perf_counter() - warmup_started
        args.output.write_text(json.dumps(result, indent=2, default=str))
        print("Warmup complete", flush=True)
        profiler = cProfile.Profile() if args.profile else None
        if profiler is not None:
            profiler.enable()
        result["samples"] = []
        for _ in range(args.repeats):
            if args.step:
                input()
            sample = run()
            result["samples"].append(sample)
            args.output.write_text(json.dumps(result, indent=2, default=str))
            print(
                {key: value for key, value in sample.items() if key != "branch_logits_first8"},
                flush=True,
            )
        result["medians"] = {
            key: statistics.median(sample[key] for sample in result["samples"])
            for key in result["samples"][0]
            if key != "branch_logits_first8"
        }
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(args.profile)
    args.output.write_text(json.dumps(result, indent=2, default=str))
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
