"""Reproduce native cumulative stream CPU observations without an inference runtime."""

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

from templates import Template

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--chunk-bytes", type=int, default=256)
parser.add_argument("--sizes", type=int, nargs="+", default=[32768, 131072, 524288])
args = parser.parse_args()
if args.chunk_bytes <= 0 or any(size <= 0 or size > 8 * 1024 * 1024 for size in args.sizes):
    parser.error("positive chunk size and stream sizes up to 8 MiB required")
root = Path(__file__).resolve().parents[3]
source = (root / "native/templates/upstream/models/templates/Qwen-Qwen3-0.6B.jinja").read_text()
records = []
with Template(source) as template:
    for workload in ("content", "reasoning", "tool"):
        tools = (
            []
            if workload != "tool"
            else [
                {
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "parameters": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                }
            ]
        )
        with template.prepare(
            [{"role": "user", "content": "hello"}], tools=tools, now=946684800
        ) as plan:
            for length in args.sizes:
                text = "a" * length
                output = (
                    text
                    if workload == "content"
                    else "<think>" + text + "</think>answer"
                    if workload == "reasoning"
                    else "<tool_call>\n"
                    + json.dumps({"name": "echo", "arguments": {"text": text}})
                    + "\n</tool_call>"
                ).encode()
                started = time.perf_counter()
                count = 0
                with plan.stream() as stream:
                    for offset in range(0, len(output), args.chunk_bytes):
                        count += len(stream.feed(output[offset : offset + args.chunk_bytes]))
                    count += len(stream.finish())
                elapsed = time.perf_counter() - started
                record = {
                    "workload": workload,
                    "bytes": len(output),
                    "chunk_bytes": args.chunk_bytes,
                    "events": count,
                    "seconds": elapsed,
                }
                print(json.dumps(record), flush=True)
                records.append(record)
    result = {
        "native": template.build_info.model_dump(),
        "platform": platform.platform(),
        "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "records": records,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "purpose": "cumulative parser CPU scaling; single observation, not performance acceptance",
    }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
