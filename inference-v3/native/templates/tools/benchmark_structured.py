"""Measure cumulative structured parsing; this is not inference acceptance."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter, process_time

from templates import Template
from templates.events import ToolArguments, ToolComplete


def main(args):
    fixtures = Path(__file__).resolve().parents[3] / "native/templates/upstream/models/templates"
    records = []
    result = {
        "platform": platform.platform(),
        "records": records,
        "purpose": "parser scaling observation, not prefill/decode acceptance",
    }
    for family, shapes in (
        ("Qwen-Qwen3-0.6B.jinja", ("string", "whitespace", "nested")),
        ("Qwen3.5-4B.jinja", ("string",)),
        ("google-gemma-4-31B-it.jinja", ("string", "nested")),
    ):
        source = (fixtures / family).read_text()
        for shape in shapes:
            parameters = (
                {"type": "object"}
                if family.startswith("google")
                else {
                    "type": "object",
                    "properties": {"text": {"type": "array" if shape == "nested" else "string"}},
                    "required": ["text"],
                }
            )
            tools = [{"type": "function", "function": {"name": "echo", "parameters": parameters}}]
            with Template(
                source, special_tokens={"bos_token": "<s>", "eos_token": "</s>"}
            ) as template:
                result["native"] = template.build_info.model_dump()
                with template.prepare(
                    [{"role": "user", "content": "hi"}],
                    tools=tools,
                    template_arguments={"enable_thinking": False},
                    now=946684800,
                ) as plan:
                    for size in args.sizes:
                        text = (
                            "a" * size
                            if shape == "string"
                            else [{"x": "a" * 100, "y": [1, 2, 3]} for _ in range(size // 128)]
                        )
                        if shape == "whitespace":
                            text = "ok"
                        if family == "Qwen-Qwen3-0.6B.jinja":
                            output = (
                                "<tool_call>\n"
                                + json.dumps({"name": "echo", "arguments": {"text": text}})
                                + "\n</tool_call>"
                            )
                            if shape == "whitespace":
                                output = output.replace('"text": ', '"text": ' + " " * size)
                        elif family.startswith("google"):
                            value = (
                                '<|"|>' + text + '<|"|>'
                                if isinstance(text, str)
                                else "["
                                + ",".join(
                                    '{x:<|"|>' + item["x"] + '<|"|>,y:[1,2,3]}' for item in text
                                )
                                + "]"
                            )
                            output = "<|tool_call>call:echo{text:" + value + "}<tool_call|>"
                        else:
                            output = (
                                "<tool_call>\n<function=echo>\n<parameter=text>\n"
                                + text
                                + "\n</parameter>\n</function>\n</tool_call>"
                            )
                        encoded = output.encode()
                        events = []
                        started, cpu = perf_counter(), process_time()
                        with plan.stream() as stream:
                            for offset in range(0, len(encoded), args.chunk_bytes):
                                events.extend(
                                    stream.feed(encoded[offset : offset + args.chunk_bytes])
                                )
                            events.extend(stream.finish())
                            parsing_ns = stream.elapsed_ns
                        elapsed, cpu = perf_counter() - started, process_time() - cpu
                        arguments = "".join(
                            event.text for event in events if isinstance(event, ToolArguments)
                        )
                        assert sum(isinstance(event, ToolComplete) for event in events) == 1
                        assert json.loads(arguments) == {"text": text}
                        record = {
                            "family": family,
                            "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "shape": shape,
                            "input_bytes": len(encoded),
                            "chunk_bytes": args.chunk_bytes,
                            "events": len(events),
                            "seconds": elapsed,
                            "cpu_seconds": cpu,
                            "parsing_ns": parsing_ns,
                        }
                        print(json.dumps(record), flush=True)
                        records.append(record)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=256)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32768, 131072, 524288])
    args = parser.parse_args()
    if args.chunk_bytes <= 0 or any(size < 128 or size > 8 * 1024**2 for size in args.sizes):
        parser.error("positive chunks and sizes between 128 bytes and 8 MiB required")
    main(args)
