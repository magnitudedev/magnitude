"""TEST-ONLY helper: a small rendered-fixture JSONL for smoke-testing a reference on real weights.

Not part of the package: real rendered fixtures come only from the harness `render` subcommand (the
engine's own template and tokenizer path). This helper exists because the harness is built in
parallel; it renders chat conversations with the GGUF's own chat template (jinja2, as transformers
renders it) and the model's Hugging Face `tokenizer.json`, and can also emit the same text without any
template (`--raw`), the out-of-distribution negative control the sanity gate must reject.

uv run --with tokenizers --with jinja2 python references/tests/render_chat.py \\
    --gguf M.gguf --tokenizer tokenizer.json --conversations C.json --model ID:VARIANT \\
    --output DIR [--raw]

`C.json` is a list of conversations, each a list of {"role": "user"|"assistant", "content": str}.
Every assistant message is scored. Writes DIR/rendered.jsonl and DIR/artifacts.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from magnitude_references.digest import sha256_file
from magnitude_references.gguf import Package


def render_template(template: str, messages: list[dict], bos: str) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str):
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, indent=None: json.dumps(value, indent=indent, ensure_ascii=False)
    environment.globals["raise_exception"] = raise_exception
    # Rendering is deterministic: templates that print the date see the evaluation's fixed date.
    environment.globals["strftime_now"] = lambda format: datetime(2026, 1, 1).strftime(format)
    return environment.from_string(template).render(messages=messages, bos_token=bos, add_generation_prompt=False)


def char_spans(text: str, messages: list[dict]) -> list[tuple[int, int]]:
    """Character ranges of each assistant message's content, located in order in the rendered text."""
    spans, cursor = [], 0
    for message in messages:
        start = text.find(message["content"].strip(), cursor)
        if start < 0:
            raise ValueError(f"message content not found in the rendered text: {message['content'][:60]!r}")
        end = start + len(message["content"].strip())
        if message["role"] == "assistant":
            spans.append((start, end))
        cursor = end
    return spans


def token_spans(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[list[int]]:
    """Half-open token ranges of the tokens overlapping each character range."""
    result = []
    for start, end in spans:
        inside = [index for index, (a, b) in enumerate(offsets) if b > start and a < end]
        result.append([max(inside[0], 1), inside[-1] + 1])  # token 0 has no predicting row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--conversations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", action="store_true", help="no chat template: BOS, then the contents joined by blank lines")
    options = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(options.tokenizer))
    package = Package.local(options.gguf)
    template = package.metadata["tokenizer.chat_template"].value
    tokens_table = package.metadata["tokenizer.ggml.tokens"].value
    # The template's `bos_token`, and the raw text's first token, only for models that add BOS.
    adds_bos = "tokenizer.ggml.add_bos_token" in package.metadata and package.metadata["tokenizer.ggml.add_bos_token"].value
    bos = tokens_table[package.metadata["tokenizer.ggml.bos_token_id"].value] if adds_bos else ""
    target_sha256 = sha256_file(options.gguf)
    conversations = json.loads(options.conversations.read_text())
    set_digest = sha256_file(options.conversations)
    options.output.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, messages in enumerate(conversations):
        if options.raw:
            text = bos + "\n\n".join(message["content"].strip() for message in messages)
        else:
            text = render_template(template, messages, bos)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        spans = token_spans(encoding.offsets, char_spans(text, messages))
        lines.append({
            "fixture_id": f"smoke/{'raw' if options.raw else 'chat'}/{index}", "fixture_set_digest": set_digest,
            "model": options.model, "target_sha256": target_sha256,
            "template_sha256": hashlib.sha256(b"" if options.raw else template.encode()).hexdigest(),
            "context_target": None, "tokens": encoding.ids, "scored_spans": spans, "generation_prefix_tokens": None,
        })
    (options.output / "rendered.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (options.output / "artifacts.json").write_text(json.dumps({
        "model": options.model, "target": str(options.gguf.resolve()), "target_sha256": target_sha256,
        "drafter": None, "projector": None}, indent=1) + "\n")
    print(json.dumps([{"fixture_id": line["fixture_id"], "tokens": len(line["tokens"]),
                       "scored": sum(e - s for s, e in line["scored_spans"])} for line in lines]))


if __name__ == "__main__":
    main()
