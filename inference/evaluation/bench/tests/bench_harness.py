"""Test-only stand-in for ``magnitude-harness render``: sizes every lane of every template file
by the contract rule at one token per character of the conversation JSON, and writes the
contract's rendered lines."""

import argparse
import hashlib
import json
from pathlib import Path

from bench_support import body_length
from magnitude_fixtures import TemplateFile

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--template", action="append", required=True)
for flag in ("--artifacts", "--output", "--build-identity"):
    parser.add_argument(flag, required=True)
arguments = parser.parse_args()
assert arguments.command == "render" and Path(arguments.build_identity).is_file()
lines = []
for path in arguments.template:
    file = TemplateFile.read(Path(path))
    for lane, session in enumerate(file.sessions):
        conversation = session.sized(file.context, body_length)
        history = next(h for h in range(len(session.units()))
                       if session.conversation_with(h) == conversation)
        tokens = [0] * body_length(conversation.messages, conversation.tools)
        lines.append(json.dumps({
            "template": file.template, "lane": lane, "chat_template_sha256": "0" * 64,
            "context": file.context, "history": history, "tokens": tokens,
            "prompt_sha256": hashlib.sha256(json.dumps(conversation.messages).encode()).hexdigest(),
            "generation_prefix_tokens": body_length(conversation.messages[: conversation.prefix],
                                                    conversation.tools),
        }))
Path(arguments.output, "rendered.jsonl").write_text("".join(line + "\n" for line in lines))
