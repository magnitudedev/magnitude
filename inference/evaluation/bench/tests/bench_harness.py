"""Test-only stand-in for ``magnitude-harness render``: one token per character of the
conversation JSON, written as the contract's rendered-fixture lines."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
for flag in ("--artifacts", "--fixtures", "--output", "--build-identity"):
    parser.add_argument(flag, required=True)
arguments = parser.parse_args()
assert arguments.command == "render" and Path(arguments.build_identity).is_file()
lines = []
for line in Path(arguments.fixtures).read_text().splitlines():
    fixture = json.loads(line)
    assert fixture["sizing"] == {"unit": "fixed"}
    body = {"messages": fixture["messages"], **({"tools": fixture["tools"]} if "tools" in fixture else {})}
    lines.append(json.dumps({"fixture_id": fixture["id"], "template_sha256": "0" * 64,
                             "tokens": [0] * len(json.dumps(body))}))
Path(arguments.output, "rendered.jsonl").write_text("".join(line + "\n" for line in lines))
