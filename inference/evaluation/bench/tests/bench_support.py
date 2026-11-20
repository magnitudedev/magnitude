"""Synthetic sessions, template files and renderers shared by the bench tests."""

import json
from pathlib import Path

from magnitude_bench.policy import LANES
from magnitude_fixtures import Session, Source, TemplateFile

#: The `inference/` tree the tests run from, which the tool is given (`--inference`).
TREE = Path(__file__).parents[3]
SOURCE = Source(name="test", digest="0" * 64)

ECHO = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Echo an integer",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    },
}


def call(identity: str, value: int) -> dict:
    return {
        "id": identity,
        "type": "function",
        "function": {"name": "echo", "arguments": json.dumps({"value": value})},
    }


def tools_session(lane: int, history: int = 24) -> Session:
    """A tool session: system head, ``history`` completed echo turns, then the probe."""
    messages: list[dict] = [{"role": "system", "content": f"Session {lane}. Use the tools."}]
    for turn in range(history):
        messages += [
            {"role": "user", "content": f"Call echo with value {turn}.", "tools": [ECHO]},
            {"role": "assistant", "content": "", "tool_calls": [call(f"h{turn}", turn)]},
            {"role": "tool", "tool_call_id": f"h{turn}", "content": "ok"},
        ]
    messages += [
        {"role": "user", "content": "Call echo with value 7."},
        {"role": "assistant", "content": "", "tool_calls": [call("probe", 7)], "scored": True},
    ]
    return Session.model_validate(
        {
            "key": "tools",
            "tools": [ECHO],
            "messages": messages,
            "generation": {
                "prefix": len(messages) - 1,
                "max_tokens": 32768,
                "expected": {"calls": [{"name": "echo", "arguments": {"value": [7]}}]},
            },
            "keep_head": 1,
        }
    )


def template_file(k: int, sessions: list[Session] | None = None) -> TemplateFile:
    """``tools:<k>k`` over the test's tool sessions: the bench's lanes and its
    qualification lane."""
    return TemplateFile(
        template=f"tools:{k}k", context=k * 1024, source=SOURCE,
        sessions=sessions or [tools_session(lane) for lane in range(LANES + 1)],
    )


def template_paths(directory: Path, *ks: int) -> list[Path]:
    """Template files at ``ks`` written into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in ks:
        path = directory / f"tools-{k}k.json"
        path.write_bytes(template_file(k).encode())
        paths.append(path)
    return paths


def body_length(messages: list[dict], tools: list[dict]) -> int:
    """The stand-in harness's rendering: one token per character of the conversation's JSON."""
    return len(json.dumps({"messages": messages, "tools": tools}))


async def character_lengths(bodies):
    """A stand-in renderer: one token per character of the body's JSON."""
    return [len(json.dumps(body)) for body in bodies]
