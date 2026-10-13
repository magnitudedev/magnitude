"""Synthetic fixtures and renderers shared by the bench tests."""

import json

from magnitude_fixtures import Fixture

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


def tools_fixture(lane: int, history: int = 12) -> Fixture:
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
    return Fixture.model_validate(
        {
            "id": f"tools/test/{lane}",
            "family": "tools",
            "source": {"name": "test", "digest": "0" * 64},
            "tools": [ECHO],
            "messages": messages,
            "generation": {
                "prefix": len(messages) - 1,
                "max_tokens": 32768,
                "expected": {"calls": [{"name": "echo", "arguments": {"value": [7]}}]},
            },
            "sizing": {"unit": "turn", "keep_head": 1},
        }
    )


async def character_lengths(bodies):
    """A stand-in renderer: one token per character of the body's JSON."""
    return [len(json.dumps(body)) for body in bodies]
