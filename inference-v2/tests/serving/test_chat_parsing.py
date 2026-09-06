import json

import pytest

from magnitude_engine.serving.arguments import decode_call
from magnitude_engine.serving.formats import format_for
from magnitude_engine.serving.parsing import OutputParser, TextDelta, ToolCall

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"},
                    "filters": {"type": "object"},
                    "exact": {"type": "boolean"},
                },
            },
        },
    }
]


def collect(parts, family, prefilled=False):
    parser = OutputParser(format_for(family), TOOLS, reasoning_prefilled=prefilled)
    events = []
    for part in parts:
        events.extend(parser.feed(part))
    events.extend(parser.feed("", final=True))
    # Transport chunk sizes may differ; semantic channel boundaries and calls cannot.
    merged = []
    for event in events:
        if isinstance(event, TextDelta) and merged and isinstance(merged[-1], TextDelta):
            previous = merged[-1]
            if previous.channel == event.channel:
                merged[-1] = TextDelta(event.channel, previous.text + event.text)
                continue
        merged.append(event)
    return merged


@pytest.mark.parametrize(
    "family,text,expected",
    [
        (
            "qwen3_5_moe",
            "<think>choose</think>\n\n<tool_call>\n<function=search>\n"
            "<parameter=query>\na < b, café\n</parameter>\n"
            "<parameter=count>\n3\n</parameter>\n"
            '<parameter=filters>\n{"a": [1, true]}\n</parameter>\n'
            "<parameter=exact>True</parameter>\n</function>\n</tool_call>done",
            [
                TextDelta("reasoning", "choose"),
                ToolCall(
                    0,
                    "search",
                    {
                        "query": "a < b, café",
                        "count": 3,
                        "filters": {"a": [1, True]},
                        "exact": True,
                    },
                ),
                TextDelta("content", "done"),
            ],
        ),
        (
            "qwen3",
            '<tool_call>{"name":"search","arguments":{"query":"hello"}}</tool_call>'
            '<tool_call>{"name":"search","arguments":"{\\"count\\":2}"}</tool_call>',
            [ToolCall(0, "search", {"query": "hello"}), ToolCall(1, "search", {"count": 2})],
        ),
        (
            "gemma4",
            "<|channel>thought\nplan<channel|>\n\n<|tool_call>call:search"
            '{query:<|"|>a,b:{c}\n<|"|>,filters:{nested:[1,true,{x:<|"|>}<|"|>}]}}'
            "<tool_call|>",
            [
                TextDelta("reasoning", "plan"),
                ToolCall(
                    0,
                    "search",
                    {"query": "a,b:{c}\n", "filters": {"nested": [1, True, {"x": "}"}]}},
                ),
            ],
        ),
    ],
)
def test_semantic_output_is_identical_at_every_chunk_boundary(family, text, expected):
    for split in range(len(text) + 1):
        assert collect([text[:split], text[split:]], family) == expected
    assert collect(list(text), family) == expected


def test_prefilled_reasoning_plain_text_and_partial_markers():
    assert collect(list("why</think>\n\n\nanswer<thin"), "qwen3", True) == [
        TextDelta("reasoning", "why"),
        TextDelta("content", "\nanswer<thin"),
    ]
    assert collect(["literal <think> x"], "unknown") == [TextDelta("content", "literal <think> x")]


@pytest.mark.parametrize(
    "wire,body",
    [
        ("json", "[]"),
        ("json", "{}"),
        ("json", '{"name":"missing","arguments":{}}'),
        ("xml", "<function=search><parameter=count>1</parameter>garbage</function>"),
        (
            "xml",
            "<function=search><parameter=count>1</parameter><parameter=count>2</parameter></function>",
        ),
        ("gemma", 'call:search{query:<|"|>unfinished}'),
        ("gemma", "call:search{filters:{x:1,x:2}}"),
        ("gemma", "call:search{} trailing"),
    ],
)
def test_malformed_tool_calls_do_not_become_successful_calls(wire, body):
    with pytest.raises(ValueError):
        decode_call(body, wire, {"search": TOOLS[0]["function"]["parameters"]})


def test_truncated_call_and_parser_lifecycle():
    parser = OutputParser(format_for("qwen3"), TOOLS)
    parser.feed('<tool_call>{"name":')
    with pytest.raises(ValueError, match="inside a tool call"):
        parser.feed("", final=True)
    complete = OutputParser(None, [])
    complete.feed("end", final=True)
    with pytest.raises(RuntimeError, match="closed"):
        complete.feed("again")


def test_xml_python_literals_remain_json_serializable():
    _, arguments = decode_call(
        "<function=search><parameter=filters>{'x': [True, None]}</parameter></function>",
        "xml",
        {"search": TOOLS[0]["function"]["parameters"]},
    )
    assert json.loads(json.dumps(arguments)) == {"filters": {"x": [True, None]}}


def test_gemma_accepts_json_nested_values_without_corrupting_quoted_keys_or_strings():
    _, arguments = decode_call(
        'call:search{filters:{"x:y":["a,b",{"escaped\\"key":"z}"}]}}',
        "gemma",
        {"search": TOOLS[0]["function"]["parameters"]},
    )
    assert arguments == {"filters": {"x:y": ["a,b", {'escaped"key': "z}"}]}}
