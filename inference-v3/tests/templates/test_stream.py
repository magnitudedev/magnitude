"""Every byte split is a distinct stream schedule over the same prepared plan."""

from pathlib import Path

import pytest

from templates import NativeError, Template
from templates.events import (
    ContentDelta,
    Finish,
    ReasoningDelta,
    ToolArguments,
    ToolComplete,
    ToolStart,
)

FIXTURES = Path(__file__).resolve().parents[2] / "native/templates/upstream/models/templates"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": "Search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
    for name in ("search", "search_more")
]


def assemble(events):
    content, reasoning, calls, terminal = "", "", [], None
    for event in events:
        match event:
            case ContentDelta(text=text):
                content += text
            case ReasoningDelta(text=text):
                reasoning += text
            case ToolStart(index=index, name=name, id=call_id):
                assert index == len(calls), "tool identity must be emitted exactly once"
                calls.append({"name": name, "id": call_id, "arguments": "", "complete": False})
            case ToolArguments(index=index, text=text):
                assert not calls[index]["complete"]
                calls[index]["arguments"] += text
            case ToolComplete(index=index):
                assert not calls[index]["complete"]
                calls[index]["complete"] = True
            case Finish(cause=cause):
                assert terminal is None
                terminal = cause
    return content, reasoning, calls, terminal


@pytest.mark.parametrize(
    "output",
    [
        "Hello 世界 🦙!",
        "<think>réflexion 世界</think>Answer 🦙",
        '<tool_call>\n{"name":"search_more","arguments":{"query":"héllo 世界"}}\n</tool_call>',
        '<think>choose a tool</think><tool_call>\n{"name":"sear'
        'ch","arguments":{"query":"a \\"quote\\""}}\n</tool_call>'
        '<tool_call>\n{"name":"search_more","arguments":{"query":"next"}}\n</tool_call>',
    ],
)
def test_qwen_complete_every_byte_split_and_one_byte_chunks(output):
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            encoded = output.encode()
            expected = assemble(plan.parse(encoded))
            for split in range(len(encoded) + 1):
                with plan.stream() as stream:
                    actual = (
                        stream.feed(encoded[:split])
                        + stream.feed(encoded[split:])
                        + stream.finish()
                    )
                    assert assemble(actual) == expected, split
            with plan.stream() as stream:
                events = []
                for byte in encoded:
                    events.extend(stream.feed(bytes([byte])))
                events.extend(stream.finish())
                assert assemble(events) == expected


def test_tool_arguments_are_published_before_call_completion():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            with plan.stream() as stream:
                events = stream.feed(b'<tool_call>\n{"name":"search","arguments":{"query":"hel')
                assert any(isinstance(event, ToolStart) for event in events)
                assert any(
                    isinstance(event, ToolArguments) and "hel" in event.text for event in events
                )
                assert not any(isinstance(event, ToolComplete) for event in events)
                events += stream.feed(b'lo"}}\n</tool_call>') + stream.finish()
                assert assemble(events)[2][0]["arguments"] == '{"query":"hello"}'


def test_stream_retains_plan_and_terminal_causes_do_not_claim_completion():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        plan = template.prepare([{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800)
        stream = plan.stream()
        plan.close()
    with stream:
        events = stream.feed(b'<tool_call>\n{"name":"search","arguments":{"query":"hel')
        events += stream.finish("length")
        result = assemble(events)
        assert result[2][0]["complete"] is False
        assert result[3] == "length"
        with pytest.raises(NativeError, match="terminal"):
            stream.feed(b"x")


def test_natural_incomplete_tool_errors_but_length_and_user_stop_remain_explicit():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            output = b'<tool_call>\n{"name":"search","arguments":{"query":"hel'
            with pytest.raises(NativeError, match="Naturally completed"):
                plan.parse(output)
            for cause in ("length", "user_stop", "cancelled", "failed"):
                assert assemble(plan.parse(output, cause=cause))[3] == cause


def test_stream_limits_and_utf8_errors_are_local():
    with Template("{% for message in messages %}{{ message.content }}{% endfor %}") as template:
        with template.prepare([{"role": "user", "content": "Hello"}], now=946684800) as plan:
            with plan.stream(max_output_bytes=3) as stream:
                stream.feed(b"ab")
                with pytest.raises(NativeError, match="byte limit"):
                    stream.feed(b"cd")
                with pytest.raises(NativeError, match="terminal"):
                    stream.finish()
            with plan.stream() as stream:
                with pytest.raises(NativeError, match="UTF-8"):
                    stream.feed(b"\xff")
            with plan.stream() as stream:
                assert stream.feed(b"\xe4") == ()
                with pytest.raises(NativeError, match="UTF-8"):
                    stream.finish()
            assert assemble(plan.parse("世界".encode()))[0] == "世界"


@pytest.mark.parametrize(
    "filename,output,expected_reasoning",
    [
        (
            "Qwen3-Coder.jinja",
            "<tool_call>\n<function=search>\n<parameter=query>"
            "\nhéllo 世界\n</parameter>\n</function>\n</tool_call>",
            "",
        ),
        (
            "Qwen3.5-4B.jinja",
            "consider</think>\n\n<tool_call>\n<function=search>\n<paramet"
            "er=query>\nhéllo 世界\n</parameter>\n</function>\n</tool_call>",
            "consider",
        ),
        (
            "openai-gpt-oss-120b.jinja",
            ' to=functions.search<|channel|>commentary json<|message|>{"query":"héllo 世界"}',
            "",
        ),
        (
            "deepseek-ai-DeepSeek-V3.1.jinja",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜"
            '>{"query":"héllo 世界"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
            "",
        ),
        (
            "google-gemma-4-31B-it.jinja",
            '<|tool_call>call:search{query:<|"|>héllo 世界<|"|>}<tool_call|>',
            "",
        ),
    ],
)
def test_real_format_tool_semantics_at_every_byte_split(filename, output, expected_reasoning):
    import json
    from copy import deepcopy

    tools = deepcopy(TOOLS)
    if filename == "google-gemma-4-31B-it.jinja":
        # Upstream Gemma enforces names but only supports unrestricted dictionaries.
        for tool in tools:
            tool["function"]["parameters"] = {"type": "object"}
    with Template((FIXTURES / filename).read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=tools, now=946684800
        ) as plan:
            encoded = output.encode()
            expected = assemble(plan.parse(encoded))
            assert expected[0] == ""
            assert expected[1] == expected_reasoning
            assert len(expected[2]) == 1
            assert expected[2][0]["name"] == "search"
            assert json.loads(expected[2][0]["arguments"]) == {"query": "héllo 世界"}
            for split in range(len(encoded) + 1):
                with plan.stream() as stream:
                    events = (
                        stream.feed(encoded[:split])
                        + stream.feed(encoded[split:])
                        + stream.finish()
                    )
                    assert assemble(events) == expected, split


def test_literal_marker_prefix_at_eof_is_prose_and_required_tools_are_not_optional():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            assert assemble(plan.parse(b"a <tool_ca"))[0] == "a <tool_ca"
        with template.prepare(
            [{"role": "user", "content": "Hello"}],
            tools=TOOLS,
            now=946684800,
            tool_choice="required",
        ) as plan:
            with pytest.raises(NativeError, match="Naturally completed"):
                plan.parse(b"ordinary prose")


def test_nonparallel_plan_rejects_a_second_call():
    output = b'<tool_call>\n{"name":"search","arguments":{"query":"hello"}}\n</tool_call>'
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}],
            tools=TOOLS,
            now=946684800,
            parallel_tool_calls=False,
        ) as plan:
            assert len(assemble(plan.parse(output))[2]) == 1
            with pytest.raises(NativeError):
                plan.parse(output + output)


@pytest.mark.parametrize(
    "filename,output",
    [
        (
            "mistralai-Mistral-Nemo-Instruct-2407.jinja",
            '[TOOL_CALLS][{"name":"search","arguments":{"query":"hello"},"id":"abc123xyz"}]',
        ),
        (
            "Mistral-Small-3.2-24B-Instruct-2506.jinja",
            '[TOOL_CALLS]search[CALL_ID]abc123xyz[ARGS]{"query":"hello"}',
        ),
    ],
)
def test_explicit_tool_ids_before_or_after_arguments_never_change(filename, output):
    with Template(
        (FIXTURES / filename).read_text(), special_tokens={"bos_token": "<s>", "eos_token": "</s>"}
    ) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            encoded = output.encode()
            expected = assemble(plan.parse(encoded))
            assert expected[2][0]["id"] == "abc123xyz"
            for split in range(len(encoded) + 1):
                with plan.stream() as stream:
                    events = (
                        stream.feed(encoded[:split])
                        + stream.feed(encoded[split:])
                        + stream.finish()
                    )
                    assert assemble(events) == expected, split


def test_delimiter_looking_text_inside_json_arguments_is_not_a_tool_boundary():
    import json

    arguments = {"query": 'look </tool_call> <think> "escaped" {nested} 🦙'}
    output = (
        "<tool_call>\n"
        + json.dumps({"name": "search", "arguments": arguments}, ensure_ascii=False)
        + "\n</tool_call>"
    ).encode()
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=TOOLS, now=946684800
        ) as plan:
            with plan.stream() as stream:
                events = []
                for byte in output:
                    events.extend(stream.feed(bytes([byte])))
                events.extend(stream.finish())
                result = assemble(events)
                assert result[:2] == ("", "")
                assert json.loads(result[2][0]["arguments"]) == arguments


def test_two_streams_of_one_plan_have_independent_publication_state():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare([{"role": "user", "content": "Hello"}], now=946684800) as plan:
            with plan.stream() as first, plan.stream() as second:
                events_a = first.feed(b"first ")
                events_b = second.feed(b"second ")
                events_a += first.feed("世界".encode()) + first.finish()
                events_b += second.feed("🦙".encode()) + second.finish()
                assert assemble(events_a)[0] == "first 世界"
                assert assemble(events_b)[0] == "second 🦙"


def test_long_whitespace_reasoning_and_utf8_stream_matches_complete_mapping():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare([{"role": "user", "content": "hi"}], now=946684800) as plan:
            output = ("<think>" + " \n" * 8192 + "é🦙" * 4096 + "</think>answer").encode()
            expected = assemble(plan.parse(output))
            with plan.stream() as stream:
                events = []
                for offset in range(0, len(output), 127):
                    events.extend(stream.feed(output[offset : offset + 127]))
                events.extend(stream.finish())
            assert assemble(events) == expected
            assert expected[0] == "answer" and expected[1].endswith("é🦙" * 4096)


@pytest.mark.parametrize("family", ["Qwen3.5-4B.jinja", "google-gemma-4-31B-it.jinja"])
def test_tagged_argument_spans_preserve_escaping_across_values_and_calls(family):
    import json

    expected = {"first": 'a"\\\n\t世界😀', "empty": "", "last": "tail"}
    parameters = (
        {"type": "object"}
        if family.startswith("google")
        else {
            "type": "object",
            "properties": {key: {"type": "string"} for key in expected},
            "required": list(expected),
        }
    )
    tools = [{"type": "function", "function": {"name": "echo", "parameters": parameters}}]
    if family.startswith("google"):
        output = (
            "<|tool_call>call:echo{"
            + ",".join(key + ':<|"|>' + value + '<|"|>' for key, value in expected.items())
            + "}<tool_call|>"
        )
    else:
        output = (
            "<tool_call>\n<function=echo>\n"
            + "\n".join(
                "<parameter=" + key + ">\n" + value + "\n</parameter>"
                for key, value in expected.items()
            )
            + "\n</function>\n</tool_call>"
        )
    encoded = (output + output).encode()
    with Template((FIXTURES / family).read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}],
            tools=tools,
            now=946684800,
            template_arguments={"enable_thinking": False},
        ) as plan:
            for chunk in (1, 3, 17, len(encoded)):
                with plan.stream() as stream:
                    events = []
                    for offset in range(0, len(encoded), chunk):
                        events.extend(stream.feed(encoded[offset : offset + chunk]))
                    events.extend(stream.finish())
                calls = assemble(events)[2]
                assert len(calls) == 2
                assert all(call["complete"] for call in calls)
                assert [json.loads(call["arguments"]) for call in calls] == [expected, expected]


def test_gemma_nested_argument_spans_preserve_scalar_types():
    import json

    expected = {"values": [True, None, -1.25, {"key": 'a"\\世界'}]}
    output = '<|tool_call>call:echo{values:[true,null,-1.25,{key:<|"|>a"\\世界<|"|>}]}<tool_call|>'
    tools = [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}]
    with Template((FIXTURES / "google-gemma-4-31B-it.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "Hello"}], tools=tools, now=946684800
        ) as plan:
            with plan.stream() as stream:
                events = []
                for byte in output.encode():
                    events.extend(stream.feed(bytes([byte])))
                events.extend(stream.finish())
            assert json.loads(assemble(events)[2][0]["arguments"]) == expected
