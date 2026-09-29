import asyncio
import json

import httpx
import pytest

from benchmark_fixtures.interactions import ExpectedCall
from session_bench.client import measure
from session_bench.suites import compile_plan
from session_bench.validation import terminal, tool_calls


async def count_context(context):
    return len(json.dumps(context.model_dump(mode="json")))


def terminal_event():
    return {
        "id": "response-1",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "timings": {
            "prompt_n": 10,
            "cache_n": 0,
            "prompt_ms": 20,
            "predicted_n": 4,
            "predicted_ms": 40,
        },
    }


def wire(finish="tool_calls"):
    return [
        {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "ec", "arguments": '{"val'},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "response-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "ho", "arguments": 'ue":7}'}}
                        ]
                    },
                }
            ],
        },
        {"id": "response-1", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
        terminal_event(),
        "[DONE]",
    ]


class Fragments(httpx.AsyncByteStream):
    def __init__(self, events):
        self.content = b"".join(
            ("data: " + (e if isinstance(e, str) else json.dumps(e)) + "\r\n\r\n").encode()
            for e in events
        )

    async def __aiter__(self):
        for i in range(0, len(self.content), 7):
            yield self.content[i : i + 7]


async def observed(interaction, events):
    async def handle(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 32768
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Fragments(events)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        request = (await compile_plan(
            [interaction], "c", ("single",), (1024,),
            counter=count_context, sizing_identity="test-bytes",
        )).requests[0]
        return await measure(client, "http://engine", "test", request, lambda _: None)


async def test_fragmented_stream_and_native_evidence(interaction):
    result = await observed(interaction, wire())
    assert result.outcome == "valid"
    assert result.tool_calls == [{"id": "call-1", "name": "echo", "arguments": '{"value":7}'}]
    assert result.terminal["timings"]["prompt_ms"] == 20
    assert result.ttft_ms is not None


async def test_parseable_truncation_is_never_valid(interaction):
    assert (await observed(interaction, wire("length"))).outcome == "truncated"


@pytest.mark.parametrize(
    "mutation", ["no_done", "no_usage", "bad_counts", "duplicate_usage", "after_done"]
)
async def test_incomplete_or_inconsistent_evidence(interaction, mutation):
    events = wire()
    if mutation == "no_done":
        events.pop()
    elif mutation == "no_usage":
        events.pop(-2)
    elif mutation == "bad_counts":
        events[-2]["usage"]["total_tokens"] = 999
    elif mutation == "duplicate_usage":
        events.insert(-1, terminal_event())
    else:
        events.append(terminal_event())
    assert (await observed(interaction, events)).outcome == "protocol-error"


def test_non_greedy_assignment():
    expected = [
        ExpectedCall(name="echo", arguments={"value": [1, 2]}),
        ExpectedCall(name="echo", arguments={"value": [1]}),
    ]
    calls = [
        {"name": "echo", "arguments": '{"value":1}'},
        {"name": "echo", "arguments": '{"value":2}'},
    ]
    assert tool_calls(expected, calls) is None
    calls[1]["arguments"] = '{"value":3}'
    assert tool_calls(expected, calls) is not None


@pytest.mark.parametrize("bad", [True, -1, float("nan"), 1.5])
def test_invalid_counts_rejected(bad):
    value = terminal_event()
    value["usage"]["prompt_tokens"] = bad
    with pytest.raises(ValueError):
        terminal(value)


async def test_cancel_records_partial_output(interaction):
    ready = asyncio.Event()

    class Hanging(Fragments):
        async def __aiter__(self):
            yield self.content
            ready.set()
            await asyncio.Event().wait()

    async def handle(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Hanging(wire()[:1])
        )

    saved = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        request = (await compile_plan(
            [interaction], "c", ("single",), (1024,),
            counter=count_context, sizing_identity="test-bytes",
        )).requests[0]
        task = asyncio.create_task(
            measure(
                client, "http://engine", "test", request, lambda _: None, cancelled=saved.append
            )
        )
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert saved[0].outcome == "cancelled"
    assert saved[0].tool_calls[0]["name"] == "ec"


def test_numeric_equivalence_does_not_accept_booleans():
    expected = [ExpectedCall(name="echo", arguments={"value": [{"nested": [1]}]})]
    assert (
        tool_calls(expected, [{"name": "echo", "arguments": '{"value":{"nested":[1.0]}}'}]) is None
    )
    assert tool_calls(expected, [{"name": "echo", "arguments": '{"value":{"nested":[true]}}'}])


async def test_validation_time_is_not_measured_as_completion_latency(interaction, monkeypatch):
    from session_bench import client as implementation

    clock = [10.0]
    monkeypatch.setattr(implementation.time, "perf_counter", lambda: clock[0])
    original = implementation.validation.tool_calls

    def slow_validation(*args):
        clock[0] += 100
        return original(*args)

    monkeypatch.setattr(implementation.validation, "tool_calls", slow_validation)
    result = await observed(interaction, wire())
    assert result.outcome == "valid"
    assert result.completed_ms == 0
