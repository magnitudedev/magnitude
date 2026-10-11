import asyncio
import json

import httpx
import pytest
from bench_support import tools_fixture
from magnitude_bench.client import measure
from magnitude_bench.plan import request as serving_request
from magnitude_bench.validation import terminal


def single(fixture=None):
    fixture = fixture or tools_fixture(0)
    return serving_request(fixture, fixture.conversation_with(0), "single", "single", 0)


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


async def observed(events):
    async def handle(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 32768
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Fragments(events)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        return await measure(client, "http://engine", "test", single(), lambda _: None)


async def test_fragmented_stream_and_native_evidence():
    result = await observed(wire())
    assert result.outcome == "valid"
    assert result.tool_calls == [{"id": "call-1", "name": "echo", "arguments": '{"value":7}'}]
    assert result.terminal["timings"]["prompt_ms"] == 20
    assert result.ttft_ms is not None


async def test_parseable_truncation_is_never_valid():
    assert (await observed(wire("length"))).outcome == "truncated"


@pytest.mark.parametrize(
    "mutation", ["no_done", "no_usage", "bad_counts", "duplicate_usage", "after_done"]
)
async def test_incomplete_or_inconsistent_evidence(mutation):
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
    assert (await observed(events)).outcome == "protocol-error"


@pytest.mark.parametrize("bad", [True, -1, float("nan"), 1.5])
def test_invalid_counts_rejected(bad):
    value = terminal_event()
    value["usage"]["prompt_tokens"] = bad
    with pytest.raises(ValueError):
        terminal(value)


async def test_cancel_records_partial_output():
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
        task = asyncio.create_task(
            measure(
                client, "http://engine", "test", single(), lambda _: None, cancelled=saved.append
            )
        )
        await ready.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert saved[0].outcome == "cancelled"
    assert saved[0].tool_calls[0]["name"] == "ec"


async def test_validation_time_is_not_measured_as_completion_latency(monkeypatch):
    from magnitude_bench import client as implementation

    clock = [10.0]
    monkeypatch.setattr(implementation.time, "perf_counter", lambda: clock[0])
    original = implementation.answers.tool_calls

    def slow_validation(*args):
        clock[0] += 100
        return original(*args)

    monkeypatch.setattr(implementation.answers, "tool_calls", slow_validation)
    result = await observed(wire())
    assert result.outcome == "valid"
    assert result.completed_ms == 0


async def test_wrong_tool_call_is_invalid_and_reasoning_is_refused():
    events = wire()
    events[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] = 'ue":8}'
    result = await observed(events)
    assert result.outcome == "invalid" and "distinct matches" in result.error
    events = wire()
    events[0]["choices"][0]["delta"]["reasoning_content"] = "thinking"
    assert (await observed(events)).outcome == "invalid"


async def test_request_is_greedy_seeded_and_thinking_free():
    body = single().body("served")
    assert body["temperature"] == 0 and body["top_p"] == 1 and body["seed"] == 42
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["tool_choice"] == "required" and body["max_tokens"] == 32768
    assert body["messages"][-1] == {"role": "user", "content": "Call echo with value 7."}
    assert all("scored" not in message and "tools" not in message for message in body["messages"])
