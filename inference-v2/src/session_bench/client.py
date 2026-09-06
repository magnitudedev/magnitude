"""Shared streaming wire client. Client times never stand in for native engine timings."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Literal

import httpx

from . import validation
from .policy import REQUEST_TIMEOUT_SECONDS
from .sessions import Record, Request

Outcome = Literal[
    "valid",
    "invalid",
    "truncated",
    "rejected",
    "timeout",
    "cancelled",
    "protocol-error",
    "transport-error",
    "target-failure",
    "dependency-failed",
]


class Observation(Record):
    request_id: str
    outcome: Outcome
    headers_ms: float | None = None
    ttft_ms: float | None = None
    completed_ms: float | None = None
    output_text: str = ""
    tool_calls: list[dict] = []
    finish_reason: str | None = None
    terminal: dict | None = None
    error: str | None = None


async def measure(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    request: Request,
    record_event: Callable[[dict], None],
    *,
    extensions: dict | None = None,
    cancelled: Callable[[Observation], None] | None = None,
) -> Observation:
    started = time.perf_counter()
    headers = ttft = completed = None
    output = ""
    calls = {}
    finish = evidence = None
    outcome: Outcome = "protocol-error"
    error = None
    body = request.body(model)
    for key, value in (extensions or {}).items():
        if key in body:
            raise ValueError(f"adapter cannot override shared request field {key}")
        body[key] = value

    def elapsed():
        return (time.perf_counter() - started) * 1000

    def observation():
        return Observation(
            request_id=request.id,
            outcome=outcome,
            headers_ms=headers,
            ttft_ms=ttft,
            completed_ms=completed if completed is not None else elapsed(),
            output_text=output,
            tool_calls=[calls[i] for i in sorted(calls)],
            finish_reason=finish,
            terminal=evidence,
            error=error,
        )

    done = False
    stream_id = None

    async def events(response):
        data = []
        async for line in response.aiter_lines():
            if line == "":
                if data:
                    yield "\n".join(data)
                    data.clear()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            yield "\n".join(data)

    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            async with client.stream(
                "POST", endpoint + "/v1/chat/completions", json=body
            ) as response:
                headers = elapsed()
                if response.status_code != 200:
                    response_body = (await response.aread()).decode(errors="replace")
                    outcome = (
                        "rejected"
                        if response.status_code in (400, 413, 422, 429)
                        else "target-failure"
                    )
                    error = f"HTTP {response.status_code}: {response_body[:4000]}"
                    return observation()
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    raise ValueError("response is not an SSE stream")
                async for raw in events(response):
                    record_event({"at_ms": elapsed(), "data": raw})
                    if done:
                        raise ValueError("data after [DONE]")
                    if raw == "[DONE]":
                        done = True
                        completed = elapsed()
                        continue
                    payload = json.loads(raw)
                    if not isinstance(payload, dict) or "error" in payload:
                        raise ValueError(f"error or non-object stream event: {payload}")
                    identity = payload.get("id")
                    if not isinstance(identity, str) or not identity:
                        raise ValueError("stream event missing ID")
                    if stream_id is not None and identity != stream_id:
                        raise ValueError("stream ID changed")
                    stream_id = identity
                    if payload.get("usage") is not None:
                        if evidence is not None or finish is None:
                            raise ValueError("duplicate or premature terminal usage")
                        evidence = validation.terminal(payload)
                        continue
                    if evidence is not None:
                        raise ValueError("output after terminal usage")
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or len(choices) != 1:
                        raise ValueError("expected exactly one streaming choice")
                    choice = choices[0]
                    if choice.get("index") != 0 or finish is not None:
                        raise ValueError("unexpected choice index or data after finish")
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        raise ValueError("missing delta")
                    content = delta.get("content") or ""
                    if not isinstance(content, str):
                        raise ValueError("content delta is not text")
                    output += content
                    semantic = bool(content)
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index")
                        if type(index) is not int or index < 0:
                            raise ValueError("invalid tool delta index")
                        current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if call.get("id"):
                            if current["id"] and current["id"] != call["id"]:
                                raise ValueError("tool call ID changed")
                            current["id"] = call["id"]
                        function = call.get("function") or {}
                        for field in ("name", "arguments"):
                            part = function.get(field) or ""
                            if not isinstance(part, str):
                                raise ValueError("tool delta must be text")
                            current[field] += part
                            semantic |= bool(part)
                    if semantic and ttft is None:
                        ttft = elapsed()
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                if not done or evidence is None or finish is None:
                    raise ValueError("stream ended without finish, consistent usage and [DONE]")
                if evidence["usage"]["completion_tokens"] > request.output_limit:
                    raise ValueError("engine exceeded the shared output allowance")
                if request.workload == "prose":
                    if calls or not output.strip() or evidence["usage"]["completion_tokens"] < 1:
                        raise ValueError("prose response must contain text and no tool calls")
                    if finish == "length":
                        if evidence["usage"]["completion_tokens"] != request.output_limit:
                            outcome, error = (
                                "truncated",
                                "context ended before the prose output budget",
                            )
                        else:
                            outcome = "valid"
                    elif finish == "stop":
                        outcome = "valid"
                    else:
                        raise ValueError(f"unexpected prose finish reason: {finish}")
                elif finish == "length":
                    outcome, error = "truncated", "engine reached its output or context limit"
                else:
                    if finish not in ("stop", "tool_calls"):
                        raise ValueError(f"unexpected finish reason: {finish}")
                    if any(not call["id"] or not call["name"] for call in calls.values()):
                        raise ValueError("incomplete tool call identity")
                    if len({call["id"] for call in calls.values()}) != len(calls):
                        raise ValueError("duplicate tool call ID")
                    error = validation.tool_calls(
                        request.expected, [calls[i] for i in sorted(calls)]
                    )
                    outcome = "invalid" if error else "valid"
    except asyncio.CancelledError:
        outcome, error = "cancelled", "request cancelled"
        if cancelled:
            cancelled(observation())
        raise
    except (TimeoutError, httpx.TimeoutException) as exc:
        outcome, error = "timeout", str(exc) or "request deadline exceeded"
    except httpx.HTTPError as exc:
        outcome, error = "transport-error", str(exc)
    except (ValueError, KeyError, TypeError) as exc:
        outcome, error = "protocol-error", str(exc)
    return observation()
