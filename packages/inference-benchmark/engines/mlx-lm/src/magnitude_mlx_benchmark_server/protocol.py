from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class RequestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ChatRequest:
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int
    temperature: float
    top_p: float
    seed: int
    enable_thinking: bool


_SUPPORTED_FIELDS = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_tokens",
    "temperature",
    "top_p",
    "seed",
    "stream",
    "stream_options",
    "chat_template_kwargs",
}


def parse_chat_request(body: Any, served_model: str) -> ChatRequest:
    if not isinstance(body, dict):
        raise RequestValidationError("request body must be an object")
    unsupported = sorted(set(body) - _SUPPORTED_FIELDS)
    if unsupported:
        raise RequestValidationError(f"unsupported fields: {', '.join(unsupported)}")
    if body.get("model") != served_model:
        raise RequestValidationError(f"model must be {served_model!r}")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestValidationError("messages must be a non-empty array")
    if not all(isinstance(message, dict) for message in messages):
        raise RequestValidationError("every message must be an object")
    tools = body.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise RequestValidationError("tools must be an array of objects")
    if body.get("stream") is not True:
        raise RequestValidationError("stream must be true")
    stream_options = body.get("stream_options")
    if stream_options != {"include_usage": True}:
        raise RequestValidationError("stream_options.include_usage must be true")
    if tools and body.get("tool_choice") != "required":
        raise RequestValidationError(
            "tool_choice must be required when tools are present"
        )
    if body.get("parallel_tool_calls") is not True:
        raise RequestValidationError("parallel_tool_calls must be true")
    max_tokens = body.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise RequestValidationError("max_tokens must be a positive integer")
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    seed = body.get("seed")
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature < 0
    ):
        raise RequestValidationError("temperature must be a non-negative number")
    if (
        not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not 0 < top_p <= 1
    ):
        raise RequestValidationError("top_p must be in (0, 1]")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RequestValidationError("seed must be an integer")
    kwargs = body.get("chat_template_kwargs")
    if not isinstance(kwargs, dict) or kwargs.get("enable_thinking") is not False:
        raise RequestValidationError(
            "chat_template_kwargs.enable_thinking must be false"
        )
    return ChatRequest(
        model=served_model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
        seed=seed,
        enable_thinking=False,
    )


def completion_chunk(
    request_id: str,
    model: str,
    created: int,
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def terminal_chunk(
    request_id: str,
    model: str,
    created: int,
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    prompt_ms: float,
    generation_ms: float,
    peak_memory_bytes: int,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
        "timings": {
            "cache_n": cached_tokens,
            "prompt_n": prompt_tokens - cached_tokens,
            "prompt_ms": prompt_ms,
            "predicted_n": completion_tokens,
            "predicted_ms": generation_ms,
        },
        "mlx": {"peak_memory_bytes": peak_memory_bytes},
    }


def sse(payload: dict[str, Any] | str) -> bytes:
    encoded = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, separators=(",", ":"))
    )
    return f"data: {encoded}\n\n".encode()
