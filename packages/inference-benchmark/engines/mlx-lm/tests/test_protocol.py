import pytest

from magnitude_mlx_benchmark_server.protocol import (
    RequestValidationError,
    parse_chat_request,
)


def valid_request():
    return {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "parallel_tool_calls": True,
        "max_tokens": 32,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_accepts_benchmark_subset():
    parsed = parse_chat_request(valid_request(), "model")
    assert parsed.max_tokens == 32
    assert parsed.enable_thinking is False


def test_rejects_unsupported_fields():
    body = valid_request()
    body["n"] = 2
    with pytest.raises(RequestValidationError, match="unsupported fields"):
        parse_chat_request(body, "model")
