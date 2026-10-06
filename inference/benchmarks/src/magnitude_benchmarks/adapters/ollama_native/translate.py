"""Pure mapping between the benchmark's OpenAI-shaped wire protocol and Ollama's native API."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

NANOSECONDS_PER_MILLISECOND = 1_000_000


def prompt_identity(body: dict[str, Any]) -> str:
    """Identity of a request's prompt: what the engine renders, not how it samples."""
    prompt = {"messages": body["messages"], "tools": body.get("tools") or []}
    encoded = json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def chat_request(body: dict[str, Any], model: str, context: int) -> dict[str, Any]:
    """The native ``/api/chat`` request for one benchmark request.

    The context is always sent: a request without ``num_ctx`` would let Ollama reload the model
    at its own default.
    """
    options: dict[str, Any] = {"num_ctx": context}
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("seed", "seed"),
        ("max_tokens", "num_predict"),
    ):
        if source in body:
            options[target] = body[source]
    request: dict[str, Any] = {
        "model": model,
        "messages": body["messages"],
        "stream": True,
        "keep_alive": -1,
        "options": options,
    }
    if body.get("tools"):
        request["tools"] = body["tools"]
    thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
    if thinking is not None:
        request["think"] = bool(thinking)
    return request


# Muse Glimmer's format, as Ollama's ``glimmer`` renderer writes it (model/renderers/glimmer.go).
GLIMMER_FAMILY = "muse-glimmer"
GLIMMER_ANSWER_HEADER = " to=user<|message|>"
GLIMMER_ROLES = {"user": "user", "assistant": "assistant to=user"}


def glimmer_prompt(messages: list[dict[str, Any]]) -> str:
    """The prompt Ollama's ``glimmer`` renderer produces for these messages with ``think`` false.

    Without the leading ``<|begin_of_text|>``: the raw completion route leaves that token to the
    runner. Only what the benchmark sends is reproduced (a system message, then text turns);
    anything the renderer would treat differently is refused rather than approximated.
    """
    if not messages or messages[0]["role"] != "system":
        raise ValueError("answer prefill needs a leading system message")
    parts = []
    for index, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or message.get("tool_calls") or message.get("thinking"):
            raise ValueError("answer prefill supports text messages only")
        if role == "system":
            # The renderer rewrites or keeps a strength the system message states itself.
            lowered = content.lower()
            stated = "reasoning strength" in lowered or "reasoning effort" in lowered
            if index or stated:
                raise ValueError("answer prefill supports one plain leading system message")
            parts.append(
                "<|start|>system<|message|>" + content + "\n\nReasoning strength: none."
                '\n\n# Valid recipients: "self", "user".<|eot|>'
            )
        elif role in GLIMMER_ROLES:
            parts.append(f"<|start|>{GLIMMER_ROLES[role]}<|message|>{content}<|eot|>")
        else:
            raise ValueError(f"answer prefill does not support the {role} role")
    return "".join(parts) + "<|start|>assistant"


def prefilled_request(chat: dict[str, Any]) -> dict[str, Any]:
    """The raw ``/api/generate`` request that continues ``chat`` from the answer header.

    Same model, options and keep-alive as the chat request; Ollama applies no template and no
    parser in raw mode, so the reply is the model's text after the header.
    """
    if chat.get("tools"):
        raise ValueError("answer prefill does not support tools")
    return {
        "model": chat["model"],
        "prompt": glimmer_prompt(chat["messages"]) + GLIMMER_ANSWER_HEADER,
        "raw": True,
        "stream": True,
        # An oversized prompt is rejected, as on the chat route, instead of being cut.
        "truncate": False,
        "keep_alive": chat["keep_alive"],
        "options": chat["options"],
        **({"logprobs": chat["logprobs"]} if "logprobs" in chat else {}),
    }


SPECULATION_STATS = re.compile(rb'msg="speculative decode stats".*?drafted=(\d+) accepted=(\d+)')
LLAMA_DRAFT_STATS = re.compile(
    rb"draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)"
)
LLAMA_LAUNCH = re.compile(rb'msg="starting llama-server" cmd="([^"]*)"')


def speculation_stats(line: bytes) -> tuple[int, int] | None:
    """Drafted and accepted tokens from a runner's per-request speculation log line.

    The MLX runner logs them itself; the llama.cpp runner logs them with its slot timings.
    """
    match = SPECULATION_STATS.search(line)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = LLAMA_DRAFT_STATS.search(line)
    return (int(match.group(2)), int(match.group(1))) if match else None


OFFLOADED_LAYERS = re.compile(rb"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
MLX_DEVICE = re.compile(rb'msg="MLX engine initialized".*?device=(\w+)')


def offloaded_layers(line: bytes) -> tuple[int, int] | None:
    """Layers the llama.cpp runner placed on the GPU, and the model's layers, from its load log.

    One line per model it loads: the target, then a separate draft model when the tag has one.
    """
    match = OFFLOADED_LAYERS.search(line)
    return (int(match.group(1)), int(match.group(2))) if match else None


def mlx_device(line: bytes) -> str | None:
    """The device the MLX runner reports when it starts."""
    match = MLX_DEVICE.search(line)
    return match.group(1).decode() if match else None


TRUNCATION = re.compile(rb'msg="truncating input prompt" limit=\d+ prompt=(\d+)')


def truncated_prompt_tokens(line: bytes) -> int | None:
    """Full size of a prompt Ollama cut to fit the context, from the warning it logs.

    Ollama renders and tokenises a registry model's prompt itself and truncates one that does
    not fit before its runner sees it, so the runner never rejects it as too large.
    """
    match = TRUNCATION.search(line)
    return int(match.group(1)) if match else None


def llama_launch(line: bytes) -> str | None:
    """The llama.cpp runner's command line, from the log line Ollama writes when it starts it."""
    match = LLAMA_LAUNCH.search(line)
    return match.group(1).decode(errors="replace") if match else None


def terminal(
    final: dict[str, Any],
    context: int | None = None,
    drafted: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Usage and native timings from Ollama's final streamed chunk.

    ``prompt_eval_count`` is the whole prompt; ``prompt_eval_cached_count`` of those were reused,
    and ``prompt_eval_duration`` covers only the tokens that were evaluated.
    """
    try:
        prompt = final["prompt_eval_count"]
        cached = final.get("prompt_eval_cached_count", 0)
        predicted = final["eval_count"]
        prompt_ns = final["prompt_eval_duration"]
        predicted_ns = final["eval_duration"]
    except KeyError as error:
        raise ValueError(f"Ollama final chunk has no {error.args[0]}") from error
    counts = (prompt, cached, predicted)
    if any(type(n) is not int or n < 0 for n in counts) or cached > prompt:
        raise ValueError(f"Ollama reported invalid token counts: {counts}")
    durations = (prompt_ns, predicted_ns)
    if any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in durations):
        raise ValueError(f"Ollama reported invalid durations: {durations}")
    timings: dict[str, Any] = {
        "cache_n": cached,
        "prompt_n": prompt - cached,
        "prompt_ms": prompt_ns / NANOSECONDS_PER_MILLISECOND,
        "predicted_n": predicted,
        "predicted_ms": predicted_ns / NANOSECONDS_PER_MILLISECOND,
    }
    if context is not None:
        # The context Ollama reports for the loaded model; its MLX runner does not enforce it.
        timings["n_ctx"] = context
    if drafted is not None:
        timings["draft_n"], timings["draft_n_accepted"] = drafted
    if "load_duration" in final:
        timings["load_ms"] = final["load_duration"] / NANOSECONDS_PER_MILLISECOND
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": predicted,
            "total_tokens": prompt + predicted,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "timings": timings,
    }


def finish_reason(final: dict[str, Any], tool_calls: bool) -> str:
    if tool_calls:
        return "tool_calls"
    reason = final.get("done_reason")
    if reason not in ("stop", "length"):
        raise ValueError(f"unexpected Ollama done_reason: {reason}")
    return reason


def delta(chunk: dict[str, Any], tool_index: int) -> dict[str, Any]:
    """One streamed message chunk as an OpenAI delta. ``tool_index`` numbers its tool calls."""
    message = chunk.get("message") or {}
    value: dict[str, Any] = {}
    if chunk.get("response"):
        # A raw completion chunk: the text is the reply itself.
        value["content"] = chunk["response"]
    if message.get("content"):
        value["content"] = message["content"]
    if message.get("thinking"):
        value["reasoning_content"] = message["thinking"]
    calls = []
    for offset, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments")
        calls.append(
            {
                "index": tool_index + offset,
                "id": call.get("id") or f"call_{tool_index + offset}",
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": (
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments or {}, separators=(",", ":"))
                    ),
                },
            }
        )
    if calls:
        value["tool_calls"] = calls
    return value


def overflow_prompt_tokens(status: int, text: str) -> int | None:
    """Prompt size from the llama.cpp runner's context-overflow rejection, if this is one.

    Ollama relays the runner's error as a JSON string inside its own error object, so the
    field may arrive with escaped quotes.
    """
    if status != 400 or "exceed_context_size_error" not in text:
        return None
    match = re.search(r'n_prompt_tokens\\?":\s*(\d+)', text)
    return int(match.group(1)) if match else None
