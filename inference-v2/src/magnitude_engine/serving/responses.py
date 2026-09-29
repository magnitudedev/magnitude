"""OpenAI wire records derived from semantic output and native completion evidence."""

import json
from dataclasses import asdict
from time import time
from uuid import uuid4

from .parsing import TextDelta, ToolCall
from .session import ChatFinished


class ChatResponse:
    def __init__(self, model: str, speculative_backend: str | None):
        self.identity = "chatcmpl-" + uuid4().hex
        self.model, self.created = model, int(time())
        self.speculative_backend = speculative_backend
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []

    def envelope(self, choices: list[dict], *, complete: bool = False) -> dict:
        return {
            "id": self.identity,
            "object": "chat.completion" if complete else "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }

    def chunk(self, delta: dict, finish_reason: str | None = None) -> dict:
        return self.envelope([{"index": 0, "delta": delta, "finish_reason": finish_reason}])

    def semantic(self, event: TextDelta | ToolCall, *, retain: bool) -> dict:
        if isinstance(event, TextDelta):
            field = "content" if event.channel == "content" else "reasoning_content"
            if retain:
                (self.content if event.channel == "content" else self.reasoning).append(event.text)
            return self.chunk({field: event.text})
        call = {
            "id": f"call_{self.identity[9:]}_{event.index}",
            "type": "function",
            "function": {
                "name": event.name,
                "arguments": json.dumps(event.arguments, ensure_ascii=False),
            },
        }
        if retain:
            self.calls.append(call)
        return self.chunk({"tool_calls": [{"index": event.index, **call}]})

    def evidence(self, event: ChatFinished) -> dict:
        finish = event.native
        timings = {
            "cache_n": finish.cached_tokens,
            "prompt_n": finish.prompt_tokens - finish.cached_tokens,
            "prompt_ms": (finish.prefill_ns + finish.first_decode_ns) / 1e6,
            "predicted_n": finish.generated_tokens,
            "predicted_ms": (finish.decode_ns - finish.first_decode_ns) / 1e6,
        }
        if self.speculative_backend is not None:
            timings.update(
                draft_n=finish.proposed_tokens,
                draft_n_accepted=finish.accepted_tokens,
                speculative_backend=self.speculative_backend,
            )
        prompt_details = {"cached_tokens": finish.cached_tokens}
        if finish.media_tokens:
            prompt_details.update(
                media_tokens=finish.media_tokens,
                text_tokens=finish.prompt_tokens - finish.media_tokens,
            )
            timings.update(
                preparation_ms=finish.preparation_ns / 1e6,
                cached_input_features=finish.cached_input_features,
            )
        return {
            "usage": {
                "prompt_tokens": finish.prompt_tokens,
                "completion_tokens": finish.generated_tokens,
                "total_tokens": finish.prompt_tokens + finish.generated_tokens,
                "prompt_tokens_details": prompt_details,
            },
            "timings": timings,
            "engine": {
                "native": asdict(finish),
                "string_stop": event.string_stop,
                "timing_basis": (
                    "active service; prompt through first emission, generation thereafter"
                ),
            },
        }

    def terminal(self, event: ChatFinished) -> dict:
        return {**self.envelope([]), **self.evidence(event)}

    def complete(self, event: ChatFinished) -> dict:
        message: dict = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = self.calls
        return {
            **self.envelope(
                [{"index": 0, "message": message, "finish_reason": event.reason}], complete=True
            ),
            **self.evidence(event),
        }


def sse(payload: dict | str) -> bytes:
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )
    return f"data: {text}\n\n".encode()
