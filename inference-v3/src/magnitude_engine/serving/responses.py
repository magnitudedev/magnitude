"""Streaming wire records with explicit native service-attribution boundaries."""

import json
from time import time
from uuid import uuid4

from magnitude_engine.serving.parsing import TextDelta, ToolCall
from magnitude_engine.serving.session import ChatFinished


class ChatResponse:
    def __init__(self, model: str):
        self.identity = "chatcmpl-" + uuid4().hex
        self.model, self.created = model, int(time())
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []

    def envelope(self, choices: list[dict], *, complete=False) -> dict:
        return dict(
            id=self.identity,
            object="chat.completion" if complete else "chat.completion.chunk",
            created=self.created,
            model=self.model,
            choices=choices,
        )

    def chunk(self, delta: dict, finish_reason=None) -> dict:
        return self.envelope([dict(index=0, delta=delta, finish_reason=finish_reason)])

    def semantic(self, event: TextDelta | ToolCall, *, retain: bool) -> dict:
        if isinstance(event, TextDelta):
            field = "content" if event.channel == "content" else "reasoning_content"
            if retain:
                (self.content if event.channel == "content" else self.reasoning).append(event.text)
            return self.chunk({field: event.text})
        call = dict(
            id=f"call_{self.identity[9:]}_{event.index}",
            type="function",
            function=dict(
                name=event.name, arguments=json.dumps(event.arguments, ensure_ascii=False)
            ),
        )
        if retain:
            self.calls.append(call)
        return self.chunk({"tool_calls": [dict(index=event.index, **call)]})

    def evidence(self, event: ChatFinished) -> dict:
        native = event.native
        return dict(
            usage=dict(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=native.generated_tokens,
                total_tokens=event.prompt_tokens + native.generated_tokens,
                prompt_tokens_details=dict(cached_tokens=0),
            ),
            timings=dict(
                cache_n=0,
                prompt_n=event.prompt_tokens,
                predicted_n=native.generated_tokens,
                prompt_ms=native.prefill_ns / 1e6,
                predicted_ms=native.decode_ns / 1e6,
                draft_n=0,
                draft_n_accepted=0,
                speculative_backend="none",
            ),
            engine=dict(
                native=native.model_dump(mode="json"),
                string_stop=event.string_stop,
                timing_basis=(
                    "equal per-row attribution of submission-through-observed-completion "
                    "batch service; replay reported separately"
                ),
            ),
        )

    def terminal(self, event: ChatFinished) -> dict:
        return {**self.envelope([]), **self.evidence(event)}

    def complete(self, event: ChatFinished) -> dict:
        message: dict = dict(role="assistant", content="".join(self.content) or None)
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            message["tool_calls"] = self.calls
        return {
            **self.envelope(
                [dict(index=0, message=message, finish_reason=event.reason)], complete=True
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
