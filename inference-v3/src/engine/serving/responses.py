"""Streaming wire records with explicit native service-attribution boundaries."""

import json
from time import time
from uuid import uuid4

from engine.serving.session import ChatFinished
from templates.events import ContentDelta, ReasoningDelta, ToolArguments, ToolStart


class ChatResponse:
    def __init__(self, model: str):
        self.identity = "chatcmpl-" + uuid4().hex
        self.model, self.created = model, int(time())
        self.content: list[str] = []
        self.reasoning: list[str] = []
        self.calls: list[dict] = []
        self.arguments: list[list[str]] = []

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

    def semantic(
        self, event: ContentDelta | ReasoningDelta | ToolStart | ToolArguments, *, retain: bool
    ) -> dict:
        if isinstance(event, (ContentDelta, ReasoningDelta)):
            field = "content" if isinstance(event, ContentDelta) else "reasoning_content"
            if retain:
                (self.content if isinstance(event, ContentDelta) else self.reasoning).append(
                    event.text
                )
            return self.chunk({field: event.text})
        if isinstance(event, ToolArguments):
            if retain:
                self.arguments[event.index].append(event.text)
            return self.chunk(
                {"tool_calls": [dict(index=event.index, function=dict(arguments=event.text))]}
            )
        call = dict(
            id=event.id,
            type="function",
            function=dict(name=event.name, arguments=""),
        )
        if retain:
            if event.index != len(self.calls):
                raise ValueError("tool-call indexes must be contiguous")
            self.calls.append(call)
            self.arguments.append([])
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
                preparation_ms=native.preparation_ns / 1e6,
                prefill_preparation_ms=native.prefill_preparation_ns / 1e6,
                decode_preparation_ms=native.decode_preparation_ns / 1e6,
                prompt_ms=native.prefill_ns / 1e6,
                predicted_ms=native.decode_ns / 1e6,
                first_decode_preparation_ms=native.first_decode_preparation_ns / 1e6,
                first_decode_ms=native.first_decode_ns / 1e6,
                draft_n=0,
                draft_n_accepted=0,
                speculative_backend="none",
            ),
            engine=dict(
                native=native.model_dump(mode="json"),
                preparation=event.preparation.model_dump(mode="json"),
                parsing=event.parsing.model_dump(mode="json"),
                string_stop=event.string_stop,
                timing_basis=(
                    "equal per-row attribution of preparation and post-preparation-through-"
                    "observed-completion intervals; preparation can overlap device execution; "
                    "sum both intervals for phase wall time; replay reported separately"
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
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": "".join(parts)}}
                for call, parts in zip(self.calls, self.arguments, strict=True)
            ]
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
