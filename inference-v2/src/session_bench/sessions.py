"""Serializable interactions and shared simulated session requests."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .policy import MAX_OUTPUT_TOKENS


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def encoded(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class ExpectedCall(Record):
    name: str
    arguments: dict[str, list[JsonValue]]


class Interaction(Record):
    id: str
    category: str
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    expected: list[ExpectedCall]
    provenance: dict[str, str]

    def completed(self, identity: str) -> list[dict[str, JsonValue]]:
        calls = []
        replies = []
        for index, call in enumerate(self.expected):
            arguments = {key: values[0] for key, values in call.arguments.items() if values}
            call_id = f"call_{identity}_{index}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": encoded(arguments),
                    },
                }
            )
            replies.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": encoded(
                        {
                            "ok": True,
                            "tool": call.name,
                            "arguments": arguments,
                        }
                    ),
                }
            )
        return [*self.messages, {"role": "assistant", "content": "", "tool_calls": calls}, *replies]


Section = Literal["single", "context", "session", "parallel", "fork", "concurrency", "memory"]


class Request(Record):
    id: str
    section: Section
    session: str
    checkpoint: int = Field(ge=0)
    concurrency: int = Field(default=1, ge=1)
    fixture_id: str
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    expected: list[ExpectedCall]
    depends_on: tuple[str, ...] = ()
    release_ms: int = Field(default=0, ge=0)

    def body(self, model: str) -> dict:
        return json.loads(
            encoded(
                {
                    "model": model,
                    "messages": self.messages,
                    "tools": self.tools,
                    "tool_choice": "required",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 42,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "n": 1,
                }
            )
        )


class Plan(Record):
    requests: tuple[Request, ...]
    parallel_sequences: int = Field(ge=1)
    corpus_digest: str
    # Sharing canonical history is not evidence that an engine retained a prefix.
    cache_policy: Literal["disabled"] = "disabled"

    @property
    def warmup(self) -> Request:
        first = self.requests[0]
        return first.model_copy(
            update={
                "id": "warmup",
                "depends_on": (),
                "messages": [
                    {"role": "system", "content": "Qualification request, independent history."},
                    *first.messages[-1:],
                ],
            }
        )

    @property
    def prepared_requests(self) -> tuple[Request, ...]:
        return (self.warmup, *self.requests)

    @property
    def identity(self) -> str:
        return digest(
            {**self.model_dump(mode="json"), "warmup": self.warmup.model_dump(mode="json")}
        )
