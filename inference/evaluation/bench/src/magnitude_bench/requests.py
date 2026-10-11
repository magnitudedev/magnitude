"""Serializable serving requests and the plan of one run."""

import json
from typing import Literal

from magnitude_fixtures import Family, Generation
from magnitude_fixtures.canonical import Record, digest, encoded
from pydantic import Field, JsonValue

from .policy import SEED

Section = Literal["single", "context", "session", "parallel", "fork", "concurrency", "memory"]


class Request(Record):
    """One serving request: a sized fixture's conversation up to its generation prefix."""

    id: str
    section: Section
    session: str
    #: The context the fixture was sized to; 0 for an unsized (head and probe only) request.
    checkpoint: int = Field(ge=0)
    concurrency: int = Field(default=1, ge=1)
    family: Family
    fixture_id: str
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    generation: Generation
    depends_on: tuple[str, ...] = ()
    release_ms: int = Field(default=0, ge=0)

    @property
    def output_limit(self) -> int:
        return self.generation.max_tokens

    def body(self, model: str) -> dict:
        """Greedy, seeded, thinking disabled; tools are required when the fixture offers them."""
        return json.loads(
            encoded(
                {
                    "model": model,
                    "messages": self.messages,
                    **({"tools": self.tools, "tool_choice": "required"} if self.tools else {}),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": self.output_limit,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": SEED,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "n": 1,
                }
            )
        )


class Plan(Record):
    warmup: Request
    requests: tuple[Request, ...]
    parallel_sequences: int = Field(ge=1)
    # Sharing a fixture's history is not evidence that an engine retained a prefix.
    cache_policy: Literal["disabled"] = "disabled"

    @property
    def prepared_requests(self) -> tuple[Request, ...]:
        return (self.warmup, *self.requests)

    @property
    def identity(self) -> str:
        return digest(self.model_dump(mode="json"))
