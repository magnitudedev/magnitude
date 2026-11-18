"""Serializable serving requests and the plan of one run."""

import json
from typing import Literal

from magnitude_fixtures import Generation, Key
from magnitude_fixtures.canonical import Record, digest, encoded
from pydantic import Field, JsonValue

from .policy import SEED

Section = Literal["context", "session", "parallel", "fork", "concurrency", "memory", "vision"]
#: What a request's answer is checked as: a template key, or the bench's own image
#: capability check (``vision``), which is not a template.
Check = Key | Literal["vision"]


class Request(Record):
    """One serving request: a template lane's sized conversation up to its generation prefix,
    or one case of the bench's vision check."""

    id: str
    section: Section
    session: str
    #: The context the template sizes its lane to; 0 for a vision case, which is not sized.
    checkpoint: int = Field(ge=0)
    concurrency: int = Field(default=1, ge=1)
    #: The template's name; None for a vision case.
    template: str | None
    lane: int = Field(ge=0)
    key: Check
    #: The digest of the lane's rendered tokens, where the catalog GGUF's rendering sized it.
    prompt_sha256: str | None = None
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    generation: Generation
    depends_on: tuple[str, ...] = ()
    release_ms: int = Field(default=0, ge=0)

    @property
    def output_limit(self) -> int:
        return self.generation.max_tokens

    def body(self, model: str) -> dict:
        """Greedy, seeded, thinking disabled; tools are required when the session offers them."""
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
    # Sharing a session's history is not evidence that an engine retained a prefix.
    cache_policy: Literal["disabled"] = "disabled"

    @property
    def prepared_requests(self) -> tuple[Request, ...]:
        return (self.warmup, *self.requests)

    @property
    def identity(self) -> str:
        return digest(self.model_dump(mode="json"))
