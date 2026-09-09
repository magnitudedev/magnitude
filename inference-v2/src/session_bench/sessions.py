"""Serializable serving requests and session schedules."""

import json
from typing import Literal, cast

from pydantic import Field, JsonValue, model_validator

from benchmark_fixtures.interactions import ExpectedCall
from benchmark_fixtures.records import Record, digest, encoded
from benchmark_fixtures.ruler import RetrievalAnswers

from .policy import MAX_OUTPUT_TOKENS, PROSE_OUTPUT_TOKENS, RETRIEVAL_OUTPUT_TOKENS

Section = Literal["single", "context", "session", "parallel", "fork", "concurrency", "memory"]


class Request(Record):
    id: str
    section: Section
    session: str
    checkpoint: int = Field(ge=0)
    concurrency: int = Field(default=1, ge=1)
    workload: Literal["tools", "prose", "retrieval"] = "tools"
    fixture_id: str
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    expected: list[ExpectedCall] | RetrievalAnswers
    fixture_provenance: dict[str, JsonValue] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    release_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def answer_contract(self):
        if (self.workload == "retrieval") != isinstance(self.expected, RetrievalAnswers):
            raise ValueError(
                "retrieval requests require RetrievalAnswers; other workloads require calls"
            )
        if self.workload == "retrieval" and self.tools:
            raise ValueError("retrieval requests do not use tools")
        return self

    @property
    def output_limit(self) -> int:
        if self.workload == "retrieval":
            return RETRIEVAL_OUTPUT_TOKENS
        return PROSE_OUTPUT_TOKENS if self.workload == "prose" else MAX_OUTPUT_TOKENS

    def body(self, model: str) -> dict:
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
    qualification: Request | None = None

    @property
    def warmup(self) -> Request:
        if self.qualification is not None:
            return self.qualification
        first = self.requests[0]
        if first.workload == "prose":
            return first.model_copy(
                update={
                    "id": "warmup",
                    "depends_on": (),
                    "messages": [
                        {"role": "system", "content": "Independent reading qualification."},
                        {
                            "role": "user",
                            "content": cast(str, first.messages[-1]["content"])[:1024],
                        },
                    ],
                    "fixture_provenance": {
                        "fixture": "prose.moby-dick",
                        "recipe": "prose-qualification-v1",
                        "corpus_digest": self.corpus_digest,
                    },
                }
            )
        return first.model_copy(
            update={
                "id": "warmup",
                "fixture_provenance": {
                    "fixture": "tools.bfcl",
                    "recipe": "qualification-decision-v1",
                    "decision": first.fixture_id,
                    "corpus_digest": self.corpus_digest,
                },
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
