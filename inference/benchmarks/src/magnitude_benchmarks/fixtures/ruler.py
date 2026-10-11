"""RULER-derived retrieval fixtures with stable facts across independently sized contexts.

Adapted from NVIDIA/RULER's NIAH record-haystack and multi-query tasks:
https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
The local recipe uses independent indexed records, explicit depth and strict JSON
answers; its scores are not official RULER scores. No upstream runtime is required.
"""

import json
import math
from typing import Literal, Self

from pydantic import Field, model_validator

from .contexts import Context, Counter, PreparedContext
from .records import Record, digest

SOURCE = {
    "repository": "https://github.com/NVIDIA/RULER",
    "commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
    "path": "scripts/data/synthetic/niah.py",
    "recipe": "ruler-record-retrieval-v1",
}
MAX_RECORDS = 65_536
INSTRUCTION = (
    "Look up the requested keys in the records. Return only a JSON object mapping each "
    "requested key to its exact string value. Include every requested key, no other keys, "
    "no explanation and no markdown."
)


class RetrievalScore(Record):
    exact_match: bool
    correct: int = Field(ge=0)
    total: int = Field(gt=0)
    format_valid: bool


class RetrievalAnswers(Record):
    values: dict[str, str] = Field(min_length=1)

    def score(self, response_text: str) -> RetrievalScore:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate answer key")
                result[key] = value
            return result

        try:
            actual = json.loads(response_text, object_pairs_hook=unique_object)
            valid = isinstance(actual, dict) and all(type(v) is str for v in actual.values())
        except (ValueError, RecursionError):
            actual, valid = {}, False
        correct = sum(actual.get(k) == v for k, v in self.values.items()) if valid else 0
        return RetrievalScore(
            exact_match=valid and actual == self.values,
            correct=correct,
            total=len(self.values),
            format_valid=valid,
        )


class PreparedRetrieval(PreparedContext):
    expected: RetrievalAnswers

    def score(self, response_text: str) -> RetrievalScore:
        return self.expected.score(response_text)


class RulerFixture(Record):
    seed: int = Field(default=42, ge=0, strict=True)
    variant: Literal["single", "multiquery"] = "single"
    haystack: Literal["records"] = "records"
    queries: int = Field(default=1, ge=1, le=16, strict=True)

    @model_validator(mode="after")
    def query_count(self) -> Self:
        if self.variant == "single" and self.queries != 1:
            raise ValueError("single retrieval requires exactly one query")
        return self

    @property
    def identity(self) -> str:
        return digest({"source": SOURCE, "configuration": self.model_dump(mode="json")})

    def _record(self, index: int) -> tuple[str, str]:
        # Indexed facts do not depend on token-count search, depth or prior prepare calls.
        fingerprint = digest({"seed": self.seed, "record": index})
        return f"{fingerprint[:8]}-{index:08x}", fingerprint[8:24]

    @property
    def expected(self) -> RetrievalAnswers:
        return RetrievalAnswers(values=dict(self._record(i) for i in range(self.queries)))

    async def prepare(
        self,
        target: int,
        counter: Counter,
        sizing_identity: str,
        *,
        needle_depth: float = 0.5,
    ) -> PreparedRetrieval:
        """Reach an approximate input target at complete-record boundaries.

        Depth is the fraction of distractor records before the target facts. The
        facts form one block; record positions and rendered prefix counts are recorded.
        This constructs independent snapshots, not an append-only cached session.
        """
        if type(target) is not int or target < 0:
            raise ValueError("context target must be a nonnegative integer")
        if not math.isfinite(needle_depth) or not 0 <= needle_depth <= 1:
            raise ValueError("needle_depth must be between zero and one")
        if not sizing_identity:
            raise ValueError("sizing_identity must identify the tokenizer and renderer")
        expected = self.expected
        needles = [f"{key}: {value}" for key, value in expected.values.items()]
        distractors: list[str] = []
        query = "Return the values for these keys: " + ", ".join(expected.values)

        def context(lines: list[str]) -> Context:
            return Context(
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": "Records:\n" + "\n".join(lines) + "\n\n" + query},
                ]
            )

        async def count(content: Context) -> int:
            size = await counter(content)
            if type(size) is not int or size < 1:
                raise ValueError("context renderer returned an invalid token count")
            return size

        async def evaluate(records: int):
            while len(distractors) < records:
                key, value = self._record(self.queries + len(distractors))
                distractors.append(f"{key}: {value}")
            before = round(records * needle_depth)
            lines = distractors[:before] + needles + distractors[before:records]
            content = context(lines)
            return content, await count(content), lines, before

        prepared, size, lines, before = await evaluate(0)
        low, high = 0, 0
        while size < target:
            if high == MAX_RECORDS:
                raise ValueError("context target exceeds the RULER fixture record limit")
            low, high = high, min(MAX_RECORDS, max(1, high * 2))
            previous = size
            prepared, size, lines, before = await evaluate(high)
            if size <= previous:
                raise ValueError("rendered context did not grow after adding records")
        while high - low > 1:
            middle = (low + high) // 2
            candidate = await evaluate(middle)
            if candidate[1] >= target:
                high = middle
                prepared, size, lines, before = candidate
            else:
                low = middle

        # A Counter counts complete rendered Contexts, not raw text spans. Preserve
        # that distinction: these are prefix render counts, not asserted token offsets
        # within the final prompt (templates can add suffixes or merge boundary tokens).
        prefix_counts = {}
        for index, key in enumerate(expected.values):
            prefix = Context(
                messages=prepared.messages[:1]
                + [{"role": "user", "content": "Records:\n" + "\n".join(lines[: before + index])}]
            )
            prefix_counts[key] = await count(prefix)

        return PreparedRetrieval(
            content=prepared,
            tokens=size,
            expected=expected,
            provenance={
                **SOURCE,
                "fixture": "retrieval.ruler",
                "fixture_digest": self.identity,
                "configuration": self.model_dump(mode="json"),
                "requested_context_tokens": target,
                "actual_context_tokens": size,
                "needle_depth": needle_depth,
                "distractor_records": high,
                "needle_record_positions": {
                    key: before + index for index, key in enumerate(expected.values)
                },
                "needle_prefix_render_tokens": prefix_counts,
                "sizing_identity": sizing_identity,
                "content_digest": digest(prepared.model_dump(mode="json")),
            },
        )
