import asyncio
import json

import pytest

from benchmark_fixtures.records import digest
from benchmark_fixtures.ruler import PreparedRetrieval, RetrievalAnswers, RulerFixture


async def count(context):
    # A deterministic renderer substitute; this suite does not measure model token counts.
    return len(json.dumps(context.model_dump(mode="json")))


def records_and_query(prepared):
    records, query = prepared.content.messages[1]["content"].split("\n\n")
    return records.splitlines()[1:], query


async def test_resize_is_paired_reentrant_and_preserves_complete_records():
    fixture = RulerFixture(seed=42, variant="multiquery", queries=4)
    first, large, small = await asyncio.gather(
        *(fixture.prepare(n, count, "test-json-bytes") for n in (4096, 16384, 2048))
    )
    repeated = await fixture.prepare(4096, count, "test-json-bytes")
    assert first == repeated
    assert first.expected == large.expected == small.expected
    assert records_and_query(first)[1] == records_and_query(large)[1] == records_and_query(small)[1]
    assert small.tokens < first.tokens < large.tokens
    for prepared, target in ((small, 2048), (first, 4096), (large, 16384)):
        assert target <= prepared.tokens < target + 100
        records = dict(line.split(": ") for line in records_and_query(prepared)[0])
        assert len(records) == prepared.provenance["distractor_records"] + fixture.queries
        assert all(records[key] == value for key, value in prepared.expected.values.items())
        assert prepared.provenance["content_digest"] == digest(
            prepared.content.model_dump(mode="json")
        )
        assert prepared.score(json.dumps(prepared.expected.values)).exact_match
        assert PreparedRetrieval.model_validate_json(prepared.model_dump_json()) == prepared
    small_records = set(records_and_query(small)[0])
    large_records = set(records_and_query(large)[0])
    assert small_records <= large_records


async def test_depth_changes_position_but_not_facts_or_question():
    fixture = RulerFixture()
    start, middle, end = [
        await fixture.prepare(4096, count, "test", needle_depth=depth) for depth in (0, 0.5, 1)
    ]
    assert start.expected == middle.expected == end.expected
    key = next(iter(start.expected.values))
    assert start.provenance["needle_record_positions"][key] == 0
    assert end.provenance["needle_record_positions"][key] == end.provenance["distractor_records"]
    assert (
        start.provenance["needle_prefix_render_tokens"][key]
        < middle.provenance["needle_prefix_render_tokens"][key]
        < end.provenance["needle_prefix_render_tokens"][key]
    )
    different = await RulerFixture(seed=43).prepare(4096, count, "test")
    assert different.expected != start.expected


@pytest.mark.parametrize(
    "response,correct,exact,valid",
    [
        (' {"a":"123", "b":"456"} ', 2, True, True),
        ('{"b":"456", "a":"123"}', 2, True, True),
        ('{"a":"123"}', 1, False, True),
        ('{"a":"123","b":"wrong"}', 1, False, True),
        ('{"a":"123","b":"456","extra":"guess"}', 2, False, True),
        ('{"a":["wrong","123"],"b":"456"}', 0, False, False),
        ('{"a":123,"b":"456"}', 0, False, False),
        ('{"a":"wrong","a":"123","b":"456"}', 0, False, False),
        ('Here are the answers: {"a":"123","b":"456"}', 0, False, False),
        ('```json\n{"a":"123","b":"456"}\n```', 0, False, False),
        ('["123","456"]', 0, False, False),
        ("null", 0, False, False),
        ('{"a":"123","b":NaN}', 0, False, False),
        ("", 0, False, False),
    ],
)
def test_strict_scoring_rejects_substring_guesses_and_duplicate_keys(
    response, correct, exact, valid
):
    score = RetrievalAnswers(values={"a": "123", "b": "456"}).score(response)
    assert (score.correct, score.total, score.exact_match, score.format_valid) == (
        correct,
        2,
        exact,
        valid,
    )


@pytest.mark.parametrize(
    "target,depth", [(-1, 0.5), (True, 0.5), (1, -0.1), (1, 1.1), (1, float("nan"))]
)
async def test_invalid_sizing_rejected(target, depth):
    with pytest.raises(ValueError):
        await RulerFixture().prepare(target, count, "test", needle_depth=depth)


async def test_bad_counter_and_capacity_exhaustion_fail_explicitly(monkeypatch):
    from benchmark_fixtures import ruler

    async def invalid(context):
        return True

    async def constant(context):
        return 1

    with pytest.raises(ValueError, match="invalid token count"):
        await RulerFixture().prepare(100, invalid, "test")
    with pytest.raises(ValueError, match="did not grow"):
        await RulerFixture().prepare(100, constant, "test")
    monkeypatch.setattr(ruler, "MAX_RECORDS", 2)
    with pytest.raises(ValueError, match="record limit"):
        await RulerFixture().prepare(10000, count, "test")


@pytest.mark.parametrize("options", [{"seed": -1}, {"queries": 0}, {"queries": 17}, {"queries": 2}])
def test_invalid_fixture_configuration_rejected(options):
    with pytest.raises(ValueError):
        RulerFixture(**options)
