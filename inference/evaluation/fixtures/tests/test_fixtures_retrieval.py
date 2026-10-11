import json

import pytest
from magnitude_fixtures import retrieval, validation
from magnitude_fixtures.retrieval import RECORDS_PER_TURN, record


def lines(content: str) -> list[str]:
    head, *records = content.split("\n")
    assert head == "Records:"
    return records


def test_needles_sit_in_the_kept_head_and_only_the_probe_answers():
    fixture = retrieval.build(count=1, queries=4)[0]
    values = fixture.generation.expected["values"]
    head = fixture.messages[: fixture.sizing.keep_head]
    assert [m["role"] for m in head] == ["system", "user", "assistant"]
    needles = [f"{key}: {value}" for key, value in values.items()]
    first = lines(head[1]["content"])
    assert len(first) == RECORDS_PER_TURN
    middle = (RECORDS_PER_TURN - 4) // 2
    assert first[middle : middle + 4] == needles
    for start, end in fixture.units()[:-1]:
        user, answer = fixture.messages[start:end]
        assert answer == {"role": "assistant", "content": "Noted."}
        assert len(lines(user["content"])) == RECORDS_PER_TURN
        assert not any(key in user["content"] for key in values)
    query, answer = fixture.messages[-2:]
    assert query["content"] == "Return the values for these keys: " + ", ".join(values)
    assert json.loads(answer["content"]) == values and answer["scored"] is True


def test_records_are_unique_and_lanes_draw_independent_facts():
    first, second = retrieval.build(count=2, queries=1, seed=7)
    assert first.id == "retrieval/ruler/single-q1-s7"
    assert second.id == "retrieval/ruler/single-q1-s8"
    assert first.generation.expected == {"values": dict([record(7, 0)])}
    assert first.generation.expected != second.generation.expected
    keys = [
        line.split(":")[0]
        for message in first.messages
        if message["role"] == "user" and message["content"].startswith("Records:")
        for line in lines(message["content"])
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("queries", [0, 17])
def test_query_count_is_bounded(queries):
    with pytest.raises(ValueError, match="queries"):
        retrieval.build(count=1, queries=queries)


def test_canonical_answer_scores_exactly():
    fixture = retrieval.build(count=1, queries=3)[0]
    score = validation.retrieval(fixture.generation.expected, fixture.messages[-1]["content"])
    assert score.exact_match and score.correct == score.total == 3
