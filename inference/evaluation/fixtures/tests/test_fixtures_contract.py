"""Every family emits the contract's fixture shape, deterministically."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from magnitude_fixtures import (
    FAMILIES,
    Fixture,
    Source,
    code,
    encode_set,
    prose,
    read_set,
    retrieval,
    tools,
    write_set,
    vision,
)

SOURCE = Source(name="test", digest="0" * 64)

KEYS = ["id", "family", "source", "tools", "messages", "generation", "sizing"]


@pytest.fixture
def families(interactions, book, files):
    return {
        "tools": lambda: tools.build(interactions, SOURCE, count=2),
        "prose-continue": lambda: prose.build(book, SOURCE, "prose-continue", count=2),
        "prose-repeat": lambda: prose.build(book, SOURCE, "prose-repeat", count=2),
        "prose-shift": lambda: prose.build(book, SOURCE, "prose-shift", count=2),
        "retrieval": lambda: retrieval.build(count=2, queries=2),
        "vision": lambda: list(vision.fixtures()),
        "code": lambda: code.build(files, SOURCE, count=2),
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_records_follow_the_contract(families, family):
    content = encode_set(families[family]())
    for line in content.decode().splitlines():
        record = json.loads(line)
        assert [key for key in KEYS if key in record] == list(record)
        assert record["family"] == family and record["id"].startswith(family + "/")
        assert set(record["source"]) == {"name", "digest"}
        assert ("tools" in record) == (family == "tools")
        assert ("generation" in record) == (family != "code")
        messages = record["messages"]
        for message in messages:
            assert message["role"] in ("system", "user", "assistant", "tool")
            assert message.get("scored", True) is True
            assert "scored" not in message or message["role"] == "assistant"
        if family == "vision":
            assert record["sizing"]["unit"] == "fixed"
            continue
        assert record["sizing"]["unit"] == "turn"
        head = record["sizing"]["keep_head"]
        assert messages[head]["role"] == "user"
        if "generation" in record:
            generation = record["generation"]
            assert set(generation) == {"prefix", "max_tokens", "expected"}
            assert messages[generation["prefix"]]["role"] == "assistant"
            assert messages[generation["prefix"] - 1]["role"] == "user"
        assert Fixture.model_validate_json(line).line() == line


@pytest.mark.parametrize("family", FAMILIES)
def test_builds_are_byte_identical(families, family):
    assert encode_set(families[family]()) == encode_set(families[family]())


def test_set_roundtrip_reports_file_digest(tmp_path, families):
    path = tmp_path / "set.jsonl"
    fixtures = families["retrieval"]()
    written = write_set(path, fixtures)
    assert written == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_set(path) == (fixtures, written)


def test_sets_are_identical_across_processes(tmp_path):
    digests = set()
    for seed in ("1", "2"):
        path = tmp_path / f"{seed}.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; from magnitude_fixtures import build, write_set; "
                "write_set(Path(sys.argv[1]), build('retrieval', count=2))",
                str(path),
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(digests) == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"messages": [{"role": "user", "content": "x", "scored": True}]}, "only assistant"),
        ({"messages": [{"role": "user", "content": "x"}]}, "at least one scored"),
        ({"generation": {"prefix": 5, "max_tokens": 1, "expected": {}}}, "outside the request"),
        ({"generation": {"prefix": 3, "max_tokens": 1, "expected": {}}}, "answer begins"),
        ({"generation": {"prefix": 2, "max_tokens": 1, "expected": {}}}, "lie in the probe"),
        (
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a", "scored": True},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a", "scored": True},
                ]
            },
            "must lie in the probe",
        ),
        ({"sizing": {"unit": "turn", "keep_head": 0}}, "user message right after"),
    ],
)
def test_invalid_fixtures_are_refused(change, reason):
    record = {
        "id": "code/x",
        "family": "code",
        "source": SOURCE.model_dump(),
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "scored": True},
        ],
        "sizing": {"unit": "turn", "keep_head": 1},
    }
    Fixture.model_validate(record)
    with pytest.raises(ValueError, match=reason):
        Fixture.model_validate({**record, **change})


def rendered(messages, tools) -> int:
    return len(json.dumps([messages, tools]))


def test_sizing_keeps_head_and_probe_then_newest_history(interactions):
    fixture = tools.build(interactions, SOURCE, count=1)[0]
    units = fixture.units()
    smallest = fixture.conversation_with(0)
    assert smallest.messages == fixture.chat_messages(1) + fixture.chat_messages()[units[-1][0] :]
    assert smallest.prefix == 2 and smallest.messages[smallest.prefix]["role"] == "assistant"
    assert smallest.scored == [smallest.prefix]
    assert smallest.tools == fixture.tools
    lengths = [rendered(c.messages, c.tools) for c in map(fixture.conversation_with, range(6))]
    assert lengths == sorted(lengths)
    assert fixture.sized(lengths[0], rendered) == smallest
    assert fixture.sized(lengths[1] - 1, rendered) == smallest
    for history in range(1, 6):
        # The most history at or under the target: exactly at, and just short of the next.
        targets = [lengths[history], *([lengths[history + 1] - 1] if history < 5 else [])]
        for target in targets:
            sized = fixture.sized(target, rendered)
            assert sized == fixture.conversation_with(history)
            assert rendered(sized.messages, sized.tools) <= target
        kept = units[len(units) - 1 - history : -1]
        assert (
            sized.messages[1 : len(sized.messages) - 3]
            == fixture.chat_messages()[kept[0][0] : kept[-1][1]]
        )
        assert sized.prefix == fixture.generation.prefix - (
            len(fixture.messages) - len(sized.messages)
        )
        assert all("scored" not in m and "tools" not in m for m in sized.messages)
    whole = fixture.conversation_with(None)
    assert fixture.sized(rendered(whole.messages, whole.tools) * 10, rendered) == whole
    with pytest.raises(ValueError, match="head and probe alone exceed"):
        fixture.sized(lengths[0] - 1, rendered)


def test_offered_tools_are_probe_tools_then_kept_unit_tools_deduplicated(interactions):
    fixture = tools.build(interactions, SOURCE, count=1)[0]
    # History is decisions 1..5 calling echo1, echo2, echo0, echo1, echo2; the probe calls echo0.
    names = [t["function"]["name"] for t in fixture.conversation_with(4).tools]
    assert names == ["echo0", "echo2", "echo1"]
    offered = fixture.conversation_with(None).tools
    assert offered[0] == interactions[0].tools[0]
    assert offered[1:] == [interactions[1].tools[0], interactions[2].tools[0]]


@pytest.mark.parametrize("index", [0, 3])
def test_unit_tools_only_on_history_questions(index):
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "scored": True},
    ]
    tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
    record = {
        "id": "code/x",
        "family": "code",
        "source": SOURCE.model_dump(),
        "messages": messages,
        "sizing": {"unit": "turn", "keep_head": 1},
    }
    Fixture.model_validate(
        {**record, "messages": [*messages[:1], {**messages[1], "tools": [tool]}, *messages[2:]]}
    )
    changed = [dict(m) for m in messages]
    changed[index]["tools"] = [tool]
    with pytest.raises(ValueError, match="history unit"):
        Fixture.model_validate({**record, "messages": changed})
