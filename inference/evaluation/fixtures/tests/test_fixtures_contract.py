"""Every template file follows the contract, deterministically; names parse one way."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from magnitude_fixtures import (
    KEYS,
    Session,
    Source,
    Template,
    TemplateFile,
    bfcl,
    bigcodebench,
    moby,
    ruler,
)

SOURCE = Source(name="test", digest="0" * 64)

SESSION_FIELDS = ["key", "tools", "messages", "generation", "keep_head"]
TEMPLATES = [f"{key}:1k" for key in KEYS]


@pytest.fixture
def built(interactions, book, tasks):
    """Each template's file from the test sources, as `Template.build` assembles it."""

    def build(name: str, lanes: int = 2) -> TemplateFile:
        template = Template.parse(name)
        sessions = {
            "repeat": lambda: moby.build(book, "repeat", lanes=lanes),
            "twist": lambda: moby.build(book, "twist", lanes=lanes),
            "retrieve": lambda: ruler.build(lanes=lanes),
            "code": lambda: bigcodebench.build(tasks, lanes=lanes),
            "tools": lambda: bfcl.build(interactions, lanes=lanes),
        }[template.key]()
        return TemplateFile(template=template.name, context=template.context, source=SOURCE,
                            sessions=sessions)

    return build


@pytest.mark.parametrize("name", TEMPLATES)
def test_files_follow_the_contract(built, name):
    file = built(name)
    record = json.loads(file.encode())
    assert list(record) == ["template", "context", "source", "sessions"]
    assert record["template"] == name and record["context"] == 1024
    for session in record["sessions"]:
        assert list(session) == SESSION_FIELDS
        assert session["key"] == name.split(":")[0]
        assert bool(session["tools"]) == (session["key"] == "tools")
        messages = session["messages"]
        for message in messages:
            assert message["role"] in ("system", "user", "assistant", "tool")
            assert message.get("scored", True) is True
            assert "scored" not in message or message["role"] == "assistant"
        assert messages[session["keep_head"]]["role"] == "user"
        generation = session["generation"]
        assert set(generation) == {"prefix", "max_tokens", "expected"}
        assert messages[generation["prefix"]]["role"] == "assistant"
        assert messages[generation["prefix"] - 1]["role"] == "user"
    assert TemplateFile.model_validate_json(file.encode()) == file


@pytest.mark.parametrize("name", TEMPLATES)
def test_builds_are_byte_identical(built, name):
    assert built(name).encode() == built(name).encode()


def test_file_roundtrip_and_digest(tmp_path, built):
    path = tmp_path / "template.json"
    file = built("retrieve:8k")
    path.write_bytes(file.encode())
    assert file.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert TemplateFile.read(path) == file


def test_files_are_identical_across_processes(tmp_path):
    digests = set()
    for seed in ("1", "2"):
        path = tmp_path / f"{seed}.json"
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; from magnitude_fixtures import Template; "
                "Path(sys.argv[1]).write_bytes(Template.parse('retrieve:4k').build(2).encode())",
                str(path),
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(digests) == 1


@pytest.mark.parametrize(
    "name,key,context",
    [("repeat:1k", "repeat", 1024), ("twist:4k", "twist", 4096),
     ("tools:32k", "tools", 32768), ("retrieve:8k", "retrieve", 8192),
     ("code:8k", "code", 8192)],
)
def test_names_parse_to_one_template_and_back(name, key, context):
    template = Template.parse(name)
    assert (template.key, template.context) == (key, context)
    assert template.name == name


@pytest.mark.parametrize(
    "name",
    ["repeat", "repeat:1024", "repeat:01k", "repeat:0k", "prose-repeat:4k",
     "moby-repeat:4k", "bfcl-tools:4k", "ruler-retrieve:4k", "code-recall:4k",
     "continue:1k", "vision:1k", "moby:1k", "repeat:1.5k"],
)
def test_other_names_are_refused_naming_the_valid_ones(name):
    with pytest.raises(ValueError, match="repeat, twist, retrieve, code, tools"):
        Template.parse(name)


def test_a_file_holds_only_its_templates_sessions(built):
    file = built("repeat:1k")
    with pytest.raises(ValueError, match="not one of twist:1k"):
        TemplateFile(template="twist:1k", context=1024, source=SOURCE, sessions=file.sessions)
    with pytest.raises(ValueError, match="holds context"):
        TemplateFile(template="repeat:1k", context=2048, source=SOURCE, sessions=file.sessions)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"messages": [{"role": "user", "content": "x", "scored": True}], "keep_head": 0}, "only assistant"),
        ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "a"}],
          "generation": {"prefix": 1, "max_tokens": 1, "expected": {}}, "keep_head": 0}, "at least one scored"),
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
        ({"keep_head": 0}, "user message right after"),
        ({"key": "unknown"}, "Input should be"),
    ],
)
def test_invalid_sessions_are_refused(change, reason):
    record = {
        "key": "repeat",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a", "scored": True},
        ],
        "generation": {"prefix": 4, "max_tokens": 1, "expected": {}},
        "keep_head": 1,
    }
    Session.model_validate(record)
    with pytest.raises(ValueError, match=reason):
        Session.model_validate({**record, **change})


def rendered(messages, tools) -> int:
    return len(json.dumps([messages, tools]))


def test_sizing_keeps_head_and_probe_then_newest_history(interactions):
    session = bfcl.build(interactions, lanes=1)[0]
    units = session.units()
    smallest = session.conversation_with(0)
    assert smallest.messages == session.chat_messages(1) + session.chat_messages()[units[-1][0] :]
    assert smallest.prefix == 2 and smallest.messages[smallest.prefix]["role"] == "assistant"
    assert smallest.scored == [smallest.prefix]
    assert smallest.tools == session.tools
    lengths = [rendered(c.messages, c.tools) for c in map(session.conversation_with, range(6))]
    assert lengths == sorted(lengths)
    assert session.sized(lengths[0], rendered) == smallest
    assert session.sized(lengths[1] - 1, rendered) == smallest
    for history in range(1, 6):
        # The most history at or under the target: exactly at, and just short of the next.
        targets = [lengths[history], *([lengths[history + 1] - 1] if history < 5 else [])]
        for target in targets:
            sized = session.sized(target, rendered)
            assert sized == session.conversation_with(history)
            assert rendered(sized.messages, sized.tools) <= target
        kept = units[len(units) - 1 - history : -1]
        assert (
            sized.messages[1 : len(sized.messages) - 3]
            == session.chat_messages()[kept[0][0] : kept[-1][1]]
        )
        assert sized.prefix == session.generation.prefix - (
            len(session.messages) - len(sized.messages)
        )
        assert all("scored" not in m and "tools" not in m for m in sized.messages)
    whole = session.conversation_with(None)
    assert session.sized(rendered(whole.messages, whole.tools) * 10, rendered) == whole
    with pytest.raises(ValueError, match="head and probe alone exceed"):
        session.sized(lengths[0] - 1, rendered)


def test_offered_tools_are_probe_tools_then_kept_unit_tools_deduplicated(interactions):
    session = bfcl.build(interactions, lanes=1)[0]
    # History is decisions 1..5 calling echo1, echo2, echo0, echo1, echo2; the probe calls echo0.
    names = [t["function"]["name"] for t in session.conversation_with(4).tools]
    assert names == ["echo0", "echo2", "echo1"]
    offered = session.conversation_with(None).tools
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
        "key": "repeat",
        "messages": messages,
        "generation": {"prefix": 4, "max_tokens": 1, "expected": {}},
        "keep_head": 1,
    }
    Session.model_validate(
        {**record, "messages": [*messages[:1], {**messages[1], "tools": [tool]}, *messages[2:]]}
    )
    changed = [dict(m) for m in messages]
    changed[index]["tools"] = [tool]
    with pytest.raises(ValueError, match="history unit"):
        Session.model_validate({**record, "messages": changed})
