"""Tool sessions from pinned BFCL v4 decisions, with canonically completed history.

Each fixture is one session: a system message, every other selected BFCL decision completed
with its ground-truth calls and a synthetic acknowledgement reply, then the probe decision.
Only the probe's call message is scored and served. The top-level ``tools`` are the probe's
functions; each history question carries the functions its calls use, so a sized
conversation offers exactly the tools of the turns it keeps.
"""

import json
from pathlib import Path

from pydantic import Field, JsonValue

from .canonical import Record, digest, encoded
from .fixture import Fixture, Generation, Message, Source, TurnSizing
from .sources import DATA, LockedFile, cache_root, fetch

CATEGORIES = {
    "simple-python": "simple_python",
    "parallel": "parallel",
    "parallel-multiple": "parallel_multiple",
}
MAX_OUTPUT_TOKENS = 32_768


class CorpusLock(Record):
    repository: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    dataRoot: str
    license: str
    files: list[LockedFile]


class ExpectedCall(Record):
    """A call's name with the allowed values of each argument (BFCL possible answers)."""

    name: str
    arguments: dict[str, list[JsonValue]]


class Interaction(Record):
    id: str
    category: str
    question: str
    tools: list[Message]
    expected: list[ExpectedCall]

    def turn(self, call_prefix: str, *, probe: bool) -> list[Message]:
        """The decision completed with its first allowed argument values.

        A history turn's question carries the functions it calls; the probe's functions are
        the fixture's top-level tools and its call message is scored.
        """
        calls: list[JsonValue] = []
        replies: list[Message] = []
        for index, call in enumerate(self.expected):
            arguments = {key: values[0] for key, values in call.arguments.items() if values}
            call_id = f"{call_prefix}_{index}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": encoded(arguments)},
                }
            )
            replies.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": encoded({"ok": True, "tool": call.name, "arguments": arguments}),
                }
            )
        question: Message = {"role": "user", "content": self.question}
        answer: Message = {"role": "assistant", "content": "", "tool_calls": calls}
        if probe:
            answer["scored"] = True
        else:
            question["tools"] = self.tools
        return [question, answer, *replies]


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    kinds = {"dict": "object", "list": "array", "tuple": "array", "float": "number"}
    result = {}
    for key, item in value.items():
        if key in ("optional", "format") or (key == "type" and item == "any"):
            continue
        if key == "required":
            properties = value.get("properties")
            if not isinstance(properties, dict):
                continue
            item = [name for name in item if name in properties]
            if not item:
                continue
        result[key] = (
            kinds.get(item, item)
            if key == "type" and isinstance(item, str)
            else normalize_schema(item)
        )
    return result


def materialize(question: dict, answer: dict, category: str) -> Interaction:
    if answer["id"] != question["id"]:
        raise ValueError("BFCL question and answer identities differ")
    turns = question["question"]
    messages = turns[0] if isinstance(turns[0], list) else turns
    if [m["role"] for m in messages] != ["user"]:
        raise ValueError(f"BFCL record {question['id']} is not one user question")
    expected = [
        ExpectedCall(
            name=name,
            arguments={
                key: values if isinstance(values, list) else [values]
                for key, values in arguments.items()
            },
        )
        for entry in answer["ground_truth"]
        for name, arguments in entry.items()
    ]
    if not expected or not question["function"]:
        raise ValueError(f"BFCL record {question['id']} is incomplete")
    return Interaction(
        id=question["id"],
        category=category,
        question=messages[0]["content"],
        expected=expected,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": f["name"],
                    "description": f.get("description", "BFCL function"),
                    "parameters": normalize_schema(f["parameters"]),
                },
            }
            for f in question["function"]
        ],
    )


def load(*, cache: Path | None = None) -> tuple[list[Interaction], Source]:
    """The pinned BFCL decisions in their stable interleaved selection order."""
    lock = CorpusLock.model_validate_json((DATA / "bfcl-v4.lock.json").read_text())
    selection = json.loads((DATA / "bfcl-v4.selection.json").read_text())
    directory = (cache or cache_root()) / "bfcl" / lock.commit
    repository = lock.repository.replace("github.com", "raw.githubusercontent.com")
    for item in lock.files:
        url = f"{repository}/{lock.commit}/{lock.dataRoot}/{item.path}"
        fetch(url, item.sha256, directory / item.path)

    def read(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    cohorts = []
    for category, stem in CATEGORIES.items():
        filename = f"BFCL_v4_{stem}.json"
        answers = {a["id"]: a for a in read(directory / "possible_answer" / filename)}
        cohort = []
        for question in read(directory / filename):
            if question["id"] in selection["exclusions"]:
                continue
            if question["id"] not in answers:
                raise ValueError(f"BFCL answer missing: {question['id']}")
            cohort.append(materialize(question, answers[question["id"]], category))
        cohorts.append(cohort)
    interactions = [
        cohort[i] for i in range(max(map(len, cohorts))) for cohort in cohorts if i < len(cohort)
    ][: selection["maximumRecords"]]
    return interactions, Source(
        name="bfcl-v4", digest=digest({"lock": lock.model_dump(), "selection": selection})
    )


def build(
    interactions: list[Interaction],
    source: Source,
    *,
    count: int,
    categories: tuple[str, ...] = tuple(CATEGORIES),
) -> list[Fixture]:
    """One session per probe: the first ``count`` decisions of the selected categories."""
    selected = [i for i in interactions if i.category in categories]
    if not 1 <= count <= len(selected):
        raise ValueError(f"tools fixtures need 1..{len(selected)} probes, got {count}")
    fixtures = []
    for lane, probe in enumerate(selected[:count]):
        history = selected[lane + 1 :] + selected[:lane]
        messages: list[Message] = [
            {"role": "system", "content": f"Session {lane}. Use the supplied tools."}
        ]
        for turn, interaction in enumerate(history):
            messages.extend(interaction.turn(f"call_{lane}_{turn}", probe=False))
        prefix = len(messages) + 1
        messages.extend(probe.turn(f"call_{lane}_{len(history)}", probe=True))
        fixtures.append(
            Fixture(
                id=f"tools/bfcl/{probe.id}",
                family="tools",
                source=source,
                tools=probe.tools,
                messages=messages,
                generation=Generation(
                    prefix=prefix,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    expected={"calls": [call.model_dump(mode="json") for call in probe.expected]},
                ),
                sizing=TurnSizing(keep_head=1),
            )
        )
    return fixtures
