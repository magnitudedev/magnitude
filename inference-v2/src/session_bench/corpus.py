"""Pinned BFCL source data, normalized independently of any inference engine."""

import hashlib
import json
from pathlib import Path

import httpx
from pydantic import Field

from .sessions import ExpectedCall, Interaction, Record, digest

CATEGORIES = {
    "simple-python": "simple_python",
    "parallel": "parallel",
    "parallel-multiple": "parallel_multiple",
}
DATA = Path(__file__).parent / "data"


class LockedFile(Record):
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class CorpusLock(Record):
    repository: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    dataRoot: str
    license: str
    files: list[LockedFile]


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    kinds = {"dict": "object", "list": "array", "tuple": "array", "float": "number"}
    return {
        key: kinds.get(item, item)
        if key == "type" and isinstance(item, str)
        else normalize_schema(item)
        for key, item in value.items()
        if not (key == "type" and item == "any")
    }


def materialize(question: dict, answer: dict, category: str, commit: str) -> Interaction:
    if answer["id"] != question["id"]:
        raise ValueError("BFCL question and answer identities differ")
    turns = question["question"]
    messages = turns[0] if isinstance(turns[0], list) else turns
    messages = [m for m in messages if m["role"] in ("system", "user")]
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
    if not messages or not expected or not question["function"]:
        raise ValueError(f"BFCL record {question['id']} is incomplete")
    return Interaction(
        id=question["id"],
        category=category,
        messages=messages,
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
        provenance={
            "commit": commit,
            "record_id": question["id"],
            "file": f"BFCL_v4_{CATEGORIES[category]}.json",
        },
    )


async def prepare(root: Path, categories: tuple[str, ...]) -> tuple[list[Interaction], str]:
    lock = CorpusLock.model_validate_json((DATA / "bfcl-v4.lock.json").read_text())
    selection = json.loads((DATA / "bfcl-v4.selection.json").read_text())
    cache = root / ".session-bench" / "corpus" / lock.commit
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for item in lock.files:
            path = cache / item.path
            content = path.read_bytes() if path.is_file() else b""
            if hashlib.sha256(content).hexdigest() != item.sha256:
                url = (
                    f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{lock.commit}/"
                    f"{lock.dataRoot}/{item.path}"
                )
                response = await client.get(url)
                response.raise_for_status()
                content = response.content
                if hashlib.sha256(content).hexdigest() != item.sha256:
                    raise ValueError(f"BFCL checksum mismatch: {item.path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)
    cohorts = []
    for category, stem in CATEGORIES.items():

        def read(path):
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        filename = f"BFCL_v4_{stem}.json"
        answers = {a["id"]: a for a in read(cache / "possible_answer" / filename)}
        cohort = []
        for question in read(cache / filename):
            if question["id"] in selection["exclusions"]:
                continue
            if question["id"] not in answers:
                raise ValueError(f"BFCL answer missing: {question['id']}")
            cohort.append(materialize(question, answers[question["id"]], category, lock.commit))
        cohorts.append(cohort)
    # Preserve the historical interleaved selection before applying user category filters.
    fixtures = [
        cohort[i] for i in range(max(map(len, cohorts))) for cohort in cohorts if i < len(cohort)
    ][: selection["maximumRecords"]]
    fixtures = [f for f in fixtures if f.category in categories]
    if not fixtures:
        raise ValueError("no BFCL interactions match the selected categories")
    return fixtures, digest(
        {
            "lock": lock.model_dump(),
            "selection": selection,
            "fixtures": [f.model_dump() for f in fixtures],
        }
    )
