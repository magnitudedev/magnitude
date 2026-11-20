"""The ``code`` fixture: stacked multi-turn Python coding sessions from BigCodeBench v0.1.4.

Each turn is one self-contained BigCodeBench task: the user supplies ``instruct_prompt`` and
the assistant answers with the canonical self-contained Python implementation in a single
fenced ``python`` block. History turns are earlier tasks so ``Session.sized`` hits any target
context length; lane ``k`` probes the ``k``-th task from the end of the dataset, so a lane's
session does not depend on how many lanes are built and lanes differ at the tail.
"""

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import Field

from .canonical import Record, digest
from .session import Generation, Message, Session, Source
from .sources import DATA, cache_root, fetch

OUTPUT_TOKENS = 1_024
SYSTEM = (
    "Coding session {lane}. For each request, write only the self-contained Python "
    "implementation in a single fenced ```python block, with no explanation."
)


class CorpusLock(Record):
    repository: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    file: str
    license: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Task(Record):
    task_id: str
    instruct_prompt: str
    code_prompt: str
    canonical_solution: str


def load(*, cache: Path | None = None) -> tuple[list[Task], Source]:
    """Fetch and verify BigCodeBench v0.1.4 on demand into the local source cache."""
    lock = CorpusLock.model_validate_json((DATA / "bigcodebench.lock.json").read_text())
    url = f"{lock.repository}/resolve/{lock.commit}/{lock.file}"
    path = (cache or cache_root()) / "bigcodebench" / lock.commit / lock.file
    fetch(url, lock.sha256, path)
    table = pq.read_table(
        path, columns=["task_id", "instruct_prompt", "code_prompt", "canonical_solution"]
    )
    tasks = [
        Task(
            task_id=table["task_id"][i].as_py(),
            instruct_prompt=table["instruct_prompt"][i].as_py(),
            code_prompt=table["code_prompt"][i].as_py(),
            canonical_solution=table["canonical_solution"][i].as_py(),
        )
        for i in range(table.num_rows)
    ]
    if len(tasks) < 2:
        raise ValueError("BigCodeBench dataset needs at least two tasks")
    return tasks, Source(name="bigcodebench", digest=digest(lock.model_dump()))


def turn(task: Task, *, scored: bool) -> list[Message]:
    code = f"{task.code_prompt}{task.canonical_solution.rstrip()}"
    answer: Message = {"role": "assistant", "content": f"```python\n{code}\n```"}
    if scored:
        answer["scored"] = True
    return [{"role": "user", "content": task.instruct_prompt.strip()}, answer]


def build(tasks: list[Task], *, lanes: int) -> list[Session]:
    """One session per lane: lane ``k`` probes the ``k``-th task from the end of the dataset
    after every preceding task."""
    if not 1 <= lanes < len(tasks):
        raise ValueError(f"code yields 1..{len(tasks) - 1} lanes, not {lanes}")
    sessions = []
    for lane in range(lanes):
        probe = len(tasks) - 1 - lane
        messages: list[Message] = [{"role": "system", "content": SYSTEM.format(lane=lane)}]
        for task in tasks[:probe]:
            messages.extend(turn(task, scored=False))
        prefix = len(messages) + 1
        messages.extend(turn(tasks[probe], scored=True))
        sessions.append(
            Session(
                key="code",
                messages=messages,
                generation=Generation(prefix=prefix, max_tokens=OUTPUT_TOKENS, expected={}),
                keep_head=1,
            )
        )
    return sessions
