"""Coding conversations over files of this repository at a pinned commit.

Each turn asks for one file and the assistant answers with its exact contents in a fenced
block. Files are read with read-only ``git show <commit>:<path>`` from the enclosing
checkout and verified against the lock, so building needs a git checkout that contains the
pinned commit. Lane ``k`` probes the ``k``-th locked file after every other file.
"""

import hashlib
import subprocess
from pathlib import Path

from pydantic import Field

from .canonical import Record, digest
from .fixture import Fixture, Message, Source, TurnSizing
from .sources import DATA, LockedFile

SYSTEM = "You are a coding assistant working in the Magnitude repository."
LANGUAGES = {".rs": "rust", ".ts": "typescript", ".py": "python"}


class CodeLock(Record):
    repository: str
    commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    files: list[LockedFile] = Field(min_length=2)


class File(Record):
    path: str
    content: str


def load(*, checkout: Path | None = None) -> tuple[list[File], Source]:
    lock = CodeLock.model_validate_json((DATA / "magnitude-code.lock.json").read_text())
    root = checkout or Path(__file__).parent

    def show(path: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{lock.commit}:{path}"],
            capture_output=True,
            check=True,
        )
        return result.stdout

    files = []
    for item in lock.files:
        content = show(item.path)
        if hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError(f"code source checksum mismatch: {item.path}")
        files.append(File(path=item.path, content=content.decode()))
    return files, Source(name="magnitude-code", digest=digest(lock.model_dump()))


def turn(file: File, *, scored: bool) -> list[Message]:
    language = LANGUAGES[Path(file.path).suffix]
    answer: Message = {
        "role": "assistant",
        "content": f"```{language}\n{file.content.rstrip()}\n```",
    }
    if scored:
        answer["scored"] = True
    return [{"role": "user", "content": f"Show me the contents of `{file.path}`."}, answer]


def build(files: list[File], source: Source, *, count: int) -> list[Fixture]:
    if not 1 <= count <= len(files):
        raise ValueError(f"code fixtures need 1..{len(files)} probes, got {count}")
    fixtures = []
    for lane, probe in enumerate(files[:count]):
        messages: list[Message] = [{"role": "system", "content": SYSTEM}]
        for file in files[lane + 1 :] + files[:lane]:
            messages.extend(turn(file, scored=False))
        messages.extend(turn(probe, scored=True))
        fixtures.append(
            Fixture(
                id=f"code/magnitude/{probe.path}",
                family="code",
                source=source,
                messages=messages,
                sizing=TurnSizing(keep_head=1),
            )
        )
    return fixtures
