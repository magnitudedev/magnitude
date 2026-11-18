"""Templates: one benchmark request each, named ``<key>:<context>`` (e.g. ``repeat:8k``).

A template is lane 0's session of its key, sized to its context; a run of C sequences uses
lanes 0 … C − 1, each a different item of the fixture sized to the same context. This module
is the only place a template name is parsed or composed.

A template file is what the harness reads: the template's name, its context in tokens, the
fixture's source and the sessions of the lanes a run needs, in lane order. Its context comes
from the name and nowhere else; sizing to it happens where the model's chat template renders.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from . import bfcl, bigcodebench, moby, ruler
from .canonical import Record
from .session import KEYS, Key, Session, Source

#: Tokens per ``k`` of a context: ``1k`` is 1024 tokens.
K = 1024
CONTEXT = re.compile(r"^([1-9][0-9]*)k$")


def _grammar() -> str:
    return f"<key>:<n>k, with keys {', '.join(KEYS)}"


class Template(Record):
    key: Key
    context: int = Field(gt=0, multiple_of=K)

    @classmethod
    def parse(cls, text: str) -> "Template":
        """The template ``text`` names, or ValueError naming the valid ones."""
        key, colon, context = text.partition(":")
        size = CONTEXT.match(context)
        if not colon or size is None:
            raise ValueError(f"{text!r} is not a template name: {_grammar()}")
        if key not in KEYS:
            raise ValueError(f"{text!r} names no template: {_grammar()}")
        return cls(key=key, context=int(size.group(1)) * K)

    @property
    def name(self) -> str:
        return f"{self.key}:{self.context // K}k"

    def build(self, lanes: int) -> "TemplateFile":
        """The template file holding lanes 0 … ``lanes`` − 1 of this key, built from its
        pinned source; fails when the fixture cannot yield that many distinct lanes."""
        if lanes < 1:
            raise ValueError("a template file holds at least one lane")
        if self.key in ("repeat", "twist"):
            text, source = moby.load()
            sessions = moby.build(text, self.key, lanes=lanes)
        elif self.key == "retrieve":
            source = ruler.source()
            sessions = ruler.build(lanes=lanes)
        elif self.key == "code":
            tasks, source = bigcodebench.load()
            sessions = bigcodebench.build(tasks, lanes=lanes)
        else:
            interactions, source = bfcl.load()
            sessions = bfcl.build(interactions, lanes=lanes)
        return TemplateFile(template=self.name, context=self.context, source=source,
                            sessions=sessions)


class TemplateFile(Record):
    template: str
    context: int
    source: Source
    sessions: list[Session] = Field(min_length=1)

    @model_validator(mode="after")
    def named(self) -> Self:
        template = Template.parse(self.template)
        if template.name != self.template or template.context != self.context:
            raise ValueError(f"a template file of {self.template} holds context {self.context}")
        for lane, session in enumerate(self.sessions):
            if session.key != template.key:
                raise ValueError(f"lane {lane} is a {session.key} session, "
                                 f"not one of {self.template}")
        return self

    def encode(self) -> bytes:
        """The file's bytes: compact JSON in field order (a chat template prints tool schemas
        in their key order, which the harness preserves)."""
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode() + b"\n"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    @classmethod
    def read(cls, path: Path) -> "TemplateFile":
        return cls.model_validate_json(path.read_bytes())
