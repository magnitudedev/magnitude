"""Fixtures: the only model inputs of evaluation, as deterministic chat conversations."""

from . import code, prose, retrieval, tools, validation
from .fixture import (
    FAMILIES,
    Conversation,
    Family,
    FixedSizing,
    Fixture,
    Generation,
    Source,
    TurnSizing,
    encode_set,
    read_set,
    write_set,
)


def build(family: Family, *, count: int = 8, queries: int = 4) -> list[Fixture]:
    """Build one family's default fixture set from its pinned sources.

    ``count`` is the number of independent sessions (lanes); ``queries`` is the number of
    keys a ``retrieval`` probe asks for.
    """
    if family == "tools":
        return tools.build(*tools.load(), count=count)
    if family in ("prose-continue", "prose-repeat", "prose-shift"):
        return prose.build(*prose.load(), family, count=count)
    if family == "retrieval":
        return retrieval.build(count=count, queries=queries)
    if family == "code":
        return code.build(*code.load(), count=count)
    raise ValueError(f"unknown fixture family: {family}")


__all__ = [
    "FAMILIES",
    "Conversation",
    "Family",
    "FixedSizing",
    "Fixture",
    "Generation",
    "Source",
    "TurnSizing",
    "build",
    "code",
    "encode_set",
    "prose",
    "read_set",
    "retrieval",
    "tools",
    "validation",
    "write_set",
]
