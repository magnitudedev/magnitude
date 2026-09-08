"""Resolve Metal dependencies and emit a deterministic, inspectable kernel source."""

from dataclasses import dataclass
from functools import cache

from .plan import Scalar, Source


@dataclass(frozen=True)
class Assembly:
    body: str
    header: str
    sources: tuple[tuple[str, str], ...]


def merge_sources(sources) -> tuple[tuple[str, str], ...]:
    resolved: dict[str, str] = {}
    definitions: dict[str, Source] = {}
    visiting: set[str] = set()

    def visit(item: Source):
        if item.path in visiting:
            raise ValueError(f"cyclic Metal dependency: {item.path}")
        if item.path in resolved:
            if definitions[item.path] != item:
                raise ValueError(f"conflicting Metal dependency: {item.path}")
            return
        visiting.add(item.path)
        for child in item.dependencies:
            visit(child)
        resolved[item.path] = item.text
        definitions[item.path] = item
        visiting.remove(item.path)

    for source in sources:
        visit(source)
    return tuple(resolved.items())


def source_files(source: Source) -> tuple[tuple[str, str], ...]:
    return merge_sources((source,))


def declaration(scalar: Scalar) -> str:
    return f"#define {scalar.name} ({scalar.literal})\n"


@cache
def assemble(source: Source, constants: tuple[Scalar, ...] = ()) -> Assembly:
    sources = source_files(source)
    header = "".join(map(declaration, constants))
    header += "\n".join(text for _, text in sources[:-1])
    return Assembly(sources[-1][1], header, sources)


@dataclass(frozen=True)
class Captures:
    """One generated ABI for typed captured operands."""

    names: tuple[str, ...]

    @property
    def parameters(self):
        return ", ".join(f"typename A{i}" for i in range(len(self.names)))

    @property
    def arguments(self):
        return ", ".join(f"decltype({name})" for name in self.names)

    @property
    def fields(self):
        return tuple(f"A{i} {name};" for i, name in enumerate(self.names))
