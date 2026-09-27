"""Compile a wholly regular completion into one lexer language when bounded.

Partial-region lexer substitution is unsafe: greedy token boundaries can change
CFG ambiguity. This optimization is used only if the *entire* root becomes one
regular expression, so no internal parser/lexer boundary survives.
"""

from copy import deepcopy
from dataclasses import dataclass

from llguidance import gbnf_to_lark as ast

from templates.regular import _references, _right_linear_regions


@dataclass
class _Expression(ast.ASTNode):
    text: str

    def __str__(self) -> str:
        return self.text


class _TooLarge(Exception):
    pass


def _bounded(text: str) -> str:
    if len(text) > 256 * 1024:
        raise _TooLarge
    return text


def _union(left: str | None, right: str) -> str:
    return right if left is None or left == right else _bounded(f"({left} | {right})")


def _sequence(*parts: str) -> str:
    kept = [part for part in parts if part != '""']
    return _bounded(" ".join(kept)) if kept else '""'


def whole_completion(rules: dict[str, ast.RuleNode]) -> str | None:
    """State elimination preserves labeled accepting paths, with bounded expansion.

    Return None without mutating input if a CFG production remains or elimination
    exceeds its 64-state, 256KiB-expression, or 1MiB-working-graph bounds.
    """
    rules = deepcopy(rules)
    edges, entries = _right_linear_regions(rules)
    try:
        for entry in sorted(entries):
            region: set[str] = set()
            pending = [entry]
            while pending and len(region) <= 64:
                name = pending.pop()
                if name not in region:
                    region.add(name)
                    pending.extend(target for _, target in edges[name] if target is not None)
            if len(region) > 64:
                return None
            start, finish = object(), object()
            graph: dict[tuple[object, object], str] = {(start, entry): '""'}
            size = 2

            def add(source, target, expression):
                nonlocal size
                key = source, target
                previous = graph.get(key)
                combined = _union(previous, expression)
                size += len(combined) - (0 if previous is None else len(previous))
                if size > 1024 * 1024:
                    raise _TooLarge
                graph[key] = combined

            for name in sorted(region):
                for nodes, target in edges[name]:
                    add(name, finish if target is None else target, _sequence(*(str(n) for n in nodes)))
            remaining = set(region)
            while remaining:
                def degree(name):
                    incoming = sum(b == name and a != name for a, b in graph)
                    outgoing = sum(a == name and b != name for a, b in graph)
                    return incoming * outgoing, name

                state = min(remaining, key=degree)
                incoming = [(a, value) for (a, b), value in graph.items() if b == state and a != state]
                outgoing = [(b, value) for (a, b), value in graph.items() if a == state and b != state]
                loop = graph.get((state, state))
                repeat = '""' if loop is None or loop == '""' else _bounded(f"({loop})*")
                for source, before in incoming:
                    for target, after in outgoing:
                        add(source, target, _sequence(before, repeat, after))
                graph = {key: value for key, value in graph.items() if state not in key}
                size = sum(len(value) for value in graph.values())
                remaining.remove(state)
            if (start, finish) not in graph:
                return None
            rules[entry].alternatives = _Expression(graph[start, finish])
    except _TooLarge:
        return None
    reachable: set[str] = set()
    pending = ["root"]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(_references(rules[name]))
    rules = {name: rule for name, rule in rules.items() if name in reachable}
    ast.resolve(rules)
    root = rules["start"]
    if not root.alternatives.is_terminal():
        return None
    root.name = "WHOLE_COMPLETION"
    return "%llguidance {}\nstart: WHOLE_COMPLETION\n" + "\n".join(str(r) for r in rules.values()) + "\n"
