"""Language-preserving orientation of bounded right-linear grammar regions.

Native delimiter scanners use right recursion. Earley parsing retains possible
scanner endings across the entire preceding text. The equivalent left-linear
productions describe paths *to* each scanner state and keep that work bounded by
the automaton, while preserving parser ambiguity at delimiter boundaries.
"""

from llguidance import gbnf_to_lark as ast


def _references(node: ast.ASTNode):
    if isinstance(node, ast.RuleRefNode):
        yield node.name
    for child in node.children():
        yield from _references(child)


def _right_linear_regions(rules: dict[str, ast.RuleNode]):
    edges: dict[str, list[tuple[list[ast.ASTNode], str | None]]] = {}
    for name, rule in rules.items():
        body = rule.alternatives
        alternatives = body.alternatives if isinstance(body, ast.AlternativeNode) else [body]
        paths = []
        for alternative in alternatives:
            nodes = (
                list(alternative.nodes)
                if isinstance(alternative, ast.SequenceNode)
                else [alternative]
            )
            tail = nodes[-1] if nodes else None
            target = tail.name if isinstance(tail, ast.RuleRefNode) else None
            if target is not None:
                nodes.pop()
            if any(tuple(_references(node)) for node in nodes):
                break
            paths.append((nodes, target))
        else:
            edges[name] = paths
    # Keep the greatest closed set, including mutually recursive scanners.
    while True:
        rejected = {
            name for name, paths in edges.items()
            if any(target is not None and target not in edges for _, target in paths)
        }
        if not rejected:
            break
        for name in rejected:
            del edges[name]
    entries = ({"root"} if "root" in edges else set()) | {
        target
        for name, rule in rules.items() if name not in edges
        for target in _references(rule) if target in edges
    }
    return edges, entries


def orient_regular_regions(rules: dict[str, ast.RuleNode]) -> None:
    """Reverse productions, never input bytes; leave nonregular CFG rules intact.

    For A -> text B, introduce path-to-B -> path-to-A text. The entry state's
    path admits epsilon. Each accepting edge A -> text contributes path-to-A
    text to the result. Thus both grammars describe the same labeled paths.
    """
    edges, entries = _right_linear_regions(rules)
    added = 0
    for index, entry in enumerate(sorted(entries)):
        region: set[str] = set()
        pending = [entry]
        while pending and len(region) <= 64:
            name = pending.pop()
            if name not in region:
                region.add(name)
                pending.extend(target for _, target in edges[name] if target is not None)
        if len(region) > 64 or added + len(region) > 4096:
            continue
        # Acyclic regions already become lexer terminals in upstream conversion.
        remaining = set(region)
        while remaining:
            leaves = {
                name for name in remaining
                if all(target not in remaining for _, target in edges[name])
            }
            if not leaves:
                break
            remaining -= leaves
        if not remaining:
            continue
        # Input names have already been alpha-renamed to root/g<number>.
        names = {name: f"scan{index}state{i}" for i, name in enumerate(sorted(region))}
        incoming: dict[str, list[ast.ASTNode]] = {name: [] for name in region}
        incoming[entry].append(ast.SequenceNode([]))
        endings: list[ast.ASTNode] = []
        for name in sorted(region):
            for nodes, target in edges[name]:
                path = ast.SequenceNode([ast.RuleRefNode(names[name]), *nodes])
                if target is None:
                    endings.append(path)
                else:
                    incoming[target].append(path)
        if not endings:
            continue
        for name in sorted(region):
            key = names[name]
            rules[key] = ast.RuleNode(key, ast.AlternativeNode(incoming[name]), "")
        rules[entry].alternatives = ast.AlternativeNode(endings)
        added += len(region)
    reachable: set[str] = set()
    pending = ["root"]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(_references(rules[name]))
    for name in set(rules) - reachable:
        del rules[name]
