"""Graph views shared by the terminal and generated documentation."""

from performance.records import Assembly


def dimensions(state: dict, view: dict, path: str) -> dict:
    key = view["assessments"].get(path)
    return state["components"].get(key, {}).get("dimensions", {})


def annotation(values: dict, *, references=False, unknown=False) -> str:
    result = []
    for name, value in values.items():
        if value["percent"] is None or value["issue"]:
            if unknown:
                result.append(f"{name}: —" if len(values) > 1 else "—")
            continue
        number = ("~" if value["estimated"] else "") + f"≥{value['percent']:.2f}%"
        label = f"{name}: {number}" if len(values) > 1 else number
        if references and value.get("benchmark") and value["benchmark"] != "composed":
            label += " @" + value["benchmark"]
        result.append(label)
    return "    [" + ", ".join(result) + "]" if result else ""


def graph_for(state: dict, view: dict) -> Assembly:
    return Assembly.read(state["compositions"][view["composition"]]["revisions"][view["revision"]])


def edges(node):
    import re

    def natural(item):
        return tuple((0, int(p)) if p.isdigit() else (1, p) for p in re.split(r"(\d+)", item[0]))

    return sorted(node.children.items(), key=natural) + sorted(
        node.dependencies.items(), key=natural
    )


def render_tree(
    state: dict, view_id: str, *, root: str | None = None, depth: int | None = None
) -> str:
    view = state["views"][view_id]
    graph = graph_for(state, view)
    seen = set()
    lines = []

    def visit(path, prefix, branch, role="", level=0):
        node = graph.nodes[path]
        reference = path in seen
        lines.append(
            prefix
            + branch
            + (role + " · " if role else "")
            + node.implementation
            + annotation(dimensions(state, view, path), references=True)
            + (f" ↗ {path}" if reference else "")
        )
        if reference:
            return
        seen.add(path)
        children = edges(node)
        if children and depth is not None and level >= depth:
            lines[-1] += " …"
            return
        for i, (role, child) in enumerate(children):
            visit(
                child,
                prefix + ("    " if branch == "└── " else "│   " if branch else ""),
                "└── " if i == len(children) - 1 else "├── ",
                role,
                level + 1,
            )

    visit(root or graph.root, "", "")
    return "\n".join(lines) + "\n"


def export_document(state: dict, view_id: str, document, *, root=None, depth=None, store=None):
    """Replace the existing Assembly code block; provenance belongs to the export artifact."""
    import re
    from pathlib import Path

    from performance.records import digest
    from performance.store import Store, atomic

    document = Path(document)
    content = document.read_text()
    pattern = r"(?m)(^## Assembly\n\n```text\n)[\s\S]*?(^```\s*$)"
    tree = render_tree(state, view_id, root=root, depth=depth)
    updated, count = re.subn(pattern, lambda match: match[1] + tree + match[2], content, count=1)
    if count != 1:
        raise ValueError("document needs one Assembly section with a text code block")
    destination = (store or Store()).root / "exports" / (digest(str(document.resolve())) + ".json")
    atomic(
        destination,
        {
            "document": str(document.resolve()),
            "generation": state["generation"],
            "view_id": view_id,
            "view": state["views"][view_id],
            "root": root,
            "depth": depth,
            "tree": tree,
            "theory_revision": state["theory_revision"],
        },
    )
    from uuid import uuid4

    temporary = document.with_name("." + document.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(updated)
        temporary.replace(document)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
