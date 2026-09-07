"""One composition selector, its actual tree, and selected-component evidence."""

from rich.console import Group
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Select, Static, Tree

from performance.presentation import annotation, edges
from performance.records import Assembly
from performance.store import Store


def composition_label(record):
    graph = record["revisions"][record["current_revision"]]
    label = record["label"]
    if label.startswith("models--"):
        label = label.split("--", 2)[-1]
    if label == "VLM forward":
        config = graph.get("artifacts", {}).get("target", {}).get("config", {})
        model = config.get("model_type", "Upstream model")
        bits = config.get("quantization", {}).get("bits")
        label = f"{model} · {bits}-bit · MLX VLM forward" if bits else f"{model} · MLX VLM forward"
    return label


def metric(value, unit):
    if value is None:
        return "—"
    if unit == "seconds":
        if value < 0.001:
            return f"{value * 1e6:.3g} µs"
        if value < 1:
            return f"{value * 1000:.3g} ms"
        return f"{value:.4g} s"
    if unit == "bytes":
        for suffix, size in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
            if value >= size:
                return f"{value / size:.4g} {suffix}"
    return f"{value:.4g} {unit}"


def workload_label(point):
    parts = [str(point[k]) for k in ("mode", "fixture") if k in point]
    histories = point.get("histories")
    context = point.get("context_tokens", point.get("prefix_tokens"))
    if histories:
        context = (
            str(histories[0])
            if min(histories) == max(histories)
            else f"{min(histories)}–{max(histories)}"
        )
    if context is not None:
        parts.append(f"context {context}")
    for label, field in (
        ("query", "query_tokens"),
        ("batch", "batch_size"),
        ("rows", "rows"),
        ("retained positions", "retained_positions"),
        ("restore", "restore_mode"),
    ):
        if field in point:
            parts.append(f"{label} {point[field]}")
    return " · ".join(parts) or "See raw evidence for the recorded workload."


class PerformanceApp(App):
    TITLE = "Inference performance"
    CSS = """
    #composition { height: 3; width: 100%; margin: 0; }
    #main { height: 1fr; }
    #tree { width: 3fr; height: 100%; border: solid $primary; }
    #details-scroll { width: 2fr; height: 100%; border: solid $primary; }
    #details { height: auto; padding: 0 1; }
    #status { height: 1; color: $text-muted; }
    """
    BINDINGS = [
        ("c", "compositions", "Composition"),
        ("f", "focus_subsystem", "Focus subtree"),
        ("escape", "back", "Back"),
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, store: Store):
        super().__init__()
        self.store = store
        self.state = {}
        self.composition_id = None
        self.graph = None
        self.roots = []
        self._shown = None

    def compose(self) -> ComposeResult:
        yield Select([], prompt="Choose composition", id="composition")
        with Horizontal(id="main"):
            yield Tree("No recorded compositions", id="tree")
            with VerticalScroll(id="details-scroll"):
                yield Static(
                    "Run or import a benchmark to populate this view.", id="details", markup=False
                )
        yield Static(
            "Current evidence per component · — no efficiency · ~ estimated",
            id="status",
            markup=False,
        )
        yield Footer()

    def on_mount(self):
        self.query_one("#tree", Tree).border_title = "Components"
        self.query_one("#details-scroll").border_title = "Selected component"
        self.action_refresh()
        self.query_one(Tree).focus()
        self.set_interval(1, self.action_refresh)

    def action_compositions(self):
        self.query_one(Select).focus()
        self.query_one(Select).expanded = True

    def action_refresh(self):
        state = self.store.state()
        if state.get("generation") == self.state.get("generation"):
            return
        self.state = state
        records = state.get("compositions", {})
        # Full assemblies precede opaque standalone controls on first opening.
        ordered = sorted(
            records,
            key=lambda k: (
                -len(records[k]["revisions"][records[k]["current_revision"]]["nodes"]),
                composition_label(records[k]),
                k,
            ),
        )
        options = [(composition_label(records[k]), k) for k in ordered]
        selector = self.query_one(Select)
        with self.prevent(Select.Changed):
            selector.set_options(options)
            selector.disabled = not options
            if options:
                selector.value = (
                    self.composition_id if self.composition_id in records else ordered[0]
                )
        self.composition_id = selector.value if isinstance(selector.value, str) else None
        self._draw()

    def on_select_changed(self, event: Select.Changed):
        if event.value != event.select.value:
            return
        self.composition_id = event.value if isinstance(event.value, str) else None
        self._draw()
        self.query_one(Tree).focus()

    def _values(self, path):
        record = self.state["compositions"][self.composition_id]
        return {
            dimension: self.state["components"][key]["dimensions"][dimension]
            for dimension, key in record["current_assessments"].get(path, {}).items()
        }

    def _draw(self):
        tree = self.query_one(Tree)
        record = self.state.get("compositions", {}).get(self.composition_id)
        if not record:
            self.graph = None
            tree.reset("No recorded compositions")
            self.query_one("#details", Static).update(
                "Run or import a benchmark to populate this view."
            )
            return
        identity = (self.composition_id, record["current_revision"])
        changed = identity != self._shown
        self._shown = identity
        graph = self.graph = Assembly.read(record["revisions"][record["current_revision"]])
        selected = None if changed else tree.cursor_node.data if tree.cursor_node else None
        expanded = set()

        def remember(node):
            if node.is_expanded:
                expanded.add(node.data)
            for child in node.children:
                remember(child)

        if changed:
            self.roots = []
        else:
            remember(tree.root)
        root = self.roots[-1] if self.roots else self.graph.root
        tree.reset(self.graph.nodes[root].implementation, data=root)
        seen = set()
        selection = tree.root

        def populate(path, entry, role="", depth=0):
            nonlocal selection
            node = graph.nodes[path]
            # The path supplies the role; the full stable ID is in the details pane.
            family, component, source, variant = node.implementation.split(":")
            name = role or component.replace("_", " ").title()
            label = Text(f"{name} · {source}:{variant}")
            label.append(annotation(self._values(path), unknown=True), style="bold cyan")
            if path in seen:
                label.append(" ↗ shared", style="dim")
                entry.set_label(label)
                entry.allow_expand = False
                return
            entry.set_label(label)
            entry.allow_expand = bool(node.children or node.dependencies)
            seen.add(path)
            if path == selected:
                selection = entry
            for name, child in edges(node):
                added = entry.add(
                    name, data=child, expand=(depth < 1 if changed else child in expanded)
                )
                populate(child, added, name, depth + 1)

        populate(root, tree.root)
        tree.root.expand()
        self.call_after_refresh(self._restore_cursor, selection)

    def _restore_cursor(self, selection):
        self.query_one(Tree).move_cursor(selection)
        self._details()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        self._details()

    def _details(self):
        tree = self.query_one(Tree)
        path = tree.cursor_node.data if tree.cursor_node else None
        if not path or self.graph is None or path not in self.graph.nodes:
            return
        node = self.graph.nodes[path]
        lines = [Text(node.implementation, style="bold"), Text(path, style="dim")]
        if node.parameters.get("opaque") or node.parameters.get("opaque_server"):
            lines.append(Text("\nUpstream component; internal hierarchy not captured."))
        values = self._values(path)
        if not any(v["observed"] is not None for v in values.values()):
            lines.append(Text("\nNo matching measurement for this component yet."))
        record = self.state["compositions"][self.composition_id]
        for dimension, value in values.items():
            bound = value["bound"]
            lines.append(Text(f"\n{dimension}", style="bold cyan"))
            lines.append(Text(f"Observed    {metric(value['observed'], bound['unit'])}"))
            kind = "Minimum" if bound["direction"] == "lower" else "Maximum"
            lines.append(Text(f"{kind}     {metric(bound['value'], bound['unit'])}"))
            percent = (
                "—" if value["percent"] is None or value["issue"] else f"{value['percent']:.2f}%"
            )
            lines.append(
                Text(f"Efficiency  {percent}" + (" (estimated)" if value["estimated"] else ""))
            )
            if value["issue"] and value["observed"] is not None:
                lines.append(Text(value["issue"].capitalize(), style="yellow"))
            if value["observed"] is not None:
                key = record["current_assessments"][path][dimension]
                assessment = self.state["components"][key]
                profile = self.state["profiles"][assessment["profile"]]
                hardware = profile["hardware"]
                machine = " · ".join(
                    str(hardware[k]) for k in ("chip", "hostname", "machine") if hardware.get(k)
                )
                lines.append(
                    Text("\n" + (machine or "Hardware recorded in raw evidence"), style="bold")
                )
                lines.append(Text(workload_label(assessment["workload"])))
                for run in value["evidence"]:
                    evidence = self.state["runs"][run]
                    lines.append(Text(f"{evidence['benchmark']} · {evidence['completed_at']}"))
                    lines.append(
                        Text(str(self.store.root / "runs" / run / "run.json"), style="dim")
                    )
            if bound["missing"]:
                lines.append(Text("\nRequired theoretical inputs", style="bold"))
                lines.extend(Text(str(item)) for item in bound["missing"])
            if bound["assumptions"]:
                lines.append(Text("\nDerivation assumptions", style="bold"))
                lines.extend(Text(str(item)) for item in bound["assumptions"])
        self.query_one("#details", Static).update(Group(*lines))
        self.query_one("#details-scroll", VerticalScroll).scroll_home(animate=False)

    def action_focus_subsystem(self):
        tree = self.query_one(Tree)
        if tree.has_focus and tree.cursor_node and tree.cursor_node.data:
            self.roots.append(tree.cursor_node.data)
            self._draw()

    def action_back(self):
        if self.roots:
            self.roots.pop()
            self._draw()
