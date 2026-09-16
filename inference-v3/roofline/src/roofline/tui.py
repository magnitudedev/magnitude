"""Browse the shared formula performance model; execution evidence is a drill-down."""

import asyncio
import json

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import App
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Select, Static, Tree

from .query import Queries
from .store import Store


def number(value):
    if value is None:
        return "—"
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:.3g}{suffix}"
    return f"{value:.3g}"


def latency(seconds):
    return "—" if seconds is None else f"{seconds * 1e3:.3g} ms"


def cell(component):
    relation = component.relation
    unit = relation["metric"]["unit"]["name"] + "/s"
    attainment = relation["attainment"]
    if attainment:
        low, high = 100 * attainment["minimum"], 100 * attainment["maximum"]
        value = f"{low:.3g}%" if low == high else f"{low:.3g}–{high:.3g}%"
        return Text(f"{unit}  {value} of ceiling · {attainment['points']} points", style="cyan")
    if relation["predictions"]:
        return Text(
            f"{unit}  conditional prediction · {len(relation['predictions'])} updates",
            style="yellow",
        )
    if relation["points"]:
        reason = (
            "bound unresolved"
            if any(point["qualified"] for point in relation["points"])
            else "numerically unqualified"
        )
        return Text(f"{unit}  {len(relation['points'])} points · {reason}", style="dim")
    return Text(f"{unit}  awaiting evidence", style="dim")


class ComponentTree(Tree):
    def render_label(self, node, base_style, style):
        label = super().render_label(node, base_style, style)
        if node.data:
            label.append("  ").append_text(cell(node.data))
        return label


def detail_view(component, report):
    relation = component.relation
    metric = relation["metric"]
    heading = Text(f"{component.label} · {metric['unit']['name']}/s", style="bold")
    notes = [
        metric["meaning"],
        "Ceiling: useful quantity / strongest resource or certified sequential bound",
        "Each point is normalized at its own hardware and operating conditions.",
    ]
    table = Table(box=None, expand=True)
    for label in ("Evidence", "Measured", "Ceiling", "Attained"):
        table.add_column(label)
    for point in relation["points"][-12:]:
        bound = point["roofline"]
        attained = point["attainment"]
        label = point["hardware"] + " · " + point["boundary"].replace("-", " ")
        table.add_row(
            label,
            number(point["rate"]),
            number(bound["ceiling"]),
            f"{100 * attained:.1f}%" if attained is not None else point["correctness"],
        )
        if bound["kind"] == "conditional":
            notes.append("Conditional ceiling: " + "; ".join(bound["assumptions"]))
        if point["bound_status"] == "challenged":
            notes.append(
                "Observed rate exceeds the ceiling: its assumptions require investigation."
            )
        if bound["missing"]:
            missing = sorted(
                {parameter for item in bound["missing"] for parameter in item.get("capacities", ())}
            )
            if missing:
                notes.append("Unbound hardware parameters: " + ", ".join(missing))
    if not relation["points"]:
        notes.append("No direct measurement at this boundary.")
        for realization in relation["realizations"][-3:]:
            model = realization["model"]
            notes.append(f"Declared quantity: {number(model['quantity'])} {model['unit']['name']}")
            for demand in model["demands"]:
                notes.append(
                    f"{demand['resource']}: {number(demand['known_amount'])} "
                    f"{demand['unit']['name']} / capacity(H)"
                    + (" + unresolved demand" if demand["unresolved"] else "")
                )
    checked_behavior = [b for b in relation.get("behavior", ()) if b["validation"]]
    for hypothesis in checked_behavior[-2:]:
        notes.append(f"Hardware transfer hypothesis: {hypothesis['efficiency']:.1%} × ceiling(x,H)")
        for validation in hypothesis["validation"]:
            notes.append(
                f"Later hardware check: predicted {number(validation['predicted_rate'])}, "
                f"observed {number(validation['observed_rate'])} {metric['unit']['name']}/s"
            )
    for contribution in relation["contributions"][-3:]:
        fraction = contribution["parent_fraction"]
        notes.append(
            "In-parent native time: "
            + latency(contribution["inclusive_seconds"])
            + (f" · {fraction:.1%} of parent device time" if fraction is not None else "")
            + f" · {contribution['coverage']} capture"
        )
    for prediction in relation["predictions"][-3:]:
        notes.append(
            f"Conditional prediction: {number(prediction['rate'])} {metric['unit']['name']}/s "
            f"({latency(prediction['seconds'])}); " + "; ".join(prediction["assumptions"])
        )
        for validation in prediction["validation"]:
            notes.append("Later measurement error: " + latency(validation["error_seconds"]))
    for constraint in relation["constraints"][-3:]:
        notes.append(
            "Parent accounting: "
            + latency(constraint["remaining_seconds"])
            + " remains jointly constrained."
        )
    return Group(heading, table, Text("\n".join(dict.fromkeys(notes))))


def artifact_name(name, index):
    if name.endswith("/authored"):
        return f"Authored formula {index}"
    names = {
        "ops-measurement": "Formula measurement",
        "production-observations": "Production observations",
        "isolated-native-observation": "Native timing observation",
        "resource-characterization": "Hardware characterization",
        "component": "Shared input boundary",
        "prose.moby-dick": "Input corpus",
    }
    if name.endswith("/device-source"):
        return "Generated device code · " + name.split("/")[0]
    if name.endswith("/host-source"):
        return "Generated host code · " + name.split("/")[0]
    return names.get(name, f"Recorded artifact {index}")


class EvidenceScreen(ModalScreen):
    CSS = """
    EvidenceScreen { align: center middle; background: $background 70%; }
    #evidence-dialog { width: 95%; height: 90%; border: solid $primary; background: $surface; }
    #evidence-body { height: 1fr; padding: 1; }
    #evidence-close { dock: bottom; }
    """
    BINDINGS = [("escape", "dismiss", "Back"), ("n", "next_page", "Next page")]

    def __init__(self, workspace, measurements):
        super().__init__()
        self.workspace, self.measurements = workspace, measurements
        self.selected = None
        self.cursor = None

    def compose(self):
        with Vertical(id="evidence-dialog"):
            yield Select(
                [
                    (f"{m.target} · {m.created[:10]} {m.created[11:19]} · {m.correctness}", str(i))
                    for i, m in enumerate(self.measurements)
                ],
                allow_blank=False,
                id="evidence-run",
            )
            yield Select([], id="evidence-artifact", prompt="Choose named evidence")
            with VerticalScroll(id="evidence-body"):
                yield Static("", id="evidence-content", markup=False)
            yield Button("Close · Esc", id="evidence-close")

    @on(Select.Changed, "#evidence-run")
    def select_run(self, event):
        if not isinstance(event.value, str):
            return
        m = self.measurements[int(event.value)]
        self.selected = m
        choices = [("Measurement details", "measurement")]
        choices.extend(
            (artifact_name(name, i), blob) for i, (name, blob) in enumerate(m.artifacts.items(), 1)
        )
        select = self.query_one("#evidence-artifact", Select)
        select.set_options(choices)
        select.value = "measurement"
        self.show_evidence("measurement")

    @on(Select.Changed, "#evidence-artifact")
    def select_artifact(self, event):
        if isinstance(event.value, str):
            self.show_evidence(event.value)

    def show_evidence(self, value, *, cursor=None):
        if self.selected is None:
            return
        with Store(self.workspace, readonly=True) as store:
            query = Queries(store)
            data = (
                query.measurement(self.selected.measurement_id, cursor=cursor)
                if value == "measurement"
                else query.artifact(value, cursor=cursor)
            )
        self.cursor = data.get("cursor")
        content = json.dumps(data, indent=2) if value == "measurement" else data["content"]
        self.query_one("#evidence-content", Static).update(
            content + ("\n\n[n] Next page" if self.cursor else "")
        )

    def action_next_page(self):
        if self.cursor:
            self.show_evidence(
                self.query_one("#evidence-artifact", Select).value, cursor=self.cursor
            )

    @on(Button.Pressed, "#evidence-close")
    def close_evidence(self):
        self.dismiss()


class RooflineApp(App):
    CSS = """
    #model { height: 3; }
    #context-note { height: auto; max-height: 2; color: $text-muted; }
    #components { height: 3fr; min-height: 6; border: solid $primary; }
    #details-pane { height: 2fr; border: solid $secondary; padding: 0 1; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("m", "choose", "Model"),
        ("r", "refresh", "Refresh"),
        ("e", "evidence", "Evidence"),
        ("enter", "focus_subtree", "Focus"),
        ("escape", "back", "Back"),
    ]

    def __init__(self, configuration):
        super().__init__()
        self.configuration = configuration
        self.selected_model = None
        self.report = None
        self.focus_path = ""
        self.navigation = []
        self.updating = False

    def compose(self):
        yield Select([], id="model", prompt="Choose model")
        yield Static("", id="context-note", markup=False)
        yield ComponentTree("Model", id="components")
        with VerticalScroll(id="details-pane"):
            yield Static("", id="details", markup=False)
        yield Footer()

    def on_mount(self):
        self.action_refresh()
        self.query_one(ComponentTree).focus()

    @on(Select.Changed, "#model")
    def change_model(self, event):
        if self.updating or not isinstance(event.value, str) or event.value == self.selected_model:
            return
        self.selected_model, self.focus_path, self.navigation = event.value, "", []
        self.populate(reload=True)

    def action_refresh(self):
        with Store(self.configuration.workspace, readonly=True) as store:
            models = set(self.configuration.models)
            models.update(store.model_names())
        self.updating = True
        select = self.query_one("#model", Select)
        select.set_options([(m, m) for m in sorted(models)])
        if self.selected_model not in models:
            self.selected_model = min(models) if models else None
        if self.selected_model:
            select.value = self.selected_model
        self.updating = False
        self.populate(reload=True)

    @work(exclusive=True, group="model-load")
    async def load_model(self, model):
        def read():
            with Store(self.configuration.workspace, readonly=True) as store:
                return Queries(store, self.configuration.models).tree(model)

        try:
            report = await asyncio.to_thread(read)
        except Exception as error:
            if model == self.selected_model:
                self.query_one("#context-note", Static).update(f"Cannot load model: {error}")
            return
        if model == self.selected_model:
            self.report = report
            self.populate()

    def populate(self, *, reload=False):
        tree = self.query_one(ComponentTree)
        if self.selected_model is not None and (
            reload or self.report is None or self.report.model != self.selected_model
        ):
            if self.report is None or self.report.model != self.selected_model:
                tree.clear()
                tree.root.data = None
                tree.root.set_label(self.selected_model + " · loading")
                self.query_one("#details", Static).update("Loading saved formula performance…")
            self.query_one("#context-note", Static).update("Loading model evidence…")
            self.load_model(self.selected_model)
            return
        expanded = set()
        selected = tree.cursor_node.data.key if tree.cursor_node and tree.cursor_node.data else ""

        def remember(node):
            if node.data and node.is_expanded:
                expanded.add(node.data.key)
            for child in node.children:
                remember(child)

        remember(tree.root)
        tree.clear()
        if self.selected_model is None:
            self.query_one("#details", Static).update("No models configured or measured.")
            return
        self.query_one("#context-note", Static).update(
            self.report.notice
            or "Formula units · hardware-normalized attainment across all recorded conditions"
        )
        if not self.report.nodes:
            tree.root.set_label(self.selected_model)
            tree.root.data = None
            self.query_one("#details", Static).update(self.report.notice)
            return
        focus = self.report.nodes.get(self.focus_path, self.report.nodes[""])
        tree.root.set_label(focus.label)
        tree.root.data = focus
        nodes = {focus.key: tree.root}
        pending = dict(self.report.nodes)
        while pending:
            added = False
            for key, component in list(pending.items()):
                if key == focus.key:
                    del pending[key]
                elif component.parent in nodes:
                    nodes[key] = nodes[component.parent].add(component.label, component)
                    del pending[key]
                    added = True
            if not added:
                break
        for key, node in nodes.items():
            node.allow_expand = bool(node.children)
            if key in expanded or node is tree.root:
                node.expand()
        self.call_after_refresh(tree.move_cursor, nodes.get(selected, tree.root))
        self.query_one("#details", Static).update(
            detail_view(nodes.get(selected, tree.root).data, self.report)
        )

    @on(Tree.NodeHighlighted)
    def highlighted(self, event):
        if event.node.data and self.report:
            self.query_one("#details", Static).update(detail_view(event.node.data, self.report))
            self.query_one("#details-pane", VerticalScroll).scroll_home(animate=False)

    def action_choose(self):
        self.query_one("#model", Select).focus()
        self.query_one("#model", Select).expanded = True

    def action_focus_subtree(self):
        node = self.query_one(ComponentTree).cursor_node
        if node and node.data and node.data.key != self.focus_path:
            self.navigation.append(self.focus_path)
            self.focus_path = node.data.key
            self.populate()

    def action_back(self):
        if self.navigation:
            self.focus_path = self.navigation.pop()
            self.populate()

    def action_evidence(self):
        node = self.query_one(ComponentTree).cursor_node
        if not node or not node.data:
            return
        references = {ref for point in node.data.relation["points"] for ref in point["evidence"]}
        with Store(self.configuration.workspace, readonly=True) as store:
            measurements = [
                store.measurement(r["measurement_id"])
                for r in store.records("measurement")
                if r["measurement_id"] in references
                or r.get("artifacts", {}).get("formula-performance") in references
            ]
        if measurements:
            self.push_screen(EvidenceScreen(self.configuration.workspace, measurements))
        else:
            self.notify("No measurement payload at this boundary.")
