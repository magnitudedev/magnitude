"""Model-first browsing of published evidence. No execution dependency or callbacks."""

from __future__ import annotations

from rich.text import Text
from textual.app import App
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Static, Tree

from .evidence import Model, RunEvidence
from .store import ObservationStore


def run_details(store: ObservationStore, run: RunEvidence) -> Text:
    context = run.context
    text = Text(
        f"{context.model.label}\n{run.created.isoformat()}\n"
        f"Engine: {context.engine}\nHost: {context.host or 'unrecorded'}\n"
        f"Hardware: {context.hardware}\n"
        f"Implementation: {context.implementation}\nArtifact: {context.artifact}\n"
        f"Numerical contract: {context.numerical_contract}\n"
        f"Scope: {run.scope.kind} {run.scope.formula.id if run.scope.formula else ''}\n"
        f"Status: {run.status}; correctness: {run.correctness}\n"
        f"Workload: {context.workload.identity}\n"
        f"Conditions: {context.conditions}\n\n"
    )
    for metric in run.metrics:
        text.append(
            f"{metric.name}: {metric.median:.6g} {metric.unit.name}\n"
            f"  {metric.boundary}: {metric.basis}\n  Samples: {metric.samples}\n"
        )
    for identity in run.measurements:
        measurement = store.measurement(identity)
        text.append(f"\n{measurement.series.formula.id} · isolated measurement\n")
        if measurement.observed_seconds is not None and not measurement.checked:
            text.append(
                f"Unqualified timing: {measurement.observed_seconds:.6g} s "
                "· numerical check failed\n"
            )
        for metric in measurement.metrics:
            text.append(f"{metric.name}: {metric.value:.6g} {metric.unit.name} · {metric.basis}\n")
        if measurement.roofline:
            text.append(
                "Empirical resource model (not a proven hardware ceiling): "
                f"{measurement.roofline.seconds:.6g} seconds\n"
            )
        for ceiling in measurement.ceilings:
            text.append(
                f"{ceiling.kind}: {ceiling.metric} {ceiling.value:.6g} {ceiling.unit.name}\n"
                f"  Assumptions: {ceiling.assumptions}\n"
            )
        if not any(c.unit.dimension == "time" for c in measurement.ceilings):
            text.append("Qualified latency floor: unavailable\n")
        for unavailable in measurement.unavailable:
            text.append(f"Unavailable {unavailable.name}: {unavailable.reason}\n")
        for name, digest in measurement.artifacts.items():
            text.append(f"Artifact {name}: {digest}\n")
    for sample in run.observations:
        native = sample.kernels
        if native is not None:
            text.append(f"\nNative attribution: {native.attribution}\n")
            owners = {}
            for activity in native.activities:
                owners.setdefault((activity.graph, activity.owner), []).append(activity)
            from ..runtime.observation import KernelObservation

            for (graph, owner), activities in owners.items():
                busy = KernelObservation(native.clock, tuple(activities)).busy_ns
                text.append(
                    f"  Graph {(graph or 'unknown')[:12]} · formula occurrence {owner}: "
                    f"{busy / 1e9 if busy is not None else 'unavailable'} s native union\n"
                )
            text.append("Component unions may overlap; their sum is not parent wall time.\n")
    if not run.measurements:
        text.append("\nFormula ceilings: unavailable for this enclosing observation\n")
    for mapping in run.mappings:
        text.append(
            f"\nReference mapping: {mapping.region} · {mapping.relationship}\n"
            f"Differences: {mapping.differences}\nEvidence: {mapping.evidence}\n"
        )
    for issue in run.unavailable:
        text.append(f"\nUnavailable: {issue}")
    return text


def performance_model(store, model):
    from formula_performance.evidence import evaluate
    from formula_performance.records import Publication

    publications = []
    for run in store.runs(model.identity):
        artifact = run.attachments.get("formula-performance")
        if artifact:
            publications.append(Publication.model_validate_json(store.artifact(artifact)))
        for measurement_id in run.measurements:
            measurement = store.measurement(measurement_id)
            artifact = measurement.artifacts.get("formula-performance")
            if artifact:
                publications.append(Publication.model_validate_json(store.artifact(artifact)))
    return evaluate(publications)


def component_details(relation):
    text = Text(relation["label"] + " · " + relation["metric"]["unit"]["name"] + "/s\n")
    text.append("One hardware-parameterized formula relation across recorded conditions.\n")
    for point in relation["points"]:
        bound = point["roofline"]
        text.append(f"\n{point['boundary']}: {point['rate']} · ceiling {bound['ceiling']} · {bound['kind']}\n")
        if point["attainment"] is not None:
            text.append(f"{point['attainment']:.1%} attained\n")
    for prediction in relation["predictions"]:
        text.append(f"Conditional prediction: {prediction['seconds']:.6g} s\n")
    return text


def model_details(store, model):
    report = performance_model(store, model)
    root = report["components"].get("")
    return component_details(root) if root else Text(model.label + " · no analytical publication")


class ModelApp(App):
    TITLE = "Model performance"
    BINDINGS = [("r", "refresh", "Refresh evidence"), ("q", "quit", "Quit")]
    CSS = "#models { width: 55%; } #detail { width: 45%; } #details { padding: 1; }"

    def __init__(self, store):
        super().__init__()
        self.store = store

    def compose(self):
        with Horizontal():
            yield Tree("Models", id="models")
            with VerticalScroll(id="detail"):
                yield Static("Choose a model", id="details", markup=False)
        yield Footer()

    def on_mount(self):
        self.action_refresh()

    def action_refresh(self):
        tree = self.query_one("#models", Tree)
        tree.clear()
        for model in self.store.models():
            report = performance_model(self.store, model)
            root = tree.root.add(model.label, data=model)
            nodes = {"": root}
            for key, relation in sorted(report["components"].items(), key=lambda item: item[0].count("/")):
                label = relation["label"] + " · " + relation["metric"]["unit"]["name"] + "/s"
                attainment = relation["attainment"]
                if attainment:
                    label += f" · {attainment['minimum']:.0%}–{attainment['maximum']:.0%}"
                if key:
                    nodes[key] = nodes[relation["parent"]].add(label, data=relation)
                else:
                    root.data = relation
            root.expand()
        tree.root.expand()
        tree.focus()

    def on_tree_node_highlighted(self, event):
        value = event.node.data
        if isinstance(value, dict):
            self.query_one("#details", Static).update(component_details(value))
        elif isinstance(value, Model):
            self.query_one("#details", Static).update(model_details(self.store, value))
