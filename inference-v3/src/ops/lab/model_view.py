"""Model-first browsing of published evidence. No execution dependency or callbacks."""

from __future__ import annotations

from rich.text import Text
from textual.app import App
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Static, Tree

from .evidence import Model, RunEvidence
from .records import Outcome
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


def model_details(store: ObservationStore, model: Model) -> Text:
    runs = store.runs(model.identity)
    groups = {}
    for run in runs:
        groups.setdefault(run.comparison_key, []).append(run)
    text = Text(
        f"{model.label}\n{len(runs)} recorded executions; {len(groups)} condition series\n"
        "Hardware, workload, precision, and protocol remain separate.\n"
        "Unmeasured conditions have no inferred performance.\n\n"
    )
    for history in groups.values():
        latest = history[0]
        text.append(
            f"{latest.context.engine} · {latest.scope.kind} "
            f"{latest.scope.formula.id if latest.scope.formula else ''}\n"
            f"  Host: {latest.context.host or 'unrecorded'} · Hardware: {latest.context.hardware}\n"
            f"  Latest measured: {latest.created.isoformat()} · {latest.status} · "
            f"{latest.context.implementation[:16]}\n"
        )
        for metric in latest.metrics:
            text.append(f"  {metric.name}: {metric.median:.6g} {metric.unit.name}\n")
        for identity in latest.measurements:
            current = store.measurement(identity)
            compatible = [store.measurement(i) for run in history for i in run.measurements]
            qualified = [
                m
                for m in compatible
                if m.series == current.series
                and m.outcome == Outcome.COMPLETE
                and m.checked
                and m.median_seconds is not None
            ]
            if current.observed_seconds is not None:
                text.append(f"  Isolated: {current.observed_seconds:.6g}s")
                if not current.checked:
                    text.append(" · numerically unqualified")
            if qualified:
                best = min(
                    qualified,
                    key=lambda m: (
                        m.median_seconds if m.median_seconds is not None else float("inf")
                    ),
                )
                text.append(f" · best correct: {best.median_seconds:.6g}s")
            text.append("\n")
        text.append(f"  History: {len(history)} executions; ceiling details on selection\n\n")
    text.append(
        "Implementation freshness beyond recorded revisions: unknown\n"
        "The agent publishes measurements; this view never starts work."
    )
    return text


class ModelApp(App):
    TITLE = "Model performance"
    BINDINGS = [("r", "refresh", "Refresh evidence"), ("q", "quit", "Quit")]
    CSS = """
    #models { width: 40%; }
    #detail { width: 60%; }
    #details { padding: 1 2; }
    """

    def __init__(self, store: ObservationStore):
        super().__init__()
        self.store = store

    def compose(self):
        with Horizontal():
            yield Tree("Models", id="models")
            with VerticalScroll(id="detail"):
                yield Static(
                    "Select a model. No measurements run when browsing.", id="details", markup=False
                )
        yield Footer()

    def on_mount(self):
        self.action_refresh()

    def action_refresh(self):
        tree = self.query_one("#models", Tree)
        tree.clear()
        for model in self.store.models():
            node = tree.root.add(Text(model.label), data=model)
            for run in self.store.runs(model.identity):
                label = (
                    f"{run.context.engine} · {run.scope.kind} · {run.created:%m-%d %H:%M} "
                    f"· {run.context.hardware[:16]} · {run.status}"
                )
                entry = node.add(Text(label), data=run)
                if run.configuration is not None:
                    configuration = next(
                        c for c in self.store.configurations() if c.identity == run.configuration
                    )
                    nodes = {}
                    for formula in configuration.formulas:
                        parent = nodes.get(formula.parent, entry)
                        nodes[formula.occurrence] = parent.add(
                            Text(formula.definition.id), data=(run, configuration, formula)
                        )
        tree.root.expand()
        tree.focus()
        if not self.store.models():
            self.query_one("#details", Static).update(
                "No model evidence has been published. The agent must run or import measurements."
            )

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        value = event.node.data
        if isinstance(value, Model):
            detail = model_details(self.store, value)
        elif isinstance(value, RunEvidence):
            detail = run_details(self.store, value)
        elif isinstance(value, tuple):
            from .tui import RecordedState, details

            run, configuration, formula = value
            from .records import History

            linked = [self.store.measurement(i) for i in run.measurements]
            exact = [
                m
                for m in linked
                if m.series.formula == formula.definition
                and m.series.semantics == formula.semantics
                and run.scope.occurrence == formula.occurrence
            ]
            history = None
            if exact:
                measurement = exact[0]
                success = measurement if measurement.outcome == Outcome.COMPLETE else None
                history = History(
                    series=measurement.series,
                    latest=measurement,
                    latest_success=success,
                    best=success,
                    observations=(measurement,),
                )
            detail = details(RecordedState(formula, history))
            detail.append(
                "\nEvidence from this selected run only; other runs remain in model history."
            )
            from ..runtime.observation import KernelObservation

            for sample in run.observations:
                native = sample.kernels
                if native is None or native.attribution != "compiled-order-and-symbols":
                    detail.append("\nIn-parent native attribution unavailable for this sample.")
                    continue
                activities = tuple(
                    a for a in native.activities if a.graph == run.attachments.get("compiled_graph")
                )
                inclusive = tuple(a for a in activities if formula.occurrence in a.origins)
                exclusive = tuple(a for a in activities if formula.occurrence == a.owner)
                for label, events in (
                    ("Inclusive (may share fused work)", inclusive),
                    ("Exclusively attributed", exclusive),
                ):
                    busy = KernelObservation(native.clock, events).busy_ns
                    detail.append(
                        f"\n{label}: {busy / 1e9 if busy is not None else 'unknown'} s GPU union"
                    )
            for external in self.store.runs(run.context.model.identity):
                for mapping in external.mappings:
                    if any(
                        scope.formula == formula.definition and scope.semantics == formula.semantics
                        for scope in mapping.scopes
                    ):
                        detail.append(
                            f"\n\nReference: {external.context.engine} · {mapping.region}"
                            f" · {mapping.relationship}\nHost: {external.context.host}"
                            f" · Hardware: {external.context.hardware}"
                            f"\nDifferences: {mapping.differences}"
                        )
                        detail.append(
                            "\nEnclosing source observations (not an inferred formula share):"
                        )
                        for metric in external.metrics:
                            detail.append(
                                f"\n  {metric.name}: {metric.median:.6g} {metric.unit.name}"
                                f" · {metric.boundary}"
                            )
        else:
            return
        self.query_one("#details", Static).update(detail)
