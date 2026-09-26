"""Formula hierarchy and measurements: a client of Lab, never a benchmark runner."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from uuid import uuid4

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Static, Tree

from ..formula import FormulaHandle
from .records import CeilingKind, Direction, History
from .archive import RecordedFormula
from .worker import FormulaState, Lab, Status


class ConfigurationApp(App):
    """The selection data is the prepared object, never a formula-path string."""

    TITLE = "Choose performance configuration"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, configurations, *, recorded=False):
        super().__init__()
        self.configurations = configurations
        self.recorded = recorded

    def compose(self) -> ComposeResult:
        yield Tree("Recorded configurations" if self.recorded else "Performance configurations", id="configurations")
        yield Footer()

    def on_mount(self):
        tree = self.query_one(Tree)
        for configuration in self.configurations:
            name = configuration.label
            if self.recorded:
                name += f" · {configuration.created.isoformat()} · {configuration.device}"
            tree.root.add_leaf(Text(name), data=configuration)
        if not self.configurations:
            tree.root.add_leaf("No configurations have been recorded yet.")
        tree.root.expand()
        tree.focus()

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        if event.node.data is not None:
            self.exit(event.node.data)


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 0.001:
        return f"{seconds * 1e6:.4g} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.4g} ms"
    return f"{seconds:.4g} s"


def label(state: FormulaState | RecordedState) -> Text:
    latest = state.history.latest_success if state.history is not None else None
    timing = duration(latest.median_seconds) if latest is not None else "—"
    pending = " · source edited" if state.source_pending else ""
    model = ""
    if latest is not None and latest.performance is not None:
        point = latest.performance["points"][-1]
        unit = latest.performance["metric"]["unit"]["name"] + "/s"
        model = f" · {point['rate']:.4g} {unit}" if point["rate"] is not None else ""
        if point["attainment"] is not None:
            model += f" · {point['attainment']:.1%} of ceiling"

    return Text(f"{state.target.definition.id} · {state.status.value}{pending} · isolated {timing}{model}")


def roofline_label(roofline, seconds):
    if roofline is None:
        return " · roofline not recorded"
    floor = roofline.seconds
    ratio = f"{100 * floor / seconds:.3g}%" if seconds else "—"
    gap = duration(max(0, seconds - floor)) if seconds is not None else "—"
    violation = " · exceeds reference" if seconds and floor > seconds else ""
    return f" · model {duration(floor)} · {ratio} · gap {gap} · {roofline.bottleneck}{violation}"


def details(state: FormulaState | RecordedState) -> Text:
    target = state.target
    lines = [f"{target.definition.id} · formula v{target.definition.version}",
             f"Status: {state.status.value}",
             "Scope: isolated production operation, including recurring I/O and completion.",
             "Parent latency is measured directly; child timings are not summed."]
    if state.status == Status.RECORDED:
        lines.append("Historical evidence loaded; current executable freshness has not been checked.")
    if isinstance(target, RecordedFormula) and not target.complete:
        lines.append("This pruned occurrence is incomplete and cannot identify a comparable full-formula measurement.")
    if state.source_pending:
        lines.append("A dependency's source file changed. Remeasure to refresh its definitions; this notice has not invalidated unrelated symbols.")
    dependencies = target.dependencies if isinstance(target, RecordedFormula) else state.dependencies
    if dependencies:
        lines.append("Input-producing occurrences: " + ", ".join(map(str, dependencies)))
    if state.phase is not None:
        lines.append(f"Running: {state.phase.value} ({state.completed}/{state.total})")
    if state.error:
        lines.append(f"Error: {state.error}")
    history = state.history
    if history is not None:
        lines += ["", "Comparable conditions", f"Device: {history.series.device}",
                  f"Precision: {history.series.precision}", f"Fixture: {history.series.fixture}",
                  f"Series: {history.series.identity}"]
        for source in history.series.sources:
            lines.append(f"Source: {source.source.kind.value} · {source.source.location or source.source.identity}"
                         f" · {source.cache.value}")
        if history.latest is not None:
            lines.append(f"Latest attempt: {history.latest.outcome.value} · {history.latest.created.isoformat()}")
        observed = history.latest_success
        if observed is not None:
            previous = next((item for item in history.observations
                             if item.identity != observed.identity and item.median_seconds is not None), None)
            lines += ["", f"Last successful: {duration(observed.median_seconds)}",
                      f"Previous successful: {duration(previous.median_seconds) if previous else '—'}",
                      f"Best recorded: {duration(history.best.median_seconds) if history.best else '—'}"]
            samples = tuple(sample.elapsed_ns / 1e9 for sample in observed.samples)
            center = median(samples)
            deviation = median(abs(value - center) for value in samples)
            lines.append(f"Samples: {len(samples)} · range {duration(min(samples))}–{duration(max(samples))}"
                         f" · median absolute deviation {duration(deviation)}")
            native = tuple(sample.kernels for sample in observed.samples if sample.kernels is not None)
            if len(native) == len(observed.samples):
                clocks = sorted({sample.clock for sample in native})
                lines.append("Native kernel clock: " + ", ".join(clocks))
            lines += ["", "Observed metrics"]
            if observed.roofline is not None:
                model = observed.roofline
                lines += ["", "Empirical resource reference",
                          f"Ideal reference time: {duration(model.seconds)} · limiting resource: {model.bottleneck}",
                          "Gap is relative to this model, not guaranteed recoverable wall time.",
                          f"Model: {model.revision} · calibration: {model.characterization}"]
                for limit in model.limits:
                    demand = limit.demand
                    amount = (f"{demand.lower:g}" if demand.lower == demand.upper else
                              f"{demand.lower:g}–{demand.upper:g}")
                    lines += [f"{demand.resource.value}/{demand.dtype.value}: {amount} {demand.unit.name}",
                              f"  / {limit.rate:.6g} {demand.unit.name}/s = {duration(limit.lower_seconds)} reference time",
                              f"  probe measurement: {limit.measurement}", f"  {demand.basis}"]
                lines.extend(f"Assumption: {assumption}" for assumption in model.assumptions)
            for metric in observed.metrics:
                lines.append(f"{metric.name}: {metric.value:.6g} {metric.unit.name}")
                lines.append(f"  {metric.basis}")
                for ceiling in observed.ceilings:
                    if ceiling.metric != metric.name:
                        continue
                    if ceiling.direction == Direction.HIGHER:
                        ratio = metric.value / ceiling.value if ceiling.value else None
                    else:
                        ratio = ceiling.value / metric.value if metric.value else None
                    kind = "empirical reference" if ceiling.kind == CeilingKind.EMPIRICAL else "theoretical bound"
                    efficiency = f" · reference ratio {100 * ratio:.3g}%" if ratio is not None else ""
                    lines.append(f"  {kind}: {ceiling.value:.6g} {ceiling.unit.name}{efficiency}")
                    lines.append(f"  {ceiling.provenance}; assumptions: {', '.join(ceiling.assumptions)}")
            if not observed.ceilings and observed.roofline is None:
                lines.append("No applicable characterized/modelled bound is recorded; efficiency is unavailable.")
            for missing in observed.unavailable:
                lines.append(f"Unavailable {missing.name}: {missing.reason}")
            if observed.implementation is not None:
                lines += ["", f"Implementation: {observed.implementation.fingerprint}",
                          f"Compiler: {observed.implementation.compiler}"]
            lines += ["", "Measurement history (newest first)"]
            for item in history.observations:
                lines.append(f"{item.created.isoformat()} · {item.outcome.value} · {duration(item.median_seconds)}")
    if state.job is not None:
        job = state.job
        lines += ["", "Development turnaround (not operation latency)",
                  f"Queue: {duration(job.queue_ns / 1e9)}",
                  f"Active through measurement publication: {duration(job.active_ns / 1e9)}",
                  f"Measurement publication: {duration(job.publication_ns / 1e9)}"]
        lines.extend(f"{phase.phase.value}: {duration(phase.elapsed_ns / 1e9)}" for phase in job.phases)
        if job.active_ns >= 5_000_000_000:
            lines.append("Over the <5 s small-operation iteration target. See phase attribution above.")
    if state.visibility is not None:
        elapsed = state.visibility.request_to_visible_ns / 1e9
        lines.append(f"Request → persisted and displayed: {duration(elapsed)}")
        if elapsed >= 5:
            lines.append("Full visible turnaround exceeds the small-operation target.")
    return Text("\n".join(lines))


class PerformanceApp(App):
    TITLE = "Formula performance"
    CSS = """
    #main { height: 1fr; }
    #formulas { width: 2fr; border: solid $primary; }
    #detail-scroll { width: 3fr; border: solid $primary; }
    #details { height: auto; padding: 0 1; }
    #scope { height: auto; max-height: 8; padding: 0 1; }
    #status { height: 2; }
    #device-evidence { height: auto; max-height: 12; overflow-y: auto; }
    """
    BINDINGS = [
        ("m", "measure", "Measure selected"),
        ("s", "subtree", "Preview subtree"),
        ("a", "affected", "Preview affected"),
        ("enter", "confirm", "Run previewed scope"),
        ("escape", "dismiss", "Dismiss scope"),
        ("c", "cancel", "Cancel work"),
        ("r", "history", "Reload history"),
        ("o", "configuration", "Configuration"),
        ("p", "characterize", "Device evidence"),
        ("P", "recharacterize", "Remeasure device"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, lab: Lab, *, configuration_label: str | None = None, can_choose: bool = False):
        super().__init__()
        self.lab = lab
        self.can_choose = can_choose
        if configuration_label:
            self.sub_title = configuration_label
        self._revision = -1
        self._formula_nodes = {}
        self._selected: FormulaHandle | None = None
        self._scope: tuple[FormulaHandle, ...] = ()
        self._inspections = []
        self._client = f"tui:{uuid4()}"
        self._acknowledged = set()
        self._request_error = None

    def action_configuration(self):
        if self.can_choose:
            self.exit("choose")

    def action_characterize(self):
        self._characterize(False)

    def action_recharacterize(self):
        self._characterize(True)

    def _characterize(self, refresh):
        self._request_error = None
        try:
            self._inspections.append(self.lab.characterize(refresh=refresh))
        except RuntimeError as error:
            self._request_error = str(error)
        self._revision = -1

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            yield Tree("Formula composition", id="formulas")
            with VerticalScroll(id="detail-scroll"):
                yield Static("Select a formula occurrence.", id="details", markup=False)
        yield Static("", id="scope", markup=False)
        yield Static("Opening persistent device worker…", id="status", markup=False)
        yield Static("Device evidence loads on the first measurement; p prepares it ahead of time.",
                     id="device-evidence", markup=False)
        yield Footer()

    def on_mount(self):
        tree = self.query_one("#formulas", Tree)
        for target in self.lab.formulas:
            parent = self._formula_nodes.get(target.parent, tree.root)
            self._formula_nodes[target] = parent.add(Text(target.definition.id), data=target, expand=False)
        tree.root.expand()
        tree.focus()
        self.set_interval(0.1, self._refresh)
        self._refresh()

    def _refresh(self):
        pending = []
        for future in self._inspections:
            if not future.done():
                pending.append(future)
            elif future.exception() is not None:
                self._request_error = f"Request failed: {future.exception()}"
                self._revision = -1
        self._inspections = pending
        snapshot = self.lab.snapshot()
        if snapshot.revision == self._revision:
            return
        self._revision = snapshot.revision
        if snapshot.characterizing:
            self.query_one("#device-evidence", Static).update("Device characterization queued/running; c cancels.")
        elif snapshot.characterization is not None:
            profile = snapshot.characterization
            lines = [f"Device empirical evidence · {profile.created.isoformat()} · not physical peak specifications"]
            for rate in profile.rates:
                lines.append(f"{rate.resource.value} {rate.dtype.value}: {rate.value:.5g} {rate.unit.name}"
                             f" · working set {rate.working_set_bytes} B · measurement {rate.measurement}")
                lines.extend(f"  {condition}" for condition in rate.conditions)
            lines.extend(f"Unavailable {item.name}: {item.reason}" for item in profile.unavailable)
            self.query_one("#device-evidence", Static).update(Text("\n".join(lines)))
        else:
            self.query_one("#device-evidence", Static).update("Device evidence loads on the first measurement; p prepares it ahead of time.")
        for state in snapshot.states:
            self._formula_nodes[state.target].set_label(label(state))
            if state.target == self._selected:
                self.query_one("#details", Static).update(details(state))
            if (state.target == self._selected and state.job is not None and
                    state.job.identity not in self._acknowledged):
                self._acknowledged.add(state.job.identity)
                self.call_after_refresh(self._acknowledge, state.job)
        busy = sum(state.status in (Status.QUEUED, Status.RUNNING) for state in snapshot.states)
        text = snapshot.error or self._request_error or snapshot.source_error or ("Closed" if snapshot.closed else
                                 f"{busy} queued/running · display refresh never runs measurements" if snapshot.ready else
                                 "Preparing device and fixture…")
        self.query_one("#status", Static).update(Text(text))

    def _acknowledge(self, result):
        try:
            self.lab.acknowledge(result, client=self._client)
        except (RuntimeError, ValueError):
            # A closed worker or old historical record has no live turnaround
            # receipt. Do not fabricate one from its UTC timestamp.
            pass

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        target = event.node.data
        if not isinstance(target, FormulaHandle):
            return
        self._selected = target
        self._revision = -1
        self._refresh()
        if not self.lab.snapshot().closed:
            try:
                self._inspections.append(self.lab.inspect(target))
            except RuntimeError as error:
                self._request_error = str(error)
                self._revision = -1

    def action_measure(self):
        if self._selected is not None:
            self._request((self._selected,))

    def action_affected(self):
        if self._selected is not None:
            self._scope = self.lab.formulas.affected((self._selected,))
            names = ", ".join(target.definition.id for target in self._scope)
            self.query_one("#scope", Static).update(Text(
                f"{len(self._scope)} dependent occurrences: {names}\nEnter to measure this scope; Escape to dismiss.",
            ))

    def action_subtree(self):
        if self._selected is not None:
            self._scope = self.lab.subtree(self._selected)
            self.query_one("#scope", Static).update(Text(
                f"{len(self._scope)} independent formula measurements, parent first. "
                "The parent retains production fusion; children compile in isolation. Times are not additive.\n"
                "Enter to measure this subtree; Escape to dismiss.",
            ))

    def _request(self, targets):
        self._request_error = None
        try:
            for target in targets:
                self.lab.measure(target)
        except RuntimeError as error:
            self._request_error = str(error)
            self._revision = -1

    def action_confirm(self):
        self._request(self._scope)
        self.action_dismiss()

    def action_dismiss(self):
        self._scope = ()
        self.query_one("#scope", Static).update("")

    def action_cancel(self):
        self.lab.cancel()

    def action_history(self):
        if self._selected is not None:
            try:
                self._inspections.append(self.lab.inspect(self._selected))
            except RuntimeError as error:
                self._request_error = str(error)
                self._revision = -1


@dataclass(frozen=True)
class RecordedState:
    target: RecordedFormula
    history: History | None
    status: Status = Status.RECORDED
    source_pending: bool = False
    phase: None = None
    completed: int = 0
    total: int = 1
    error: None = None
    job: None = None
    visibility: None = None


class RecordedApp(App):
    """Read-only composition and comparable history, with no executable handles."""
    TITLE = "Recorded formula performance"
    CSS = PerformanceApp.CSS
    BINDINGS = [("r", "history", "Refresh history"), ("o", "configuration", "Configuration"),
                ("q", "quit", "Quit")]

    def __init__(self, store, configuration):
        super().__init__()
        self.store, self.configuration = store, configuration
        self.sub_title = configuration.label
        self._selected = None

    def compose(self):
        with Horizontal(id="main"):
            yield Tree("Recorded formula composition", id="formulas")
            with VerticalScroll(id="detail-scroll"):
                yield Static("Select a recorded formula.", id="details", markup=False)
        yield Static("Historical evidence only. Open a prepared configuration to remeasure.", id="status", markup=False)
        yield Footer()

    def on_mount(self):
        tree = self.query_one("#formulas", Tree)
        nodes = {}
        overview = self.store.recorded_overview(self.configuration)
        rooflines = self.store.recorded_rooflines(self.configuration)
        for target in self.configuration.formulas:
            parent = nodes.get(target.parent, tree.root)
            seconds, outcome = overview.get(target.occurrence, (None, "unmeasured"))
            nodes[target.occurrence] = parent.add(Text(
                f"{target.definition.id} · recorded {outcome} · isolated {duration(seconds)}"
                + roofline_label(rooflines.get(target.occurrence), seconds)), data=target)
        tree.root.expand()
        tree.focus()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted):
        if isinstance(event.node.data, RecordedFormula):
            self._selected = event.node.data
            self.action_history()

    def action_history(self):
        target = self._selected
        if target is None:
            return
        history = self.store.recorded_history(self.configuration, target.occurrence)
        state = RecordedState(target, history)
        text = details(state)
        if history is None:
            text.append("\nThis occurrence has no measured input conditions in this recording.")
        self.query_one("#details", Static).update(text)

    def action_configuration(self):
        self.exit("choose")
