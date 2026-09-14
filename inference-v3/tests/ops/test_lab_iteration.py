"""One final-gate edit/compile/check/persist/render campaign; never a routine unit sweep."""

import asyncio
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch
from textual.widgets import Tree

import ops
from engine import DevicePlan
from ops.lab import Fixture, Lab
from ops.lab.records import Outcome, Phase
from ops.lab.refresh import ModuleSource, OperationSources
from ops.lab.store import ObservationStore
from ops.lab.tui import PerformanceApp
from ops.lab.worker import Status


@ops.formula(id="qualification.edit.square")
def edited(value):
    return value * value


@ops.formula(id="qualification.edit.sibling")
def sibling(value):
    return value + value


@ops.formula(id="qualification.edit.parent")
def parent(left, right):
    return edited(left), sibling(right)


@pytest.mark.device
@pytest.mark.performance
@pytest.mark.asyncio
async def test_changed_helper_to_visible_result_under_five_seconds(monkeypatch, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("requires the designated Metal qualification device")
    original_read = OperationSources._read
    package = "ops.kernels._qualification_edit"
    fixture_root = Path(__file__).with_name("lab_edit_fixture")
    sources = {package: ModuleSource(package, tmp_path / "__init__.py", b"", True)}
    for name in ("helpers", "body"):
        path = tmp_path / f"{name}.py"
        path.write_bytes((fixture_root / f"{name}.py").read_bytes())
        sources[f"{package}.{name}"] = ModuleSource(f"{package}.{name}", path, path.read_bytes(), False)

    def read():
        return {**original_read(), **{
            name: (source if source.package else ModuleSource(name, source.path, source.path.read_bytes(), False))
            for name, source in sources.items()
        }}

    monkeypatch.setattr(OperationSources, "_read", staticmethod(read))
    OperationSources().refresh()
    ops.operation(edited)(sys.modules[f"{package}.body"].square)
    shape = ops.TensorSpec((65539,), ops.DType.F32)
    graph = ops.trace(parent, ops.Signature((ops.Argument(shape, "left"), ops.Argument(shape, "right"))))
    values = np.arange(shape.shape[0], dtype=np.float32) % 16 / 16
    fixture = Fixture.from_inputs(graph, {identity: values for identity in graph.inputs})
    path = Path(os.environ.get("MAGNITUDE_ROOFLINE_STORE", tmp_path / "iteration.sqlite"))
    plan = DevicePlan.discover(backend="metal", maximum_bytes=1 << 30)
    lab = Lab(fixture=fixture, device=lambda: ops.DeviceRuntime.open(plan), store=path,
              options=ops.CompileOptions(mode="prefill"), label="Changed-operation qualification")
    try:
        profile = await asyncio.wait_for(asyncio.wrap_future(lab.characterize()), timeout=300)
        assert profile.rates
        target, = lab.formulas.occurrences(edited)
        unchanged, = lab.formulas.occurrences(sibling)
        enclosing, = lab.formulas.occurrences(parent)
        initial = {}
        for item in (target, unchanged, enclosing):
            result = await asyncio.wait_for(asyncio.wrap_future(lab.measure(item).result), timeout=120)
            assert result.outcome == Outcome.COMPLETE, result.error
            history = await asyncio.wrap_future(lab.inspect(item))
            assert history.latest_success.roofline is not None
            initial[item] = result

        app = PerformanceApp(lab)
        async with app.run_test(size=(140, 45)) as pilot:
            tree = app.query_one(Tree)
            tree.root.expand_all()
            await pilot.pause()
            tree.select_node(app._formula_nodes[target])
            await pilot.pause()
            assert app._selected == target
            helper = sources[f"{package}.helpers"].path
            helper.write_text(helper.read_text().replace("return 64", "return 128"))
            started = perf_counter()
            await pilot.press("m")
            assert any(state.target == target and
                       (state.status in (Status.QUEUED, Status.RUNNING) or
                        state.job is not None and state.job.identity != initial[target].identity)
                       for state in lab.snapshot().states), "TUI did not request the selected measurement"
            deadline = started + 120
            while True:
                states = {state.target: state for state in lab.snapshot().states}
                state = states[target]
                if (state.job is not None and state.job.identity != initial[target].identity
                        and state.visibility is not None):
                    break
                assert perf_counter() < deadline, "edited operation never became visible"
                await asyncio.sleep(0.02)
            assert state.job.outcome == Outcome.COMPLETE, state.job.error
            assert state.visibility.client.startswith("tui:")
            assert state.history.latest_success.checked
            assert state.history.latest_success.roofline.characterization == profile.identity
            assert states[enclosing].status == Status.STALE
            assert states[unchanged].status == Status.CURRENT
            assert states[unchanged].job == initial[unchanged]
            checked = state.history.latest_success
            assert any(phase.phase == Phase.COMPILE and phase.elapsed_ns > 0 for phase in checked.phases)
            previous = state.history.observations[1]
            assert previous.series == checked.series
            assert previous.implementation != checked.implementation
            print("changed-operation phases:", state.job.model_dump_json())
            print("TUI visibility:", state.visibility.model_dump_json())
            assert state.visibility.request_to_visible_ns < 5_000_000_000
            assert perf_counter() - started < 5.0

        cached = await asyncio.wait_for(asyncio.wrap_future(lab.measure(unchanged).result), timeout=30)
        assert cached.outcome == Outcome.COMPLETE, cached.error
        history = await asyncio.wrap_future(lab.inspect(unchanged))
        assert not any(phase.phase == Phase.COMPILE for phase in history.latest_success.phases)
    finally:
        lab.close()
        for name in sources:
            sys.modules.pop(name, None)

    with ObservationStore(path) as store:
        assert len(store.history(checked.series).observations) == 2
        assert store.visibility(state.job.identity)
        configuration = store.configurations()[0]
        assert len(store.recorded_rooflines(configuration)) == 3
