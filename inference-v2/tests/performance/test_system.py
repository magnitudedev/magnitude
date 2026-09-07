import json
from dataclasses import replace

import pytest

from magnitude_engine.components import ComponentId
from performance.assembly import BoundAssembly
from performance.facts import (
    AttentionGeometry,
    Configuration,
    KVGeometry,
    KVStorage,
    NeuralParameters,
    RecurrentStorage,
    TensorFacts,
)
from performance.records import Assembly, Node, Observation, Profile
from performance.runner import recording
from performance.store import Store
from performance.theory.resources import Demands, Extent, join, subtract, time_bound


class Operator:
    def __call__(self):
        return 2


def binding(*, parent="v1", child="stable", mode="serial"):
    graph = Assembly(
        "root",
        {
            "root": Node(
                ComponentId("MODEL:QWEN35:MAG:LAYERWISE"),
                parent,
                children={"attention": "attention"},
                execution=mode,
            ),
            "attention": Node(ComponentId("MODEL:ATTENTION:MAG:PAGED"), child),
        },
        "test",
    )
    return BoundAssembly(graph, {"root": Operator(), "attention": Operator()}, {})


POINT = {
    "histories": [9],
    "query_tokens": 1,
    "batch_size": 1,
    "geometry": {
        "query_heads": 2,
        "kv_heads": 1,
        "key_width": 2,
        "value_width": 2,
        "element_bytes": 2,
    },
}
PROFILE = Profile(
    {"machine": "one"}, {"mlx": "test"}, {"dram_bytes_per_second": 1000, "fast_storage_bytes": 0}
)


def measured(bound, path, directory, *, profile=PROFILE, seconds=1):
    ticks = iter((0, int(seconds * 1e9), int(2 * seconds * 1e9), int(3 * seconds * 1e9)))
    with recording(
        bound.at(path),
        benchmark="attention.decode",
        workload=POINT,
        profile=profile,
        warmup=0,
        repetitions=2,
        output=directory,
        clock=lambda: next(ticks),
    ) as run:
        run.measure(lambda: 2, validate=lambda x: Observation(str(x)), deterministic=True)
    return run


def test_shared_component_results_recompute_both_compositions(tmp_path):
    first = binding()
    second_graph = binding(parent="v2")
    measured(first, "root", tmp_path)
    measured(second_graph, "root", tmp_path)
    measured(first, "attention", tmp_path, seconds=2)
    state = Store(tmp_path).state()
    assert len(state["compositions"]) == 1
    assert len(next(iter(state["compositions"].values()))["revisions"]) == 2
    child_keys = {v["assessments"]["attention"] for v in state["views"].values()}
    assert len(child_keys) == 1
    assessment = state["components"][child_keys.pop()]["dimensions"]["EXEC"]
    assert assessment["observed"] == 2
    assert assessment["percent"] == pytest.approx(4.4)
    # Direct parent observations remain direct, not child time plus parent time.
    for view in state["views"].values():
        assert (
            state["components"][view["assessments"]["root"]]["dimensions"]["EXEC"]["observed"] == 1
        )


def test_hardware_and_revision_do_not_mix(tmp_path):
    a = measured(binding(), "attention", tmp_path)
    measured(
        binding(),
        "attention",
        tmp_path,
        profile=replace(PROFILE, hardware={"machine": "two"}),
        seconds=2,
    )
    measured(binding(child="changed"), "root", tmp_path)
    state = Store(tmp_path).state()
    assert len(state["profiles"]) == 2
    changed = next(
        v
        for v in state["views"].values()
        if v["revision"] == binding(child="changed").graph.revision
    )
    assert (
        state["components"][changed["assessments"]["attention"]]["dimensions"]["EXEC"]["observed"]
        is None
    )
    assert json.loads(a.path.read_text())["status"] == "complete"
    current = next(iter(state["compositions"].values()))["current_assessments"]
    value = state["components"][current["attention"]["EXEC"]]["dimensions"]["EXEC"]
    assert value["observed"] is None  # Current source cannot reuse the previous implementation.


def test_import_is_idempotent_and_rebuild_order_independent(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    a = measured(binding(), "attention", source)
    b = measured(binding(), "attention", source, seconds=2)
    store = Store(dest)
    assert store.ingest(b.record)
    assert store.ingest(a.record)
    assert not store.ingest(a.record)
    assert store.state() == Store(source).state()
    broken = dict(a.record, status="failed")
    with pytest.raises(ValueError, match="checksum"):
        store.ingest(broken)


def test_failure_and_interrupt_are_persisted(tmp_path):
    for error in (ValueError("invalid output"), KeyboardInterrupt()):
        with pytest.raises(type(error)):
            with recording(
                binding().at("attention"),
                benchmark="failure",
                workload=POINT,
                profile=PROFILE,
                output=tmp_path,
                warmup=0,
                repetitions=1,
            ) as run:

                def fail(error=error):
                    raise error

                run.measure(fail)
        record = json.loads(run.path.read_text())
        assert record["status"] in ("failed", "interrupted")
        assert len(record["samples"]) == 1
        assert all(
            d["observed"] is None
            for c in Store(tmp_path).state()["components"].values()
            for d in c["dimensions"].values()
        )


def test_timing_completion_and_validation_order(tmp_path):
    events = []
    ticks = iter((0, 10))

    def clock():
        events.append("clock")
        return next(ticks)

    with recording(
        binding().at("attention"),
        benchmark="order",
        workload=POINT,
        profile=PROFILE,
        output=tmp_path,
        warmup=0,
        repetitions=1,
        clock=clock,
    ) as run:
        run.measure(
            lambda: events.append("invoke"),
            prepare=lambda: events.append("prepare"),
            complete=lambda _: events.append("complete"),
            validate=lambda _: (events.append("validate"), Observation())[1],
        )
    assert events == ["prepare", "clock", "invoke", "complete", "clock", "validate"]


def test_resource_union_eliminates_internal_traffic_but_not_arithmetic():
    a = Demands((Extent("weights", 0, 100),), (Extent("hidden", 0, 20),), {"scalar": 50})
    b = Demands(
        (Extent("weights", 50, 120), Extent("hidden", 0, 20)),
        (Extent("output", 0, 10),),
        {"scalar": 30},
    )
    d = join(a, b)
    assert d.inputs == (Extent("weights", 0, 120),)
    assert d.outputs == (Extent("output", 0, 10),)
    assert d.operations == {"scalar": 80}
    assert join(a, b, retained=(Extent("hidden", 0, 20),)).outputs == (
        Extent("hidden", 0, 20),
        Extent("output", 0, 10),
    )
    assert subtract((Extent("a", 0, 20),), (Extent("a", 5, 10),)) == (
        Extent("a", 0, 5),
        Extent("a", 10, 20),
    )
    bound = time_bound(d, PROFILE)
    assert bound.value == 0.12
    assert "scalar_per_second" in bound.missing


def test_zero_bound_and_missing_capacity_are_not_fake_efficiencies():
    assert time_bound(Demands(), PROFILE).value == 0
    b = time_bound(Demands((Extent("x", 0, 20),)), Profile({}, {}))
    assert b.value is None
    assert set(b.missing) == {"dram_bytes_per_second", "fast_storage_bytes"}


def test_dimensions_select_evidence_independently_and_version_contracts(tmp_path):
    graph = Assembly(
        "state",
        {
            "state": Node(
                ComponentId("STATE:RECURRENT:MAG:CHECKPOINTED"),
                "source",
                RecurrentStorage(layouts=()),
            )
        },
        "state",
    )
    bound = BoundAssembly(graph, {"state": Operator()}, {}).at("state")
    point = {
        "retained_shapes": [{"identity": "s", "shape": [4], "element_bytes": 4}],
        "restore_mode": "saved_boundary",
    }
    for dimension in (None, "RESTORE"):
        with recording(
            bound,
            benchmark="state.test",
            workload=point,
            profile=PROFILE,
            output=tmp_path,
            warmup=0,
            repetitions=1,
        ) as run:
            run.measure(
                lambda: None,
                dimension=dimension,
                validate=lambda _, dimension=dimension: Observation(
                    metrics={"MEM": 32} if dimension is None else {}
                ),
            )
    values = next(iter(Store(tmp_path).state()["components"].values()))["dimensions"]
    assert values["MEM"]["percent"] == 50
    assert values["RESTORE"]["observed"] is not None
    assert values["MEM"]["evidence"] != values["RESTORE"]["evidence"]
    published = Store(tmp_path).state()
    current = next(iter(published["compositions"].values()))["current_assessments"]["state"]
    assert published["components"][current["MEM"]]["dimensions"]["MEM"] == values["MEM"]
    assert published["components"][current["RESTORE"]]["dimensions"]["RESTORE"] == values["RESTORE"]
    with recording(
        bound,
        benchmark="state.test",
        workload=point,
        profile=PROFILE,
        contract_version="2",
        output=tmp_path,
        warmup=0,
        repetitions=1,
    ) as run:
        run.measure(lambda: None, dimension="RESTORE")
    state = Store(tmp_path).state()
    assert len(state["components"]) == 2


def test_explicit_child_boundary_updates_parent_without_false_parent_timing(tmp_path):
    graph = binding(mode="serial")
    with recording(
        graph.at("root"),
        benchmark="model.test",
        workload={"mode": "parent"},
        boundary="parent",
        bindings={"attention": {"workload": POINT, "boundary": "component-through-ready"}},
        profile=PROFILE,
        output=tmp_path,
        warmup=0,
        repetitions=1,
    ) as run:
        run.measure(lambda: None, dimension=None)
    child = measured(graph, "attention", tmp_path, seconds=2)
    state = Store(tmp_path).state()
    view = next(v for v in state["views"].values() if v["boundary"] == "parent")
    values = state["components"][view["assessments"]["root"]]["dimensions"]["EXEC"]
    assert values["observed"] == 2
    assert values["estimated"]
    assert values["evidence"] == [child.record["id"]]


def test_preflight_shared_kv_and_invalid_inputs_are_data():
    from performance.assessment import formulate, preflight

    leaf = dict(POINT["geometry"], kv_source=0)
    graph = Assembly(
        "model",
        {
            "model": Node(
                ComponentId("MODEL:GEMMA4:MAG:LAYERWISE"),
                "x",
                parameters=NeuralParameters(),
                children={"a": "model.layers.0.attention", "b": "model.layers.1.attention"},
            ),
            **{
                f"model.layers.{i}.attention": Node(
                    ComponentId("MODEL:ATTENTION:MAG:GATHERED"),
                    "x",
                    parameters=AttentionGeometry(**leaf),
                )
                for i in range(2)
            },
        },
        "shared KV",
    )
    result = formulate(graph, POINT, PROFILE)
    assert sum(e.end - e.start for e in result["demands"]["model"].inputs) == 72
    broken = preflight(graph, POINT | {"query_tokens": -1}, PROFILE)
    assert "invalid binding" in broken["model.layers.0.attention"]["EXEC"]["missing"][0]


def test_finalized_runs_are_immutable_and_active_recovery_rejected(tmp_path):
    result = measured(binding(), "attention", tmp_path)
    with pytest.raises(ValueError, match="immutable"):
        Store(tmp_path).finalize(result.record)
    with recording(
        binding().at("attention"),
        benchmark="active",
        workload=POINT,
        profile=PROFILE,
        output=tmp_path,
        warmup=0,
        repetitions=1,
    ) as run:
        with pytest.raises(ValueError, match="still active"):
            Store(tmp_path).recover(run.record["id"])
        run.measure(lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (160, 55)])
async def test_tui_and_document_use_published_assessments(tmp_path, size):
    from textual.widgets import Select, Tree

    from performance.presentation import export_document, render_tree
    from performance.tui.app import PerformanceApp

    measured(binding(), "attention", tmp_path)
    second = binding(mode="joint")
    second.graph = replace(second.graph, artifacts={"test": "another-model"})
    measured(second, "root", tmp_path)
    store = Store(tmp_path)
    state = store.state()
    view = next(iter(state["views"]))
    document = tmp_path / "model.md"
    document.write_text("# Model\n\n## Assembly\n\n```text\nold\n```\n")
    export_document(state, view, document, store=store)
    assert render_tree(state, view) in document.read_text()
    async with PerformanceApp(store).run_test(size=size) as pilot:
        await pilot.pause()
        app = pilot.app
        tree = app.query_one(Tree)
        selector = app.query_one(Select)
        assert len(app.query(Select)) == 1
        assert selector.region.height == 3
        assert tree.region.height >= size[1] - 6
        details = app.query_one("#details-scroll")
        assert details.region.height == tree.region.height
        assert details.region.right <= size[0]
        assert tree.root.children and tree.root.is_expanded
        assert tree.has_focus
        await pilot.press("down")
        assert tree.cursor_node.data == "attention"
        assert "MODEL:ATTENTION" in app.export_screenshot()
        await pilot.press("f")
        assert tree.root.data == "attention"
        await pilot.press("escape")
        assert tree.root.data == "root"
        # A newly published generation keeps the selected component and expansion.
        tree.select_node(tree.root.children[0])
        measured(binding(), "attention", tmp_path, seconds=2)
        app.action_refresh()
        await pilot.pause()
        assert tree.cursor_node.data == "attention"
        assert "4.40%" in str(tree.cursor_node.label)
        # Exercise the real selector, not just its backing field.
        await pilot.press("c")
        assert selector.expanded
        previous = selector.value
        await pilot.press("end", "enter")
        await pilot.pause()
        if selector.value == previous:
            await pilot.press("c", "home", "down", "enter")
            await pilot.pause()
        assert selector.value != previous
        assert tree.has_focus
        assert tree.root.data == "root"
        assert tree.root.children


@pytest.mark.asyncio
async def test_tui_empty_store_is_visible(tmp_path):
    from textual.widgets import Select, Tree

    from performance.tui.app import PerformanceApp

    async with PerformanceApp(Store(tmp_path)).run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert pilot.app.query_one(Select).disabled
        assert pilot.app.query_one(Tree).region.height >= 18
        assert "No recorded compositions" in str(pilot.app.query_one(Tree).root.label)


def test_concurrent_imports_publish_one_complete_generation(tmp_path):
    import subprocess
    import sys

    source = tmp_path / "source"
    records = [measured(binding(), "attention", source, seconds=n) for n in (1, 2)]
    command = (
        "from pathlib import Path; import json,sys; from performance.store import Store; "
        "Store(Path(sys.argv[1])).ingest(json.loads(Path(sys.argv[2]).read_text()))"
    )
    children = [
        subprocess.Popen([sys.executable, "-c", command, str(tmp_path / "dest"), str(r.path)])
        for r in records
    ]
    try:
        for child in children:
            assert child.wait(timeout=15) == 0
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
    assert Store(tmp_path / "dest").state() == Store(source).state()


def test_recover_dead_process_keeps_complete_journal_samples(tmp_path):
    import platform

    from performance.records import digest

    result = measured(binding(), "attention", tmp_path)
    original = result.record
    interrupted = {
        **original,
        "id": "f" * 32,
        "status": "running",
        "process": {"hostname": platform.node(), "pid": 2147483647, "started_at": 0},
    }
    interrupted.pop("checksum")
    directory = tmp_path / "runs" / interrupted["id"]
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps(interrupted))
    (directory / "samples.jsonl").write_text(json.dumps(original["samples"][0]) + '\n{"truncated":')
    record = Store(tmp_path).recover(interrupted["id"])
    assert record["status"] == "interrupted"
    assert len(record["samples"]) == 1
    assert record["checksum"] == digest({k: v for k, v in record.items() if k != "checksum"})


def test_formula_revision_rebuilds_existing_raw_evidence(tmp_path, monkeypatch):
    measured(binding(), "attention", tmp_path)
    old = Store(tmp_path).state()
    monkeypatch.setattr("performance.assessment.revision", lambda: "revised-formula")
    current = Store(tmp_path).refresh()
    assert current["theory_revision"] == "revised-formula"
    assert current["runs"] == old["runs"]
    assert current["generation"] != old["generation"]


def test_hybrid_memory_composes_disjoint_backing_once(tmp_path):
    graph = Assembly(
        "state",
        {
            "state": Node(
                ComponentId("STATE:QWEN35:MAG:HYBRID"),
                "s",
                Configuration(),
                children={"kv": "kv", "recurrent": "r"},
            ),
            "kv": Node(
                ComponentId("KV:STORE:MAG:PAGED"),
                "s",
                parameters=KVStorage(
                    layers=(KVGeometry(heads=1, key_width=2, value_width=2),),
                    element_bytes=2,
                    page_size=16,
                    slab_pages=32,
                    max_pages=1024,
                ),
            ),
            "r": Node(
                ComponentId("STATE:RECURRENT:MAG:CHECKPOINTED"),
                "s",
                parameters=RecurrentStorage(
                    layouts=(
                        (TensorFacts(identity="state", shape=(1, 4), bytes=16, dtype="float32"),),
                    )
                ),
            ),
        },
        "hybrid",
    )
    assembly = BoundAssembly(graph, {p: Operator() for p in graph.nodes}, {})
    for path, size in (("kv", 32), ("r", 64)):
        with recording(
            assembly.at(path),
            benchmark="state.retention",
            profile=PROFILE,
            workload={"retained_positions": 2, "retained_rows": 1},
            warmup=0,
            repetitions=1,
            output=tmp_path,
        ) as run:
            run.measure(
                lambda: None,
                dimension=None,
                validate=lambda _, size=size: Observation(metrics={"MEM": size}),
            )
    state = Store(tmp_path).state()
    view = next(iter(state["views"].values()))
    value = state["components"][view["assessments"]["state"]]["dimensions"]["MEM"]
    assert value["observed"] == 96
    assert value["bound"]["value"] == 32
    assert value["estimated"]
    assert len(value["evidence"]) == 2
