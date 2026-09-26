from types import SimpleNamespace

import pytest
from analytical import publication
from rich.console import Console
from test_evidence import measurement
from textual.widgets import Select

from roofline.model_view import build_tree
from roofline.query import Queries
from roofline.store import Store
from roofline.tui import ComponentTree, RooflineApp, cell, detail_view

MODEL = "qwen:gguf:q4"


def put(store, name="A", bandwidth=200e9, seconds=0.001, phase="decode"):
    p = publication(name, bandwidth, seconds, phase)
    artifact = store.put_blob(p.model_dump_json().encode())
    m = measurement(
        measurement_id=name + phase,
        target=name,
        scope=phase,
        artifacts={"formula-performance": artifact},
    )
    store.put("measurement", m.measurement_id, m)
    return m


def test_model_consolidates_all_hardware_and_conditions(tmp_path):
    with Store(tmp_path) as store:
        put(store)
        put(store, "B", 800e9, 0.00025, "prefill")
        report = build_tree(store, MODEL)
        assert list(report.nodes) == ["", "block[0]"]
        relation = report.nodes["block[0]"].relation
        assert len(relation["points"]) == 2
        assert relation["attainment"] == {"minimum": 0.5, "maximum": 0.5, "points": 2}
        assert Queries(store).model(MODEL)["performance"] == report.analysis
        assert "50%" in cell(report.nodes["block[0]"]).plain
        console = Console(record=True, width=100)
        console.print(detail_view(report.nodes["block[0]"], report))
        text = console.export_text()
        assert "token/s" in text
        assert "A · complete operation" in text


def test_atomic_invalid_publication_does_not_publish_measurement(tmp_path):
    with Store(tmp_path) as store:
        p = publication().model_dump(mode="json")
        p["observations"][0]["hardware"] = "missing"
        from roofline.contracts import encoded

        artifact = store.put_blob(encoded(p))
        m = measurement(artifacts={"formula-performance": artifact})
        with pytest.raises(ValueError):
            store.put("measurement", m.measurement_id, m)
        assert store.records("measurement") == []
        assert store.records("performance-model") == []


def test_historical_measurements_are_not_requalified(tmp_path):
    with Store(tmp_path) as store:
        m = measurement()
        store.put("measurement", m.measurement_id, m)
        tree = build_tree(store, MODEL)
        assert not tree.nodes
        assert tree.analysis["historical_without_contract"] == 1
        assert "lack analytical contracts" in tree.notice


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40)])
async def test_browser_needs_only_model_and_never_groups_by_hardware(tmp_path, size):
    with Store(tmp_path) as store:
        put(store)
        put(store, "B", 800e9, 0.00025, "prefill")
    app = RooflineApp(SimpleNamespace(workspace=tmp_path, models={MODEL: None}))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        assert len(app.query(Select)) == 1
        tree = app.query_one(ComponentTree)
        assert tree.root.data.key == ""
        assert tree.root.children[0].data.key == "block[0]"
        await pilot.press("down")
        await pilot.pause()
        text = "\n".join(s.text for s in app.screen._compositor.render_strips())
        assert "50%" in text and "Block 0" in text
        assert "context" not in [s.id for s in app.query(Select)]
        await pilot.press("e")
        await pilot.pause()
        assert app.screen.id is None
        await pilot.press("escape")


def test_component_query_keeps_all_workloads_and_hardware(tmp_path):
    with Store(tmp_path) as store:
        put(store)
        put(store, "B", 800e9, 0.00025, "prefill")
        selected = Queries(store).model(MODEL, filters={"scope": "block[0]"})["performance"]
        assert selected["selected_component"] == "block[0]"
        assert len(selected["components"]["block[0]"]["points"]) == 2


def test_batch_publication_rolls_back_all_records_on_invalid_closure(tmp_path):
    from roofline.contracts import encoded

    with Store(tmp_path) as store:
        good = publication()
        bad = publication("B").model_dump(mode="json")
        bad["observations"][0]["hardware"] = "absent"
        a = measurement(
            measurement_id="a",
            artifacts={"formula-performance": store.put_blob(good.model_dump_json().encode())},
        )
        b = measurement(
            measurement_id="b", artifacts={"formula-performance": store.put_blob(encoded(bad))}
        )
        with pytest.raises(ValueError):
            store.put_many((("measurement", "a", a), ("measurement", "b", b)))
        assert store.records("measurement") == []
        assert store.records("performance-snapshot") == []
        assert store.db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0


def test_export_includes_transfer_baselines_and_replays_atomically(tmp_path):
    from formula_performance.records import Publication, Transfer

    from roofline.bundles import export_bundle, import_bundle

    with Store(tmp_path / "source") as store:
        original = publication()
        parent = original.observations[0].model_copy(
            update={"identity": "parent", "component": "", "samples": (0.02,)}
        )
        original = original.model_copy(update={"observations": (*original.observations, parent)})
        baseline = measurement(
            measurement_id="baseline",
            source_id=None,
            artifacts={"formula-performance": store.put_blob(original.model_dump_json().encode())},
        )
        store.put("measurement", "baseline", baseline)
        contract = Publication(
            transfers=(
                Transfer(
                    identity="cross-publication",
                    child="block[0]",
                    parent="",
                    baseline_child=original.observations[0].identity,
                    baseline_parent="parent",
                    contribution=0.001,
                    relationship="conditional-serial",
                    assumptions=("stable sibling costs",),
                    evidence=("parent",),
                ),
            )
        )
        child = measurement(
            measurement_id="contract",
            source_id=None,
            artifacts={"formula-performance": store.put_blob(contract.model_dump_json().encode())},
        )
        store.put("measurement", "contract", child)
        expected = store.derive(MODEL)
        path = tmp_path / "transfer.zip"
        assert export_bundle(store, ["contract"], path)["measurements"] == 2
    with Store(tmp_path / "import") as store:
        import_bundle(store, path)
        before = store.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        import_bundle(store, path)
        assert store.db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == before
        assert store.derive(MODEL) == expected


@pytest.mark.asyncio
async def test_slow_model_analysis_does_not_block_first_paint_or_input(tmp_path, monkeypatch):
    import asyncio
    import threading

    from textual.widgets import Static

    with Store(tmp_path) as store:
        put(store)
    started, release = threading.Event(), threading.Event()
    original = Queries.tree

    def slow_tree(self, *args, **kwargs):
        started.set()
        if not release.wait(10):
            raise RuntimeError("test did not release model analysis")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Queries, "tree", slow_tree)
    app = RooflineApp(SimpleNamespace(workspace=tmp_path, models={MODEL: None}))
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            assert await asyncio.to_thread(started.wait, 2)
            await pilot.pause()
            text = "\n".join(s.text for s in app.screen._compositor.render_strips())
            assert "Loading" in text
            await pilot.press("m")
            assert app.query_one("#model", Select).expanded
            await pilot.press("escape")
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.query_one(ComponentTree).root.data.key == ""
            assert "Loading" not in str(app.query_one("#context-note", Static).render())
    finally:
        release.set()
