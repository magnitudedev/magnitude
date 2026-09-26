from datetime import UTC, datetime

import pytest

from ops.lab.archive import RecordedConfiguration, RecordedFormula
from ops.lab.store import ObservationStore
from ops.lab.tui import RecordedApp, RecordedState, details
from tests.ops.test_observation_store import measurement, native_measurement, series


def configuration():
    condition = series()
    return RecordedConfiguration(identity="recording", label="Qwen decode", created=datetime.now(UTC),
                                 device=condition.device, formulas=(
        RecordedFormula(occurrence=0, definition=condition.formula, semantics=condition.semantics,
                        complete=True),
    ))


def test_recording_links_only_exact_formula_device_and_input_conditions(tmp_path):
    record = configuration()
    with ObservationStore(tmp_path / "history.sqlite") as store:
        store.publish_configuration(record)
        assert store.recorded_history(record, 0) is None
        first = measurement("first", "old-operation", 200)
        store.publish(first)
        store.link_series(record, 0, first.series)
        second = measurement("second", "new-operation", 100)
        store.publish(second)
        assert store.recorded_history(record, 0).latest_success == second
        assert store.recorded_overview(record) == {0: (1e-7, "complete")}
        with pytest.raises(ValueError, match="input conditions changed"):
            store.link_series(record, 0, first.series.model_copy(update={"fixture": "different"}))
        with pytest.raises(ValueError, match="occurrence"):
            store.link_series(record, 1, first.series)
        assert store.configurations() == (record,)


def test_native_and_wall_metrics_render_from_the_same_recorded_history(tmp_path):
    record = configuration()
    evidence = native_measurement()
    with ObservationStore(tmp_path / "native.sqlite") as store:
        store.publish_configuration(record)
        store.publish(evidence)
        store.link_series(record, 0, evidence.series)
        text = details(RecordedState(record.formulas[0], store.recorded_history(record, 0))).plain
        assert "Native kernel clock: test-native-clock" in text
        assert "kernel-device-time: 2.5e-07" in text
        assert "kernel-rate:arithmetic: 8e+08" in text
        assert "elapsed: 1e-06" in text
        assert "complete-operation wall time" in text


@pytest.mark.asyncio
async def test_recorded_browser_selects_history_without_a_device(tmp_path):
    record = configuration()
    with ObservationStore(tmp_path / "history.sqlite") as store:
        store.publish_configuration(record)
        evidence = measurement("evidence", "body", 100)
        store.publish(evidence)
        store.link_series(record, 0, evidence.series)
        app = RecordedApp(store, record)
        async with app.run_test() as pilot:
            await pilot.press("down")
            assert app._selected == record.formulas[0]
            assert app.store.recorded_history(record, 0).latest == evidence
            await pilot.press("o")
        assert app.return_value == "choose"
