from magnitude_compare import report


def test_unavailable_tuning_is_visible_without_inventing_load_counts():
    lines = []
    report._tuning(lines, [{
        "run": "base", "budget_seconds": 60, "fresh": True, "outcome": None,
        "unavailable": "baseline does not expose a structured trace",
    }])
    table = "\n".join(lines)
    assert "Unavailable: baseline does not expose a structured trace" in table
    assert "0 units" not in table
