"""The forced-pressure cell judges only the resizes in the held pressure's window, each by what it
costs the decode, and reports a step slow on its own as a slow step, not as a resize's stall."""

import json

from magnitude_rig.host import memory

STEP = 0.025


def decode(slow_at=None, slow=0.0):
    """Outputs of one token every step from t=0, one of them `slow` seconds late."""
    steps, at = [], 0.0
    for token in range(1, 201):
        at += STEP + (slow if token == slow_at else 0.0)
        steps.append((token, at))
    return steps


def resize(at, ms=1.0, release=0.002):
    return {"rung": "Bank", "bank_slots": 200, "ms": ms, "queued": 0.0, "planned": 0.5, "moves": 0.3,
            "decided": at, "prepared": at + 0.0005, "installed": at + 0.001, "released": at + 0.001 + release}


def report(steps, resizes, window=(50, 150), served=()):
    """`served` the streamed steps' reads as they completed."""
    pressure = {"steps": steps, "window": {"from": window[0], "to": window[1]} if window else {}}
    memory.report_stalls(pressure, resizes, 0.0, list(served))


def test_a_slow_step_no_resize_costs_is_a_slow_step(capsys):
    steps = decode(slow_at=100, slow=0.060)
    at = dict(steps)
    # The streamed step completed in the slow wait missed 12 experts.
    read = {"at": at[100] - 0.001, "misses": 12, "missed_bytes": 24_000_000, "read_micros": 55_000,
            "parked_micros": 3_000}
    report(steps, [resize(at[60] + 0.010)], served=[read])
    out = capsys.readouterr().out
    assert "1 of 1 resizes in it" in out
    assert "by thread time: 1 of 1" in out
    assert "slow steps of their own in the window: 1" in out
    assert "missed 12 experts, 24.0 MB read on demand, 55.0 ms reading (3.0 ms parked)" in out


def test_resizes_every_step_are_judged_against_their_neighbours(capsys):
    # A resize in every step of the window, one step 34 ms slow that its 0.6 ms resize does not explain.
    steps = decode(slow_at=100, slow=0.034)
    at = dict(steps)
    report(steps, [resize(at[token] + 0.005, ms=0.6, release=0.001) for token in range(50, 150)])
    out = capsys.readouterr().out
    assert "by thread time: 100 of 100" in out
    assert "slow steps of their own in the window: 1" in out


def test_resizes_outside_the_window_are_not_judged(capsys):
    steps = decode()
    at = dict(steps)
    report(steps, [resize(at[10] + 0.010)])
    assert "NO RESIZE IN THE WINDOW" in capsys.readouterr().out


def test_a_window_that_never_closed_tested_nothing(capsys):
    report(decode(), [resize(1.0)], window=None)
    assert "NOTHING TESTED" in capsys.readouterr().out


def _shrink(at, charged, footprint=(None, None), holding="Host", claimed=(None, None)):
    fields = {"message": "memory released", "at": at, "rung": "Bank", "micros": 1000, "bank_slots": 4,
              "holding": holding, "from": 8, "to": 0, "released": 80,
              "charged_before": charged[0], "charged_after": charged[1],
              "claimed_before": claimed[0], "claimed_after": claimed[1],
              "footprint_before": footprint[0], "footprint_after": footprint[1]}
    fields.update({f"{key}_micros": 100 for key in memory.SHRINK_STEPS},
                  **{f"{key}_before": 0 for key in ("decided", "prepared", "installed", "released")})
    return memory.EVENT + __import__("json").dumps(fields)


def test_the_pressure_gate_fails_on_missing_transitions_and_wrong_tokens(tmp_path, capsys):
    charge = physical_cell()
    charge.update(decoded_tokens=[1, 2, 3])
    charge["pressure"]["steps"] = decode()
    failures = memory.report_pressure(charge, [], [1, 2, 4], tmp_path)
    assert any("never shrank" in failure for failure in failures)
    assert any("differs" in failure for failure in failures)
    capsys.readouterr()


def test_the_pressure_gate_fails_a_bank_step_that_needs_new_memory(tmp_path, capsys):
    """A shrink's charge and footprint fall at every step: one that grew either fails the gate."""
    charge = physical_cell()
    charge.update(decoded_tokens=[1, 2, 3])
    charge["pressure"]["steps"] = decode()
    grew = memory.report_pressure(charge, [_shrink(3, (1e9, 2e9))], [1, 2, 3], tmp_path)
    assert any("1 of 1 bank steps grew" in failure for failure in grew)
    footprint = memory.report_pressure(charge, [_shrink(3, (2e9, 1e9), (5e9, 6e9))], [1, 2, 3], tmp_path)
    assert any("1 of 1 bank steps grew" in failure for failure in footprint)
    fell = memory.report_pressure(charge, [_shrink(3, (2e9, 1e9), (6e9, 5e9))], [1, 2, 3], tmp_path)
    assert not any("bank steps grew" in failure for failure in fell)
    # A discrete device's holding is judged by its store's claim, not the process's host footprint.
    device = lambda claimed: memory.report_pressure(
        charge, [_shrink(3, (2e9, 1e9), (5e9, 6e9), "Experts", claimed)], [1, 2, 3], tmp_path)
    assert any("1 of 1 bank steps grew" in failure for failure in device((3e9, 4e9)))
    assert not any("bank steps grew" in failure for failure in device((3e9, 2e9)))
    capsys.readouterr()


def physical_cell():
    observation = {"store": "HostRam", "name": "HostRam", "headroom_bytes": 100, "ceiling_bytes": 50}
    return {"pressure": {"started": {"at": 1}, "approached": {"at": 1.5, "before": observation},
                         "occupied": {"at": 2, "observation": observation},
                         "freeing": {"at": 3.9},
                         "released": {"at": 4, "observation": observation},
                         "freed": {"at": 4.5, "observation": observation},
                         "window": {"from": 10, "to": 30}},
            "bank": {"holdings": {"Host": {"stores": ["HostRam"], "initial": 8, "floor": 0, "physical_floor": 0, "storage_kind": "Host"},
                                  "Experts": {"stores": ["Gpu"], "initial": 16, "floor": 4, "physical_floor": 4, "storage_kind": "Device"}}}}


SAMPLES = [{"at": 2.5, "charge": 80, "ceiling": 70}, {"at": 3, "charge": 0, "ceiling": 10}, {"at": 3.8, "charge": 0, "ceiling": 10}]


def test_physical_pressure_requires_the_same_holding_to_shrink_and_regrow_to_all_it_held():
    charge = physical_cell()
    shrink = {"at": 3, "decided": 2.5, "installed": 2.8, "released": 2.9, "rung": "Bank", "holding": "Host", "from": 8, "to": 0, "bytes": 80}
    grow = {"at": 5, "decided": 4.1, "installed": 4.2, "released": 4.3, "rung": "Growth", "holding": "Host", "from": 0, "to": 8, "bytes": 80}
    transitions = lambda events, samples=SAMPLES: memory.pressure_transitions(charge, events, samples)
    assert transitions([shrink, grow]) == []
    # Growth while the occupier frees, within what it leaves, is no failure.
    assert transitions([shrink, {**grow, "decided": 3.4, "installed": 3.5}]) == []
    assert "never shrank" in transitions([grow])[0]
    assert "regrow" in transitions([shrink])[0]
    assert "regrow" in transitions([shrink, {**grow, "to": 4}])[0]
    # Short of all it held, but at what its record justifies: more saves nothing.
    assert transitions([shrink, {"at": 4.1, "decision": "Host", "justified": 4}, {**grow, "to": 4}]) == []
    assert "regrow" in transitions([shrink, {"at": 4.1, "decision": "Host", "justified": 6}, {**grow, "to": 4}])[0]
    assert "regrow" in transitions([shrink, {**grow, "holding": "Experts"}])[0]
    assert "never shrank" in transitions([{**shrink, "bytes": 0}, grow])[0]
    assert "never shrank" in transitions([{**shrink, "holding": "Experts"}, grow])[0]
    assert "never shrank" in transitions([{**shrink, "released": 0.9}, grow])[0]
    assert "unverified" in transitions([shrink, {**grow, "decided": None}])[0]


def test_the_load_answers_the_ceiling_and_never_grows_over_it():
    growth = {"installed": 3.1}
    under = {"charge": 5, "ceiling": 10}
    over = {"charge": 15, "ceiling": 10}
    within = lambda samples, growths=(): memory.within_ceiling(samples, growths, 2, 4)
    answered = [{"at": 2.5, **over}, {"at": 3, **under}, {"at": 3.5, **under}]
    assert within(answered) == ([], 1)
    assert "as the pressure ends" in within([{"at": 3, **under}, {"at": 3.5, **over}])[0][0]
    assert "growth step" in within([{"at": 3, **under}, {"at": 3.2, **over}, {"at": 3.5, **under}], [growth])[0][0]
    assert "no reading" in within([{"at": 1, **under}, {"at": 5, **over}])[0][0]


def test_memory_samples_read_the_pressed_store():
    line = lambda store: memory.EVENT + __import__("json").dumps(
        {"at": 1.0, "message": "memory read", "store": store, "headroom": 9, "planning": 4, "charged": 20})
    assert memory.memory_samples([line("HostRam"), line("Gpu"), "other"], "HostRam") == [{"at": 1.0, "charge": 20, "ceiling": 25}]


def test_pressure_rejects_missing_observations_windows_and_early_costlier_release():
    charge = physical_cell()
    shrink = {"at": 3, "decided": 2.5, "installed": 2.8, "released": 2.9, "rung": "Bank", "holding": "Host", "from": 8, "to": 0, "bytes": 80}
    grow = {"at": 5, "decided": 4.1, "installed": 4.2, "released": 4.3, "rung": "Growth", "holding": "Host", "from": 0, "to": 8, "bytes": 80}
    early = {"at": 2.5, "rung": "Prefix", "cause": "Reclaim", "reclaim_device_deficit_bytes": 0, "reclaim_host_deficit_bytes": 10}
    assert any("costlier" in failure for failure in memory.pressure_transitions(charge, [early, shrink, grow], SAMPLES))
    assert memory.pressure_transitions(charge, [shrink, {**early, "at": 3.5}, grow], SAMPLES) == []
    del charge["pressure"]["freed"]
    assert "missing freed" in memory.pressure_transitions(charge, [shrink, grow], SAMPLES)[0]
    charge = physical_cell()
    charge["pressure"]["window"] = {}
    assert "incomplete" in memory.pressure_transitions(charge, [shrink, grow], SAMPLES)[0]
    charge = physical_cell()
    charge["pressure"]["occupied"]["observation"] = {"store": "Gpu"}
    assert "different physical stores" in memory.pressure_transitions(charge, [shrink, grow], SAMPLES)[0]


def test_pressure_selector_and_dedicated_serving_options_share_the_cli_judge():
    base = ["--bin", "bin", "--backend", "cuda", "--memory", "dedicated", "--scratch", "scratch",
            "--artifacts", "artifacts", "--context", "4096", "--weight-cap", "1000",
            "--residency", "harness", "fixture", "256", "--pressure", "100"]
    assert "required together" in memory.refusal(memory.parser().parse_args(base))
    assert memory.refusal(memory.parser().parse_args([*base, "--pressure-store", "device"])) is None


def test_resize_event_retains_holding_and_backing_bytes():
    event = {"message": "memory grown", "at": 5, "micros": 1000, "holding": "Host",
             "from": 0, "to": 4, "grown": 40,
             **{f"{key}_before": 0 for key in ("decided", "prepared", "installed", "released")}}
    [record] = memory.releases([memory.EVENT + json.dumps(event)])
    assert (record["holding"], record["from"], record["to"], record["bytes"]) == ("Host", 0, 4, 40)


def test_dedicated_pressure_plans_the_existing_serving_harness(monkeypatch):
    from argparse import Namespace
    from pathlib import Path
    from magnitude_rig.instruments import memory as instrument, harness
    monkeypatch.setattr(instrument.device, "memory", lambda host, backend: "dedicated")
    arguments = Namespace(backend="cuda", model="qwen3.6-35b-a3b:gguf:q4", weight_cap=1000,
                          context=4096, residency=256, pressure=100, pressure_store="device",
                          pressure_prompt=False, recommit=None, fixture=Path("fixture.jsonl"))
    plan = instrument.plan(arguments, None, None)
    assert harness.BUILD in plan.builds
    assert [name for name, job in plan.jobs] == ["measure", "memory"]
    assert plan.files == (arguments.fixture,)
    assert "--pressure-store" in plan.jobs[-1][1].argv


def test_physical_exhaustion_plateau_is_not_a_premature_release():
    charge = physical_cell()
    charge["bank"]["holdings"]["Host"]["physical_floor"] = 1
    events = [{"at": 3, "decided": 2.5, "installed": 2.8, "released": 2.9, "rung": "Bank", "holding": "Host", "from": 8, "to": 1, "bytes": 80},
              {"at": 3.5, "rung": "Prefix", "cause": "Reclaim", "reclaim_device_deficit_bytes": 0, "reclaim_host_deficit_bytes": 10},
              {"at": 5, "decided": 4.1, "installed": 4.2, "released": 4.3, "rung": "Growth", "holding": "Host", "from": 1, "to": 8, "bytes": 70}]
    assert memory.pressure_transitions(charge, events, SAMPLES) == []
    events.insert(0, {"refusal": {"holding": "Host", "at": 4, "refusal": "partial recommit"}})
    assert "unverified after resize refusal" in memory.pressure_transitions(charge, events, SAMPLES)[0]


def test_costlier_release_proof_uses_the_selected_stores_reclaim_cause():
    charge = physical_cell()
    shrink = {"at": 3, "decided": 2.5, "installed": 2.8, "released": 2.9, "rung": "Bank", "holding": "Host", "from": 8, "to": 0, "bytes": 80}
    grow = {"at": 5, "decided": 4.1, "installed": 4.2, "released": 4.3, "rung": "Growth", "holding": "Host", "from": 0, "to": 8, "bytes": 80}
    release = {"at": 2, "rung": "Prefix", "cause": "Reclaim",
               "reclaim_device_deficit_bytes": 10, "reclaim_host_deficit_bytes": 0}
    # GPU pressure may evict a prefix while host backing remains above its floor.
    assert memory.pressure_transitions(charge, [release, shrink, grow], SAMPLES) == []
    selected = {**release, "reclaim_host_deficit_bytes": 10}
    assert "exhausted reclaimable backing" in memory.pressure_transitions(charge, [selected, shrink, grow], SAMPLES)[0]
    for unknown in ({**selected, "cause": "LiveClaim"}, {**selected, "cause": None},
                    {**selected, "reclaim_host_deficit_bytes": None}):
        assert "unverified" in memory.pressure_transitions(charge, [unknown, shrink, grow], SAMPLES)[0]


def test_a_moment_reads_from_the_harness_event_and_the_engine_log_line():
    record = {"moment": "loaded", "footprint_bytes": 7}
    event = memory.EVENT + json.dumps({"at": 1.0, "message": "load memory", "record": json.dumps(record)})
    logged = "2026-11-03T00:00:00Z  INFO magnitude_engine::execution: load memory record=" + json.dumps(record)
    other = memory.EVENT + json.dumps({"at": 1.0, "message": "memory read"})
    assert memory.moments([(1.0, event), (2.0, other)]) == {"loaded": (1.0, record)}
    assert memory.moments([(3.0, logged), (4.0, "engine: other")]) == {"loaded": (3.0, record)}
