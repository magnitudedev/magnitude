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
    return {"resize": "Shrink", "bank_slots": 200, "ms": ms, "queued": 0.0, "planned": 0.5,
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


def _shrink(at, held, holding="Host", returned=80):
    fields = {"message": "memory released", "at": at, "micros": 1000, "bank_slots": 4,
              "holding": holding, "from": 8, "to": 0, "released": 80, "returned": returned,
              "held_before": held[0], "held_after": held[1]}
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


def test_the_pressure_gate_fails_a_bank_shrink_that_frees_nothing_or_needs_new_memory(tmp_path, capsys):
    """A shrink returns its empty blocks to the OS, and what the process holds never rises across
    one: a shrink that returns nothing, or that grew what the process holds, fails the gate."""
    charge = physical_cell()
    charge.update(decoded_tokens=[1, 2, 3])
    charge["pressure"]["steps"] = decode()
    grew = memory.report_pressure(charge, [_shrink(3, (1e9, 2e9))], [1, 2, 3], tmp_path)
    assert any("1 of 1 bank shrinks grew" in failure for failure in grew)
    for returned in (0, None):
        kept = memory.report_pressure(charge, [_shrink(3, (2e9, 2e9), "Experts", returned)], [1, 2, 3], tmp_path)
        assert any("1 of 1 bank shrinks returned nothing to the OS" in failure for failure in kept)
    freed = memory.report_pressure(charge, [_shrink(3, (2e9, 1e9), "Experts")], [1, 2, 3], tmp_path)
    assert not any("bank shrinks" in failure for failure in freed)
    capsys.readouterr()


def physical_cell():
    observation = {"store": "HostRam", "name": "HostRam", "headroom_bytes": 100}
    return {"pressure": {"started": {"at": 1}, "approached": {"at": 1.5, "before": observation},
                         "occupied": {"at": 2, "observation": observation},
                         "freeing": {"at": 3.9},
                         "released": {"at": 4, "observation": observation},
                         "freed": {"at": 4.5, "observation": observation},
                         "window": {"from": 10, "to": 30}},
            "bank": {"holdings": {"Host": {"stores": ["HostRam"], "initial": 8, "floor": 0},
                                  "Experts": {"stores": ["Gpu"], "initial": 16, "floor": 4}}}}


SAMPLES = [{"at": 2.5, "charge": 80, "ceiling": 70}, {"at": 3, "charge": 0, "ceiling": 10}, {"at": 3.8, "charge": 0, "ceiling": 10}]
SHRINK = {"at": 3, "decided": 2.5, "installed": 2.8, "released": 2.9, "resize": "Shrink", "holding": "Host", "from": 8, "to": 0, "bytes": 80}
GROW = {"at": 5, "decided": 4.1, "installed": 4.2, "released": 4.3, "resize": "Growth", "holding": "Host", "from": 0, "to": 8, "bytes": 80}


def test_physical_pressure_requires_the_same_holding_to_shrink_and_regrow_to_all_it_held():
    charge = physical_cell()
    shrink, grow = SHRINK, GROW
    transitions = lambda events, samples=SAMPLES: memory.pressure_transitions(charge, events, samples)
    assert transitions([shrink, grow]) == []
    # Growth while the occupier frees, within what it leaves, is no failure.
    assert transitions([shrink, {**grow, "decided": 3.4, "installed": 3.5}]) == []
    assert "never shrank" in transitions([grow])[0]
    assert "regrow" in transitions([shrink])[0]
    assert "regrow" in transitions([shrink, {**grow, "to": 4}])[0]
    # Short of all it held, but at the target the bank derives once the press is back, while the
    # decode follows it: a target read once the occupier exited, after the decode, sets nothing.
    target = lambda at, host: {"at": at, "charge": 0, "ceiling": 10, "targets": {"Experts": 16, "Host": host}}
    back = lambda host: [*SAMPLES, target(4.3, host)]
    assert transitions([shrink, {**grow, "to": 4}], back(4)) == []
    assert "regrow" in transitions([shrink, {**grow, "to": 4}], back(6))[0]
    assert transitions([shrink, {**grow, "to": 4}], [*back(4), target(4.6, 8)]) == []
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
    line = lambda store: memory.EVENT + json.dumps(
        {"at": 1.0, "message": "memory read", "store": store, "budget": 30, "guarantee": 4, "committed": 15,
         "cache": 5, "bank_target": 12, "host_target": 0, "wired_limit": 0, "wire_count": 0})
    assert memory.memory_samples([line("HostRam"), line("Gpu"), "other"], "HostRam") == [
        {"at": 1.0, "charge": 20, "ceiling": 26, "targets": {"Experts": 12, "Host": 0}}]
    wired = memory.EVENT + json.dumps(
        {"at": 1.0, "message": "memory read", "store": "HostRam", "budget": 30, "guarantee": 4, "committed": 15,
         "cache": 5, "bank_target": 12, "host_target": 0, "wired_limit": 100, "wire_count": 60})
    assert memory.memory_samples([wired], "HostRam")[0]["wired"] == (100, 60)


def test_the_machine_wires_no_more_than_its_ceiling():
    at = lambda time, count, limit=100: {"at": time, "charge": 0, "ceiling": 10, "wired": (limit, count)}
    # Held window 2..4: the wire count over the limit is a failure wherever it is read.
    assert memory.within_ceiling([at(2.5, 100), at(3, 60)], [], 2, 4) == ([], 0)
    assert memory.within_ceiling([{"at": 3, "charge": 0, "ceiling": 10}], [], 2, 4) == ([], 0)
    failures, _ = memory.within_ceiling([at(1, 101), at(3, 60), at(3.5, 130)], [], 2, 4)
    assert len(failures) == 1 and "130 bytes over its 100 wired ceiling at 2 of 3 readings, first at 1" in failures[0]


def test_accounting_counters_must_stay_zero():
    record = lambda generation, wired_limit=0, **counted: {
        "generation": generation, "wired_limit": wired_limit,
        "accounting_violations": 0, "unobserved_readings": 0, "unwired_readings": 0, **counted}
    clean = {"HostRam": [record(1, 100), record(2, 100)], "Gpu": [record(1), record(2)]}
    assert memory.uncounted(clean) == []
    # Off Metal, readings with unobserved bytes are the OS's inexact count, never a failure.
    assert memory.uncounted({"Gpu": [record(1), record(2, unobserved_readings=3)]}) == []
    failures = memory.uncounted({"HostRam": [record(1, 100), record(2, 100, accounting_violations=1),
                                             record(3, 100, accounting_violations=1, unwired_readings=2)],
                                 "Gpu": [record(1), record(2, accounting_violations=1)]})
    assert [name for name in memory.COUNTERS if any(name in failure for failure in failures)] == [
        "accounting_violations", "unwired_readings"]
    assert len(failures) == 3 and any(failure.startswith("Gpu: accounting_violations counted 1") for failure in failures)


def test_others_never_rise_on_the_load_own_shrink_beyond_what_it_freed():
    record = lambda generation, at, others: {"generation": generation, "at": at, "others": others}
    shrink = {**SHRINK, "store": "Gpu", "reading": 1, "decided": 2.5, "released": 2.9, "bytes": 80}
    records = lambda *others: {"Gpu": [record(1, 2.4, 1000), *(record(2 + i, 2.6 + 0.4 * i, value)
                                                               for i, value in enumerate(others))]}
    shrinks = lambda recorded, event=shrink: memory.others_after_shrinks([event], recorded, 2, 3.9)
    # Others unmoved, or up by what the shrink freed (the reclaim lag's most): no failure.
    assert shrinks(records(1000, 1000)) == []
    assert shrinks(records(1080, 1000)) == []
    # Up by more than the 80 it freed, or no reading to judge it by.
    assert "others rose 81 bytes over reading 1, more than the 80 it freed" in shrinks(records(1081, 1000))[0]
    assert "no reading after it" in shrinks({"Gpu": [record(1, 2.4, 1000)]})[0]
    # Out of the held pressure (the occupier still taking memory), or nothing freed: not judged.
    assert shrinks(records(1500), {**shrink, "decided": 1.0}) == []
    assert shrinks(records(1500), {**shrink, "bytes": 0}) == []


def test_pressure_rejects_missing_observations_and_windows():
    charge = physical_cell()
    shrink, grow = SHRINK, GROW
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
            "--artifacts", "artifacts", "--weight-cap", "1000",
            "--residency", "template.json", "256"]
    assert "unified-memory load" in memory.refusal(memory.parser().parse_args(base))
    assert memory.refusal(memory.parser().parse_args([*base, "--pressure-store", "device"])) is None
    assert "it takes --pressure-store" in memory.refusal(memory.parser().parse_args([*base, "--pressure-prompt"]))
    assert "takes no --context" in memory.refusal(memory.parser().parse_args(
        [*base, "--pressure-store", "device", "--context", "4096"]))


def test_owner_events_retain_resizes_yields_and_shortages():
    event = {"message": "memory grown", "at": 5, "micros": 1000, "holding": "Host",
             "from": 0, "to": 4, "grown": 40,
             **{f"{key}_before": 0 for key in ("decided", "prepared", "installed", "released")}}
    took = {"message": memory.YIELD, "at": 6, "micros": 500, "held_before": 9, "held_after": 7}
    short = {"message": memory.SHORTAGE, "at": 7, "store": "HostRam", "cause": "Room { over: 3 }"}
    grown, yielded, shortage = memory.releases([memory.EVENT + json.dumps(line) for line in (event, took, short)])
    assert (grown["resize"], grown["holding"], grown["from"], grown["to"], grown["bytes"]) == ("Growth", "Host", 0, 4, 40)
    assert (yielded["yield"], yielded["held_before"], yielded["held_after"]) == (0.5, 9, 7)
    assert (shortage["shortage"], shortage["store"]) == ("Room { over: 3 }", "HostRam")


def test_dedicated_pressure_plans_the_existing_serving_harness(monkeypatch):
    from argparse import Namespace
    from magnitude_fixtures import Template
    from magnitude_rig import templates
    from magnitude_rig.instruments import memory as instrument, harness
    monkeypatch.setattr(instrument.device, "memory", lambda host, backend: "dedicated")
    arguments = Namespace(backend="cuda", model="qwen3.6-35b-a3b:gguf:q4", weight_cap=1000,
                          context=None, residency=256, pressure_store="device",
                          pressure_prompt=False, template=Template.parse("repeat:4k"))
    plan = instrument.plan(arguments, None, None)
    assert harness.BUILD in plan.builds
    assert [name for name, job in plan.jobs] == ["measure", "memory"]
    (template,) = plan.files
    assert template == templates.file(arguments.template, 1)
    argv = plan.jobs[-1][1].argv
    assert "--pressure-store" in argv and "--context" not in argv
    assert argv[argv.index("--residency") + 1:argv.index("--residency") + 3] == (templates.argument(template), "256")


def _response(at, store="HostRam", **taken):
    """A real shortage of `store` at `at`, its response taking the `COSTLIER` steps named true."""
    return {"at": at, "shortage": "Supply", "store": store,
            **{key: taken.get(key, 0 if key == "preempt" else False) for key in memory.COSTLIER}}


def _answered(at, host):
    """The pressed store's reading that answered a shortage at `at`, the host tier's target `host`."""
    return [*SAMPLES, {"at": at, "charge": 0, "ceiling": 10, "targets": {"Experts": 16, "Host": host}}]


def test_physical_exhaustion_plateau_is_not_a_premature_release():
    """A holding whose target reached its floor has no cache left to give: a costlier response
    then (a preemption, new work waiting) is the order's next step, not one taken early."""
    charge = physical_cell()
    charge["bank"]["holdings"]["Host"]["floor"] = 1
    shrink = {**SHRINK, "to": 1}
    grow = {**GROW, "from": 1, "bytes": 70}
    for taken in ({"preempt": 2}, {"admission_waits": True}):
        assert memory.pressure_transitions(charge, [shrink, _response(3.5, **taken), grow], _answered(3.5, 1)) == []
    # Above its floor, the holding still had cache to yield.
    assert "costlier response (preempt)" in memory.pressure_transitions(
        charge, [shrink, _response(3.5, preempt=2), grow], _answered(3.5, 4))[0]


def test_costlier_release_proof_uses_the_selected_stores_shortage():
    charge = physical_cell()
    early = _answered(2.0, 8)
    # The device's shortage may preempt while host RAM's holdings remain above their floor.
    assert memory.pressure_transitions(charge, [_response(2.0, "Gpu", preempt=1), SHRINK, GROW], early) == []
    # Host RAM's own shortage preempting, or unloading, before its holdings gave their cache is early.
    for taken in ({"preempt": 1}, {"unload": True}, {"admission_waits": True}):
        assert "before the pressed store's holdings reached their floor" in memory.pressure_transitions(
            charge, [_response(2.0, **taken), SHRINK, GROW], early)[0]
    # A shortage the cache answers alone takes nothing costlier.
    assert memory.pressure_transitions(charge, [_response(2.0), SHRINK, GROW], early) == []
    # A shortage whose record lacks its response, or no reading answering it, leaves the order unverified.
    for unknown in ({**_response(2.0, preempt=1), "preempt": None}, {**_response(2.0), "unload": None}):
        assert "unverified" in memory.pressure_transitions(charge, [unknown, SHRINK, GROW], early)[0]
    assert "unverified" in memory.pressure_transitions(charge, [_response(9.0, preempt=1), SHRINK, GROW], early)[0]


def _read(at, generation, store="HostRam", **terms):
    return memory.EVENT + json.dumps({"at": at, "message": "memory read", "store": store, "generation": generation,
                                      "budget": 30, "guarantee": 4, "committed": 15, "cache": 5, "others": 10,
                                      "bank_target": 12, "host_target": 0, **terms})


def _grown(at, reading, term, before=None, store="HostRam"):
    return memory.EVENT + json.dumps({"message": "memory grown", "at": at, "micros": 1000, "holding": "Host",
                                      "from": 0, "to": 4, "grown": 40, "store": store, "reading": reading,
                                      "term": term, **({} if before is None else {"before_generation": before}),
                                      **{f"{key}_before": 0 for key in
                                         ("decided", "prepared", "installed", "released")}})


def test_each_resize_is_matched_to_the_term_its_reading_moved():
    """A resize follows a target derived afresh at a reading, so it traces to that reading's record
    and a term there that moved since the reading the store's previous resize followed: never the
    reading just before it. A store's first resize is the load's own, named `load`."""
    lines = [_read(1, 7), _read(1.5, 8, store="Gpu", others=99), _read(2, 9, others=10), _read(3, 10, others=6)]
    check = lambda *resizes: memory.untraced(
        [event for event in memory.releases([*lines, *resizes]) if "resize" in event], memory.readings(lines))
    load = _grown(1.1, 7, "load")
    assert check(load) == []
    # Others moved from 10 at reading 7 to 6 at reading 10, though not across the reading before (9).
    assert check(load, _grown(3.1, 10, "others", before=7)) == []
    assert "held at 15 from reading 7 to 10" in check(load, _grown(3.1, 10, "committed", before=7))[0]
    assert "held at 10 from reading 7 to 9" in check(load, _grown(2.1, 9, "others", before=7))[0]
    assert "no record holds 8" in check(load, _grown(3.1, 10, "others", before=8))[0]
    assert "do not carry" in check(load, _grown(3.1, 10, "harm", before=7))[0]
    # `load` only for a store's first resize, and a first resize is only `load`.
    assert "an earlier resize of HostRam" in check(load, _grown(3.1, 10, "load"))[0]
    assert "an earlier resize of HostRam" in check(load, _grown(3.1, 10, "others"))[0]
    assert "first resize of HostRam but names others" in check(_grown(3.1, 10, "others", before=7))[0]
    assert check(_grown(1.1, 7, "load", before=6)) == [], "a first resize names whatever reading came before it"
    # Each store has its own first resize.
    assert check(load, _grown(1.6, 8, "load", store="Gpu")) == []
    untraced = json.loads(_grown(3.1, 10, "others", before=7)[len(memory.EVENT):])
    del untraced["term"], untraced["reading"]
    assert "names no reading, term" in check(memory.EVENT + json.dumps(untraced))[0]


def test_steady_decode_makes_no_resize(capsys):
    """Before the press the load decodes beside the occupier's approach, which leaves it its
    guarantee: any resize decided there is one steady decode made."""
    pressure = {"steps": decode(), "started": {"at": 1.0}}
    assert memory.steady_resizes(pressure, [resize(1.2), resize(0.0)]) == []
    assert "2 resizes in steady decode" in memory.steady_resizes(pressure, [resize(0.1), resize(0.9), resize(1.0)])[0]
    assert "no steady decode" in memory.steady_resizes({"steps": []}, [])[0]
    capsys.readouterr()


def test_a_moment_reads_from_the_harness_event_and_the_engine_log_line():
    record = {"moment": "loaded", "footprint_bytes": 7}
    event = memory.EVENT + json.dumps({"at": 1.0, "message": "load memory", "record": json.dumps(record)})
    logged = "2026-11-03T00:00:00Z  INFO magnitude_engine::execution: load memory record=" + json.dumps(record)
    other = memory.EVENT + json.dumps({"at": 1.0, "message": "memory read"})
    assert memory.moments([(1.0, event), (2.0, other)]) == {"loaded": (1.0, record)}
    assert memory.moments([(3.0, logged), (4.0, "engine: other")]) == {"loaded": (3.0, record)}
