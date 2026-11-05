"""`mrig gate`: which topologies the hosts' recorded inventory gates and where each runs, how each
is estimated from recorded step runs, what each one's line says, and the AND of them all."""

import json
import time
import types
from pathlib import Path

import pytest
from test_instruments import record_topology

from magnitude_rig import engine, gate, state, store
from magnitude_rig.cli import parser
from magnitude_rig.connection import HostError
from magnitude_rig.hosts import HostConfig, Hosts
from magnitude_rig.instruments import JobResult

CUDA, VULKAN, METAL = (gate.Topology("cuda", "dedicated"), gate.Topology("vulkan", "unified"),
                       gate.Topology("metal", "unified"))


def hosts(*aliases: str) -> Hosts:
    return Hosts(HostConfig("", "~/r", "~/m"), {}, {alias: HostConfig(alias, "~/r", "~/m") for alias in aliases})


def record_gpus(results: Path, run: str, host: str, gpus: dict[str, str]) -> None:
    """A `mrig device` run of `host` whose topology states each backend's GPU memory beside the
    host CPU's, as `record_topology` states one."""
    record_topology(results, run, host, "cpu", "Host")
    log = results / run / "logs" / "job-topology.log"
    document = json.loads(log.read_text().splitlines()[-1])
    for index, (backend, memory) in enumerate(gpus.items()):
        identity = {"Uuid": [index]}
        document["topology"]["physical"].append({"identity": identity, "memory": memory})
        document["topology"]["executors"].append({"selector": {backend.capitalize(): {"uuid": [index]}},
                                                  "physical": identity})
    log.write_text(json.dumps(document) + "\n")


def test_the_gate_runs_each_recorded_topology_once_among_its_hosts(tmp_path, monkeypatch):
    """Topologies come from the listed hosts' latest recorded topologies, the CPU backend not
    gated: hosts of one topology share its run, a new host joins once recorded, and a host of
    another topology refuses a topology's plan. A host with none recorded is information beside
    the gate, never part of it: reachable, with its likely topology from its platform and who
    covers it; unreachable, saying so."""
    monkeypatch.setattr(store, "results", lambda: tmp_path)
    record_gpus(tmp_path, "20261101T000000Z-device-000001", "rtx-a", {"cuda": "Own", "vulkan": "Own"})
    record_gpus(tmp_path, "20261101T000000Z-device-000002", "rtx-b", {"cuda": "Own"})
    record_gpus(tmp_path, "20261101T000000Z-device-000003", "gb10", {"cuda": "Host", "vulkan": "Host"})
    record_gpus(tmp_path, "20261101T000000Z-device-000004", "mac", {"metal": "Host"})
    record_gpus(tmp_path, "20261101T000000Z-device-000005", "unlisted", {"metal": "Own"})  # not a listed host
    record_gpus(tmp_path, "20261101T000000Z-device-000006", "mixed", {"vulkan": "Own"})
    record_gpus(tmp_path, "20261102T000000Z-device-000007", "mixed", {"vulkan": "Own", "cuda": "Own"})
    record_gpus(tmp_path, "20261101T000000Z-device-000008", "split", {"cuda": "Own", "vulkan": "Own"})
    log = tmp_path / "20261101T000000Z-device-000008" / "logs" / "job-topology.log"
    document = json.loads(log.read_text())
    document["topology"]["physical"].append({"identity": {"Uuid": [9]}, "memory": "Host"})
    document["topology"]["executors"].append({"selector": {"Vulkan": {"uuid": [9]}}, "physical": {"Uuid": [9]}})
    log.write_text(json.dumps(document) + "\n")
    platforms = {"fresh": ("Linux", "x86_64", ("cpu", "cuda", "vulkan")), "mini": ("Darwin", "arm64", ("cpu", "metal")),
                 "arm": ("Linux", "aarch64", ("cuda",))}

    def connecting(config):
        if config.alias not in platforms:
            raise HostError(f"{config.alias}: unreachable")
        system, machine, backends = platforms[config.alias]
        return types.SimpleNamespace(system=system, machine=machine, backends=backends)

    monkeypatch.setattr(gate, "connect", connecting)
    found = gate.inventory(hosts("rtx-a", "rtx-b", "gb10", "mac", "mixed", "split", "fresh", "mini", "arm", "away"))
    assert found.topologies == {
        CUDA: ("rtx-a", "rtx-b", "mixed", "split"), gate.Topology("cuda", "unified"): ("gb10",),
        METAL: ("mac",), gate.Topology("vulkan", "dedicated"): ("rtx-a", "mixed"), VULKAN: ("gb10",),
    }
    # A host whose devices of one backend disagree on their memory fails the gate for it.
    assert found.disagreeing == (
        ("vulkan", "split's topology (20261101T000000Z-device-000008) states vulkan devices of memory "
                   "['Host', 'Own'], not one"),)
    assert list(found.unrecorded) == [
        "fresh (likely cuda/dedicated, covered by rtx-a, rtx-b, mixed, split; vulkan/dedicated, covered by rtx-a, mixed)",
        "mini (likely metal/unified, covered by mac)",
        "arm (likely cuda/?, no other host)",
        "away (unreachable: away: unreachable)",
    ]
    assert gate.unrecorded_line(found.unrecorded[1:2]) == (
        "no topology recorded (information, not part of the verdict): mini (likely metal/unified, covered by mac); "
        "mrig device --host HOST records one")

    # A topology's run is placed only among its own hosts.
    work = gate._work(CUDA, source=None)
    assert work("rtx-b").device == "device" and [job.name for job in work("rtx-b").jobs] == \
        ["step", "trim", "logits", "finite"]
    with pytest.raises(engine.Refused, match="cuda device is unified, not dedicated"):
        work("gb10")
    with pytest.raises(engine.Refused, match="run mrig device --host fresh first"):
        work("fresh")


def test_each_topology_streams_a_capped_load_at_the_fewest_steps_and_checks_its_logits():
    timed, trim, logits, finite = (step for _, step in gate.jobs("vulkan"))
    argv = list(timed.argv)
    option = lambda flag: argv[argv.index(flag) + 1]
    assert timed.fresh and logits.fresh  # each a streamed load, its experts read from storage
    assert option("--weight-cap") == str(gate.WEIGHT_CAP) and option("--contexts") == str(gate.CONTEXT)
    assert option("--device") == "vulkan" and option("--steps") == "2" and option("--warm") == "0"
    assert option("--prefill-samples") == "2" and option("--attribution-steps") == "0"
    assert option("--lookahead") == "on" and "--untimed" in argv and "--tuning-defaults" in argv
    assert "{runs}/trim/rendered.jsonl" in logits.argv and "--tuning-defaults" in logits.argv
    assert finite.argv[-1] == "{runs}/logits/logits.f32"


def test_the_host_checks_cut_the_rendering_and_find_a_non_finite_logit(tmp_path):
    import array
    import subprocess
    import sys

    rendered = tmp_path / "rendered.jsonl"
    rendered.write_text(json.dumps({"fixture_id": "a", "scored_spans": [[100, 300], [400, 500]]}) + "\n"
                        + json.dumps({"fixture_id": "b", "scored_spans": [[50, 60]]}) + "\n")
    cut = tmp_path / "cut.jsonl"
    subprocess.run([sys.executable, "-c", gate.TRIM, str(rendered), str(cut), "4"], check=True)
    assert [json.loads(line) for line in cut.read_text().splitlines()] == \
        [{"fixture_id": "a", "scored_spans": [[100, 104]]}]

    def check(values) -> subprocess.CompletedProcess:
        path = tmp_path / "logits.f32"
        path.write_bytes(array.array("f", values).tobytes())
        return subprocess.run([sys.executable, "-c", gate.FINITE, str(path)], capture_output=True, text=True)

    assert check([0.5, -3.0, 7.0]).returncode == 0
    for poison in (float("nan"), float("inf")):
        failed = check([0.5, poison, 1.0, poison])
        assert failed.returncode == 1 and "non-finite logits: 2 of 4 values, the first at value 1" in failed.stderr
    assert check([]).returncode == 1


def test_the_estimate_past_the_build_is_each_hosts_latest_streamed_step_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "results", lambda: tmp_path)

    def step_run(run: str, host: str, backend: str, phases: dict, codes: dict, failed=None):
        (tmp_path / run).mkdir()
        (tmp_path / run / "rig.json").write_text(json.dumps({
            "host": host, "backend": backend, "failed": failed, "codes": codes,
            "phases": [{"phase": name, "seconds": seconds} for name, seconds in phases.items()]}))

    own = {"stage": 5, "install-source": 1, "build-source": 300, "models": 1, "queue-timed": 900,
           "gate": 180, "job-measure": 7, "job-step": 140, "fetch": 3}
    step_run("20261101T000000Z-step-000001", "rtx-a", "cuda", {**own, "job-streamed-7": 60}, {"streamed-7": 0})
    step_run("20261102T000000Z-step-000002", "rtx-a", "cuda", {**own, "job-streamed-7": 50}, {"streamed-7": 1})
    step_run("20261102T000000Z-step-000003", "rtx-b", "cuda", {**own, "build-source": 100, "job-streamed-7": 40},
             {"streamed-7": 0})
    step_run("20261103T000000Z-step-000004", "rtx-b", "cuda", own, {"step": 0})  # no streamed cell
    step_run("20261103T000000Z-step-000005", "rtx-b", "vulkan", {**own, "job-streamed-7": 900}, {"streamed-7": 0})
    step_run("20261103T000000Z-step-000006", "rtx-b", "cuda", {**own, "job-streamed-7": 1}, {"streamed-7": 0},
             failed={"phase": "build-source"})
    # Past their builds and waits: rtx-a's latest run's cell failed, so its earlier one stands.
    assert gate.after_build(CUDA, ("rtx-a", "rtx-b")) == {"rtx-a": 10 + 2 * 60, "rtx-b": 10 + 2 * 40}
    estimate = gate.estimate(CUDA, ("rtx-a", "rtx-b", "rtx-c"))
    assert estimate("rtx-b", 1500.0) == 1500 + 90  # its start where placed, then its own record
    assert estimate("rtx-c", 1500.0) == 1500 + 130  # a host with none: the most of its topology's
    assert gate.estimate(CUDA, ("rtx-c",)) is None


def test_the_gate_passes_only_when_every_topology_passed():
    assert gate.verdict([0, 0, 0, 0, 0]) == 0
    assert gate.verdict([0, 2, 0]) == 2  # a topology answered no
    assert gate.verdict([0, 2, 1]) == 1  # one has no answer at all (a failed build, no host)
    assert gate.verdict([]) == 1  # nothing checked passes nothing
    # The verdict's words are over recorded topologies alone; hosts with none recorded never enter it.
    assert gate.verdict_line([0, 0, 0, 0, 0]) == "gate: PASS · every recorded topology passed (5 of 5)"
    assert gate.verdict_line([0, 2, 0]) == "gate: FAIL · 2 of 3 recorded topologies passed"


def test_a_run_past_its_estimate_is_running_and_followed_to_its_result(tmp_path, monkeypatch, capsys):
    """A topology's run past its estimate is said to be running, and how far past, as it
    happens; it is followed on to its end, and its result is its answer, never a failure."""
    from test_engine import _fake_run

    directory, child = _fake_run(tmp_path, 3)
    started = time.time()
    monkeypatch.setattr(state, "run_state", lambda run: {"started": started, "estimate": 1.0})
    code, transcript = gate.followed(CUDA, directory.name, directory, child)
    child.wait()
    running = capsys.readouterr().out
    assert running.startswith("cuda/dedicated    RUNNING past its 1s estimate by ") and directory.name in running
    assert code == 0 and "the answer" in transcript and gate.headline(code, transcript, directory.name) == "PASS"


def test_each_topologys_line_is_pass_or_its_first_error(tmp_path):
    answered = ("mrig gate · cuda → rtx-a · ...\npurpose: p\n"
                "step exited 1: magnitude-harness: graph binding aliases another\n  log: x\n7m of 9m\n")
    failed = "mrig gate · ...\nFAILED in build-source: `cargo` exited 101\n  src/a.rs:3:1: error: oops\nlog: l\n"
    assert gate.headline(0, "purpose: p\nanything\n", "r") == "PASS"
    assert gate.headline(2, answered, "r") == "step exited 1: magnitude-harness: graph binding aliases another"
    assert gate.headline(1, failed, "r") == "FAILED in build-source: `cargo` exited 101 · src/a.rs:3:1: error: oops"
    assert gate.headline(1, "mrig: no cuda host among any host: none runs cuda\n", "r").startswith("mrig: no cuda host")

    def job(name: str, code: int, *lines: str) -> JobResult:
        log = tmp_path / f"{name}.log"
        log.write_text("".join(f"{line}\n" for line in ("rig: $ harness", *lines)))
        return JobResult(tmp_path / name, log, code)

    arguments = parser().parse_args(["gate", "--topology", "cuda/dedicated", "--purpose", "p"])
    panicked = gate.answer(arguments, {
        "step": job("step", 101, "[stderr] thread 'main' panicked at src/plan.rs:9:5:", "[stderr] bindings alias",
                    "[stderr] note: run with RUST_BACKTRACE=1"),
        "trim": job("trim", 1, "FileNotFoundError"), "logits": job("logits", 1), "finite": job("finite", 1)})
    assert not panicked.ok
    assert panicked.lines[0] == "step exited 101: thread 'main' panicked at src/plan.rs:9:5: bindings alias"
    passed = gate.answer(arguments, {name: job(name, 0) for name in ("step", "trim", "logits")}
                         | {"finite": job("finite", 0, "gate: logits finite: 600000 values")})
    assert passed.ok and passed.lines[0].endswith("logits finite: 600000 values")
