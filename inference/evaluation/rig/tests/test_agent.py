"""The host agent: which holds conflict, arrival order, the hold order, eviction and the queue
estimates other runs see (its sessions in this process); and its runners, as real detached
processes: a run's plan carried out to its end whatever happens to the connections that follow
it, stopped only when asked, and kept until fetched."""

import errno
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import waiting
from magnitude_rig import engine
from magnitude_rig.host import agent
from magnitude_rig.host.agent import checkout_bytes

AGENT = Path(agent.__file__)


# Sessions: holds, judged in this process ---------------------------------------------------


def session(root: Path, run: str, *, estimate=60.0) -> agent.Session:
    return agent.Session(agent.Rig(str(root)), {
        "who": "tester", "run": run, "purpose": f"purpose of {run}", "estimate": estimate,
        "started": time.time(), "protocol": agent.PROTOCOL,
    })


def hold(current: agent.Session, kind: str) -> None:
    current.request(kind)
    assert current.blocking(kind) == [], f"{kind} is blocked"


def blocked(current: agent.Session, kind: str) -> list:
    current.request(kind)
    return [b["run"] for b in current.blocking(kind)]


def end(current: agent.Session) -> None:
    current.release_all()
    current.run_lock.close()


BUILD = "b"  # every test run's build (`engine.build_key`)


def status(root: Path, *sources: str) -> dict:
    """The host's status for a run of `sources`, each of tree `<source>-tree`."""
    argv = [sys.executable, str(AGENT), "status", "--root", str(root)]
    for source in sources:
        argv += ["--source", source, "--tree", source + "-tree", "--build", BUILD]
    return json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)


def test_builds_share_the_host_and_devices_take_turns(tmp_path):
    first, second = session(tmp_path, "first"), session(tmp_path, "second")
    hold(first, "build")
    hold(second, "build")
    for current in (first, second):
        current.release("build")
    hold(first, "device")
    assert blocked(second, "device") == ["first"]
    end(first)
    assert second.blocking("device") == []
    end(second)


def test_a_measurement_waits_for_builds_and_new_builds_wait_behind_it(tmp_path):
    builder, measurer, later = (session(tmp_path, run) for run in ("builder", "measurer", "later"))
    hold(builder, "build")
    assert blocked(measurer, "timed") == ["builder"]
    # Arrival order: a build requested after the waiting measurement waits behind it.
    assert blocked(later, "build") == ["measurer"]
    end(builder)
    assert measurer.blocking("timed") == []
    end(measurer)
    assert later.blocking("build") == []
    end(later)


def test_a_measurement_shares_its_host_with_no_installation_or_acquisition(tmp_path):
    installer, models, measurer = (session(tmp_path, run) for run in ("installer", "models", "measurer"))
    hold(installer, "install")
    hold(models, "models")
    assert sorted(blocked(measurer, "timed")) == ["installer", "models"]
    end(installer)
    end(models)
    assert measurer.blocking("timed") == []
    later_installer, later_models = session(tmp_path, "later-installer"), session(tmp_path, "later-models")
    assert blocked(later_installer, "install") == ["measurer"]
    assert blocked(later_models, "models") == ["measurer"]
    end(measurer)
    assert later_installer.blocking("install") == [] and later_models.blocking("models") == []
    end(later_installer)
    end(later_models)


def measured(root: Path, **record: int) -> None:
    """Record what builds of `BUILD` measured on the host."""
    (root / "rig" / agent.MEASURED).mkdir(parents=True, exist_ok=True)
    (root / "rig" / agent.MEASURED / f"{BUILD}.json").write_text(json.dumps(record))


def claim(current: agent.Session, source: str, fallback: int = 0, *, commit: str = "c0", seconds: float = 60,
          shares: bool = False) -> str:
    """Claim a checkout for a run of `source`, of tree `<source>-tree` and build `BUILD`."""
    name, shared, others = current.claim(source, source + "-tree", BUILD, fallback, commit, seconds)
    assert others == [] and shared == shares
    return name


def keeps(checkout: Path, *builds: str) -> None:
    """Record that `checkout` keeps `builds`, referencing no unit."""
    (checkout / agent.REFERENCES).write_text(json.dumps({build: {"at": time.time(), "units": []} for build in builds}))


def test_each_run_claims_a_checkout_of_its_own_last_built_from_its_source_if_free(tmp_path):
    """Runs never share a checkout: a run takes a free one last installed from its source, else
    the most recently used free one, else a new one; runs install at once."""
    base = tmp_path / "checkouts"
    first, second = session(tmp_path, "first"), session(tmp_path, "second")
    assert claim(first, "a") == "slot-0" and claim(second, "a") == "slot-1"  # slot-0 is first's
    assert (base / "slot-0" / agent.SOURCE).read_text() == "a" and (base / "slot-0" / "used").exists()
    hold(first, "install")
    hold(second, "install")
    end(first)
    end(second)
    os.utime(base / "slot-0" / agent.SOURCE, (1000, 1000))  # claimed before slot-1
    (base / "unclaimed").mkdir()  # a directory no run claimed: never preferred
    other, again = session(tmp_path, "other"), session(tmp_path, "again")
    assert status(tmp_path, "b", "a")["checkouts"] == [
        {"name": "slot-1", "source": "a", "size": 0, "warmth": agent.OTHER, "commit": None, "built": False,
         "shared": False},
        {"name": "slot-0", "source": "a", "size": 0, "warmth": agent.SAME, "commit": None, "built": False,
         "shared": False},
    ]
    assert claim(other, "b") == "slot-1"  # the most recently used: no free one was built from b
    assert claim(again, "a") == "slot-0"  # the one built from a, though used less recently
    assert status(tmp_path, "a")["checkouts"][0]["name"] == "unclaimed"  # the last one free
    end(other)
    end(again)


def test_runs_of_one_tree_and_build_share_its_checkout_once_it_is_installed(tmp_path):
    """A run shares the checkout live runs of its tree and build hold once they have installed
    it, and owes the host nothing more than their build; a run of another tree or build, its own
    other side, or a run arriving before the install, claims its own."""
    first, early, same, other_build = (session(tmp_path, run) for run in ("first", "early", "same", "other-build"))
    assert claim(first, "a") == "slot-0"
    assert claim(early, "a") == "slot-1"  # slot-0 is not installed yet
    first.installed("slot-0")
    assert (tmp_path / "checkouts" / "slot-0" / agent.TREE).read_text() == "a-tree"
    found = status(tmp_path, "a")["checkouts"][0]
    assert found["name"] == "slot-0" and found["shared"] and found["owed"] == 60 and status(tmp_path, "a")["needs"] == [0]
    assert claim(same, "a", shares=True) == "slot-0"
    name, shared, _ = other_build.claim("a", "a-tree", "another build", 0, "c0", 60)
    assert name == "slot-2" and not shared
    name, shared, _ = first.claim("a", "a-tree", BUILD, 0, "c0", 60)  # its own second side
    assert name == "slot-3" and not shared
    # A sharer waits on its holders' build of this checkout alone, not on their other sides'.
    assert status(tmp_path, "a")["checkouts"][0]["owed"] == 60
    # A run whose sides are one tree and build shares on one side only, as its claims would.
    found = status(tmp_path, "a", "a")["checkouts"]
    assert found[0]["shared"] and found[1] is None  # the second claims one of its own: none is free
    # A sharer owes no build: never counted among the host's builds, for room or for CPU.
    assert "checkout:slot-0" not in same.builds
    (entry,) = [e for e in status(tmp_path)["entries"] if e["run"] == "same"]
    assert "build" not in entry and "need" not in entry and "seconds" not in entry
    end(early), end(other_build)
    assert status(tmp_path)["building"] == [120]  # first's two sides, never the sharer's
    same.shared_built("slot-0")  # waited on first's build, found every unit fresh: records nothing
    assert not (tmp_path / "rig" / agent.MEASURED).exists()
    assert not (tmp_path / "checkouts" / "slot-0" / agent.REFERENCES).exists()
    for current in (first, same):
        end(current)


def test_a_free_checkout_is_chosen_by_what_it_keeps_then_its_tree(tmp_path):
    """Among free checkouts, one that keeps the build and holds the tree, then one that keeps
    the build, then one that holds the tree, then one of the source."""
    base = tmp_path / "checkouts"
    for name in ("source", "tree", "kept", "both"):
        (base / name).mkdir(parents=True)
    (base / "source" / agent.SOURCE).write_text("a")
    for name in ("tree", "both"):
        (base / name / agent.TREE).write_text("a-tree")
    for name in ("kept", "both"):
        keeps(base / name, BUILD)
    runs = [session(tmp_path, "run-%d" % index) for index in range(4)]
    assert [claim(current, "a") for current in runs] == ["both", "kept", "tree", "source"]
    for current in runs:
        end(current)


def test_a_build_keeps_the_units_its_kept_builds_reference_and_trims_the_rest(tmp_path):
    """After a build, the target keeps each unit a kept build references, removes every other
    hashed artifact of a unit directory, and leaves what it cannot tie to a unit (a profile's own
    outputs) and every incremental directory, however long since a build compiled into it (a
    fresh unit writes nothing there)."""
    checkout = tmp_path / "checkouts" / "slot-0"
    current = session(tmp_path, "run")
    assert claim(current, "a") == "slot-0"
    target = checkout / "target"
    keep, stale, other = "0123456789abcdef", "fedcba9876543210", "00112233445566ff"
    files = [f"release/deps/libfoo-{keep}.rlib", f"release/deps/foo-{keep}.d", f"release/deps/foo-{keep}.provenance.json",
             f"release/deps/libfoo-{stale}.rlib", f"release/build/foo-{keep}/out/x", f"release/build/foo-{stale}/out/x",
             f"debug/deps/libbar-{other}.rmeta", "release/magnitude-inference", f"release/.fingerprint/foo-{stale}/x",
             "debug/incremental/old-1/s", "debug/incremental/new-1/s"]
    for relative in files:
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_bytes(b"\0" * 10)
    os.utime(target / "debug" / "incremental" / "old-1", (1000, 1000))
    (checkout / agent.REFERENCES).write_text("{}")  # it has recorded builds: none it keeps
    units = agent.units_of(str(target), [str(target / f"release/deps/libfoo-{keep}.rlib"),
                                         str(target / f"release/build/foo-{keep}/out"), str(target / "release" / "x")])
    assert units == {f"release/deps/{keep}", f"release/build/{keep}"}
    built = current.built("slot-0", 1.0, units)
    left = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    assert left == sorted(["debug/incremental/new-1/s", "debug/incremental/old-1/s",
                           "release/.fingerprint/foo-%s/x" % stale, f"release/build/foo-{keep}/out/x",
                           f"release/deps/foo-{keep}.d", f"release/deps/foo-{keep}.provenance.json",
                           f"release/deps/libfoo-{keep}.rlib", "release/magnitude-inference"])
    assert built["trimmed"] == {"bytes": 30, "entries": 3}
    assert BUILD in json.loads((checkout / agent.REFERENCES).read_text())
    end(current)


def test_a_checkout_that_recorded_no_build_keeps_what_it_held_as_one_more_build(tmp_path):
    """The first build a checkout records keeps the units the checkout already held."""
    checkout = tmp_path / "checkouts" / "slot-0"
    current = session(tmp_path, "run")
    assert claim(current, "a") == "slot-0"
    held, mine = "0123456789abcdef", "fedcba9876543210"
    for name in (f"libheld-{held}.rlib", f"libmine-{mine}.rlib"):
        (checkout / "target" / "release" / "deps").mkdir(parents=True, exist_ok=True)
        (checkout / "target" / "release" / "deps" / name).write_text("x")
    (checkout / "target" / "debug" / "incremental" / "held-1").mkdir(parents=True)
    os.utime(checkout / "target" / "debug" / "incremental" / "held-1", (1000, 1000))
    assert current.built("slot-0", 1.0, {f"release/deps/{mine}"})["trimmed"]["bytes"] == 0  # incremental kept too
    kept = json.loads((checkout / agent.REFERENCES).read_text())
    assert kept[agent.UNRECORDED]["units"] == [f"release/deps/{held}"] and kept[BUILD]["units"] == [f"release/deps/{mine}"]
    end(current)


def test_what_a_checkout_held_before_recording_goes_once_builds_kept_were_recorded(tmp_path):
    """The units held before a checkout's first record outlive `BUILDS_KEPT` recorded builds,
    however few kinds of build they are, and then go."""
    checkout = tmp_path / "checkouts" / "slot-0"
    current = session(tmp_path, "run")
    assert claim(current, "a") == "slot-0"
    held, mine = "0123456789abcdef", "fedcba9876543210"
    deps = checkout / "target" / "release" / "deps"
    deps.mkdir(parents=True)
    for name in (f"libheld-{held}.rlib", f"libmine-{mine}.rlib"):
        (deps / name).write_text("x")
    for _ in range(agent.BUILDS_KEPT):
        assert (deps / f"libheld-{held}.rlib").exists()
        current.built("slot-0", 1.0, {f"release/deps/{mine}"})
        current.builds["checkout:slot-0"] = agent.Owed(agent.SAME, BUILD, 0, 0, "c0", 60)
    assert agent.UNRECORDED not in json.loads((checkout / agent.REFERENCES).read_text())
    assert sorted(p.name for p in deps.iterdir()) == [f"libmine-{mine}.rlib"]
    end(current)


def test_a_build_measures_the_room_it_needed_before_its_trim(tmp_path):
    """What a build added is what it wrote before its trim removed what no kept build
    references; the checkout's size is what is left."""
    checkout = tmp_path / "checkouts" / "slot-0"
    current = session(tmp_path, "run")
    assert claim(current, "a") == "slot-0"  # a cold checkout
    deps = checkout / "target" / "release" / "deps"
    deps.mkdir(parents=True)
    stale, mine = "0123456789abcdef", "fedcba9876543210"
    (deps / f"libstale-{stale}.rlib").write_bytes(b"\0" * 300)
    (deps / f"libmine-{mine}.rlib").write_bytes(b"\0" * 200)
    (checkout / agent.REFERENCES).write_text("{}")  # it keeps no build: the stale unit goes
    built = current.built("slot-0", 1.0, {f"release/deps/{mine}"})
    assert built["trimmed"]["bytes"] == 300 and built["bytes"] == int((checkout / "size").read_text())
    assert built["added_bytes"] == built["bytes"] + 300
    assert json.loads((tmp_path / "rig" / agent.MEASURED / f"{BUILD}.json").read_text())["size"] == built["added_bytes"]
    end(current)


def test_a_lock_another_holds_reads_as_held(tmp_path):
    path = str(tmp_path / "lock")
    holder, other = agent.FileLock(path), agent.FileLock(path)
    try:
        assert holder.try_acquire()
        assert not other.try_acquire()
    finally:
        holder.close()
        other.close()


@pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX's lock")
def test_a_lock_failure_other_than_contention_raises(tmp_path, monkeypatch):
    import fcntl

    def refuse(fd, operation):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    lock = agent.FileLock(str(tmp_path / "lock"))
    try:
        with pytest.raises(OSError) as raised:
            lock.try_acquire()
        assert raised.value.errno == errno.ENOLCK
    finally:
        lock.close()


def test_device_holds_take_the_hosts_device_lock(tmp_path):
    current = session(tmp_path, "run")
    hold(current, "device")
    probe = subprocess.run(
        [sys.executable, "-c",
         "import fcntl, sys; f = open(sys.argv[1], 'a+b'); "
         "fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", str(tmp_path / "lock")],
        capture_output=True,
    )
    assert probe.returncode != 0
    end(current)


def test_status_estimates_the_wait_from_the_holders_own_estimates(tmp_path):
    holder = session(tmp_path, "holder", estimate=600)
    hold(holder, "device")
    report = status(tmp_path, "mine")
    (entry,) = report["entries"]
    assert entry["purpose"] == "purpose of holder" and entry["state"] == "held"
    assert 590 < report["waits"]["device"] <= 600 and report["waits"]["build"] == 0
    assert report["checkouts"] == [None]
    end(holder)


def test_a_run_past_its_estimate_is_priced_as_long_again_as_it_has_taken(tmp_path):
    """A timed run 17 minutes into its 9-minute estimate is not priced as about to end. Past its
    estimate, its estimate says nothing of its end: it is expected to take as long again as it
    has, more than its overrun."""
    holder = session(tmp_path, "holder", estimate=540)
    holder.info["started"] -= 17 * 60
    hold(holder, "timed")
    report = status(tmp_path)
    (entry,) = report["entries"]
    assert 8 * 60 <= entry["over"] < 8 * 60 + 10
    assert 17 * 60 <= entry["remaining"] < 17 * 60 + 10
    assert report["waits"]["device"] == report["waits"]["build"] == entry["remaining"]
    end(holder)


def test_the_wait_counts_every_run_ahead_once_and_a_queued_run_by_its_work_not_its_wait(tmp_path, monkeypatch):
    """A new device request waits behind the holder and every run queued ahead of it. A queued
    run's progress stops while it waits, so however long it has queued it still owes the work
    its estimate leaves; and a run is counted once, however many entries it has."""
    holder = session(tmp_path, "holder", estimate=600)
    hold(holder, "device")
    queued = session(tmp_path, "queued", estimate=600)
    began = time.time()
    queued.info["started"] = began - 1000  # 10 s of work, then queued for the device 990 s ago
    monkeypatch.setattr(agent.time, "time", lambda: began - 990)
    assert blocked(queued, "device") == ["holder"]
    monkeypatch.undo()
    other = session(tmp_path, "other", estimate=300)  # a run of another protocol, with two entries
    other.info["protocol"] = agent.PROTOCOL + 1
    hold(other, "build")
    hold(other, "models")
    report = status(tmp_path)
    left = {e["run"]: e["remaining"] for e in report["entries"]}
    (waiting,) = [e for e in report["entries"] if e["run"] == "queued"]
    assert 990 <= waiting["queued"] <= 991 and 9 <= waiting["held"] <= 10 + 1  # queued apart from its work
    assert 590 <= left["queued"] <= 590 + 1 and 590 < left["holder"] <= 600 and 290 < left["other"] <= 300
    assert report["waits"]["device"] == pytest.approx(left["holder"] + left["queued"] + left["other"])
    assert [e["state"] for e in report["entries"] if e["run"] == "queued"] == ["waiting"]
    end(holder)
    assert queued.blocking("device") == []  # its wait ends: its progress runs again from here
    assert json.loads((tmp_path / "runs" / "queued" / agent.WAITED).read_text())["since"] is None
    for current in (queued, other):
        end(current)


def test_status_reports_the_builds_live_runs_owe_and_a_build_records_its_seconds(tmp_path):
    """A run staged but holding nothing owes the slowest build measured on the host (the prior
    while none is); one holding its checkout owes what its caller priced that checkout's build
    at, less what it has built; a run whose build ended owes none. The build is recorded with
    the commits it built from and to, and the checkout reports the commit it built and that it
    has built that build."""
    (tmp_path / "checkouts" / "a").mkdir(parents=True)
    (tmp_path / "checkouts" / "a" / "source").write_text("source")
    (tmp_path / "checkouts" / "a" / agent.COMMIT).write_text("c0")
    staged, warm = session(tmp_path, "staged"), session(tmp_path, "warm")
    report = status(tmp_path)
    assert report["slowest"] == agent.BUILD_PRIOR_SECONDS and report["building"] == [1200, 1200]
    assert claim(warm, "source", commit="c1", seconds=60) == "a"
    assert sorted(status(tmp_path)["building"]) == [60, 1200]
    hold(warm, "build")
    time.sleep(1.1)
    owed = sorted(status(tmp_path)["building"])
    assert 58 <= owed[0] < 59 and owed[1] == 1200
    warm.built("a", 42.4, set())
    report = status(tmp_path, "source")
    assert report["building"] == [42] and report["slowest"] == 42  # the staged run, by the build measured
    assert report["measured"][BUILD]["builds"] == [{"seconds": 42, "from": "c0", "to": "c1", "built": False}]
    end(warm)
    report = status(tmp_path, "source")
    assert report["checkouts"][0]["commit"] == "c1" and report["checkouts"][0]["built"]
    end(staged)
    assert status(tmp_path)["building"] == []


def _never_fits(root: Path) -> None:
    """Record a build larger than any volume holds: eviction removes all that may go."""
    measured(root, size=10**18)


def test_eviction_removes_the_least_recently_used_unheld_checkouts(tmp_path):
    base = tmp_path / "checkouts"
    for index, key in enumerate(("old", "held", "new")):
        (base / key / "target").mkdir(parents=True)
        (base / key / "used").write_text("x")
        os.utime(base / key / "used", (1000 + index, 1000 + index))
    _never_fits(tmp_path)
    holding, evicting = session(tmp_path, "holding"), session(tmp_path, "evicting")
    hold(holding, "checkout:held")
    assert claim(evicting, "source") == "new"  # the most recently used free one
    evicted = evicting.evict("new")
    assert evicted["removed"] == ["checkouts/old"]
    assert evicted["free_bytes"] < evicted["needed_bytes"]
    assert sorted(p.name for p in base.iterdir()) == ["held", "new"]
    end(holding)
    end(evicting)


def test_a_build_records_the_checkout_its_size_and_the_most_a_rebuild_has_added(tmp_path):
    """A build into a cold checkout measures its build's size, and every later build of it
    there the most one adds; the checkout's next build of it needs that, another one's its
    size, and a host that measured neither the claim's fallback."""
    checkout = tmp_path / "checkouts" / "slot-0"
    current = session(tmp_path, "run")
    assert claim(current, "source", fallback=777) == "slot-0"
    assert status(tmp_path)["room"]["owed"] == 777  # nothing measured here yet: the fallback
    (checkout / "target").mkdir()
    added = []
    for size in (5000, 5400, 5300):  # from cold, then two warm builds adding less
        (checkout / "target" / "artifact").write_bytes(b"\0" * size)
        recorded = int((checkout / "size").read_text()) if added else 0
        built = current.built("slot-0", 1.0, set())
        current.builds["checkout:slot-0"] = agent.Owed(agent.SAME, BUILD, 0, 0, "c0", 60)  # built again
        assert built["added_bytes"] == built["bytes"] - recorded and built["trimmed"]["bytes"] == 0
        assert int((checkout / "size").read_text()) == built["bytes"]
        added.append(built["added_bytes"])
    assert added[0] > added[1] > added[2]
    recorded = json.loads((tmp_path / "rig" / agent.MEASURED / f"{BUILD}.json").read_text())
    assert {name: recorded[name] for name in ("size", "growth")} == {"size": added[0], "growth": added[1]}  # its rebuilds' most, not the latest
    assert BUILD in json.loads((checkout / agent.REFERENCES).read_text())
    end(current)
    report = status(tmp_path, "source")
    assert report["needs"] == [added[1]] and report["measured"][BUILD]["size"] == added[0]
    assert report["largest"] == added[0]
    (checkout / agent.REFERENCES).unlink()  # a checkout that does not keep it needs its whole size
    assert status(tmp_path, "source")["needs"] == [added[0]]


def test_eviction_counts_what_every_building_checkout_may_still_add(tmp_path):
    """Two runs build on the host while a third is about to: each held checkout and the one
    about to build may add its build's need (its size, or, in a checkout that built it before,
    the most a rebuild has added), less what it has added since its last recorded build; a warm
    checkout far past its first build counts in full."""
    base = tmp_path / "checkouts"
    holds = {"partial": 400, "fresh": 0, "warm": 23_000, "idle": 0}
    for key, size in holds.items():
        (base / key / "target").mkdir(parents=True)
        (base / key / "target" / "artifact").write_bytes(b"\0" * size)
        (base / key / agent.SOURCE).write_text(key)  # each run claims its own
    (base / "partial" / "size").write_text("100")
    (base / "idle" / "size").write_text("5000")
    keeps(base / "warm", BUILD)
    measured(tmp_path, size=1000, growth=700)
    first, second, building = (session(tmp_path, run) for run in ("first", "second", "building"))
    for current, key in ((building, "warm"), (first, "partial"), (second, "fresh")):  # warm keeps the build
        assert claim(current, key) == key
    (base / "warm" / "size").write_text(str(10**6))  # shrunk since (a mirror pruned its source)
    now = {key: checkout_bytes(str(base / key)) for key in holds}
    evicted = building.evict("warm")
    # idle is unheld, so it is not building; partial has added what it holds past its 100 bytes.
    assert evicted["needed_bytes"] == (1000 - (now["partial"] - 100)) + (1000 - now["fresh"]) + 700
    assert evicted["removed"] == []  # the volume holds that much already
    for current in (first, second, building):
        end(current)


def _volume(monkeypatch, root: Path, others: int) -> None:
    """The rig root's volume as a test sets it: `others` bytes free beside the rig, less what the
    root's checkouts hold by their recorded sizes, so an eviction frees its checkout's size."""
    def usage(path):
        held = sum(int((checkout / "size").read_text()) for checkout in (root / "checkouts").iterdir()
                   if (checkout / "size").exists())
        return shutil._ntuple_diskusage(10**13, 10**13 - others + held, others - held)

    monkeypatch.setattr(agent.shutil, "disk_usage", usage)


def _stale(root: Path, name: str, size: int, used: float) -> None:
    (root / "checkouts" / name).mkdir(parents=True)
    (root / "checkouts" / name / "size").write_text(str(size))
    (root / "checkouts" / name / "used").write_text("x")
    os.utime(root / "checkouts" / name / "used", (used, used))


def test_after_a_build_the_rig_keeps_its_checkouts_within_its_share_of_the_volume(tmp_path, monkeypatch):
    """Once a build ends, the rig removes its own checkouts no run holds, least recently used
    first, until they hold no more than its SHARE of the volume, and no further, however little
    of the volume is free; it never waits, and never takes a held checkout or a run's directory."""
    gb = 10**9
    _stale(tmp_path, "held", 1 * gb, 999)
    _stale(tmp_path, "oldest", 6 * gb, 1000)
    _stale(tmp_path, "older", 5 * gb, 1001)
    _stale(tmp_path, "recent", 4 * gb, 1002)
    (tmp_path / "runs" / "gone").mkdir(parents=True)
    total = int(11 * gb / agent.SHARE)  # a share of 11 GB; the other users leave nothing free
    monkeypatch.setattr(agent.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(total, total, 0))
    holding, building = session(tmp_path, "holding"), session(tmp_path, "building")
    hold(holding, "checkout:held")
    assert building.reclaim() == {"removed": ["checkouts/oldest"], "freed_bytes": 6 * gb}  # 10 GB left
    assert building.reclaim() == {"removed": [], "freed_bytes": 0}
    assert sorted(p.name for p in (tmp_path / "checkouts").iterdir()) == ["held", "older", "recent"]
    assert (tmp_path / "runs" / "gone").is_dir()
    end(holding)
    end(building)


def test_room_is_made_from_the_rigs_own_stale_checkouts_least_recently_used_first(tmp_path, monkeypatch):
    """A build that does not fit beside what is free takes it from the rig's own checkouts no run
    holds, least recently used first, until it fits, and no further; a held checkout is never
    taken, and the evictions are announced with what they freed."""
    gb = 10**9
    _stale(tmp_path, "oldest", 6 * gb, 1000)
    _stale(tmp_path, "older", 5 * gb, 1001)
    _stale(tmp_path, "recent", 4 * gb, 1002)
    _stale(tmp_path, "held", 7 * gb, 900)  # least recently used of all, but a live run holds it
    (tmp_path / "checkouts" / "mine").mkdir()
    (tmp_path / "checkouts" / "mine" / agent.SOURCE).write_text("source")  # the run's own, claimed
    _volume(monkeypatch, tmp_path, others=25 * gb)  # 3 GB free beside the rig's 22 GB
    measured(tmp_path, size=10 * gb)
    holding, building = session(tmp_path, "holding"), session(tmp_path, "building")
    hold(holding, "checkout:held")
    assert claim(building, "source") == "mine"
    evicted = building.evict("mine")
    assert evicted["removed"] == ["checkouts/oldest", "checkouts/older"]  # 3 + 6 + 5 GB ≥ 10 GB
    assert evicted["freed_bytes"] == 11 * gb and evicted["free_bytes"] >= evicted["needed_bytes"]
    assert sorted(path.name for path in (tmp_path / "checkouts").iterdir()) == ["held", "mine", "recent"]
    end(holding)
    end(building)


def test_a_build_is_refused_only_when_what_the_rig_does_not_own_leaves_no_room(tmp_path, monkeypatch):
    """Every stale checkout taken, a live run's held one kept: the build is refused when that
    room, and the held checkout once its run ends, still fall short of its need."""
    gb = 10**9
    _stale(tmp_path, "stale", 6 * gb, 1000)
    _stale(tmp_path, "held", 7 * gb, 900)
    (tmp_path / "checkouts" / "mine").mkdir()
    (tmp_path / "checkouts" / "mine" / agent.SOURCE).write_text("source")
    _volume(monkeypatch, tmp_path, others=16 * gb)  # 3 GB free beside the rig's 13 GB
    measured(tmp_path, size=17 * gb)  # more than all 16 GB the rig could ever free
    holding, building = session(tmp_path, "holding"), session(tmp_path, "building")
    hold(holding, "checkout:held")
    assert claim(building, "source") == "mine"
    evicted = building.evict("mine")
    assert evicted["removed"] == ["checkouts/stale"] and (tmp_path / "checkouts" / "held").is_dir()
    # The runner's refusal: free, and the live runs' checkouts as they end, short of its own need.
    assert evicted["free_bytes"] + evicted["releasable_bytes"] < evicted["own_bytes"]
    end(holding)
    end(building)


def test_eviction_keeps_live_and_unfetched_runs_and_removes_gone_ones(tmp_path):
    """A gone run's directory goes in last-use order with unheld checkouts; a live run's never
    does, nor (for an agent of another version, which locks none) one a live entry names, nor
    an ended one its caller has not fetched."""
    (tmp_path / "checkouts" / "new").mkdir(parents=True)
    _never_fits(tmp_path)
    live = session(tmp_path, "live")
    versioned = session(tmp_path, "versioned")
    versioned.run_lock.close()  # as an agent that locks no run directory
    hold(versioned, "checkout:other")
    building = session(tmp_path, "building")
    assert claim(building, "source") == "new"
    for index, path in enumerate(("runs/gone", "checkouts/idle", "runs/later", "runs/unfetched")):
        (tmp_path / path).mkdir(parents=True)
        used = tmp_path / path / "used" if path.startswith("checkouts") else tmp_path / path
        used.touch()
        os.utime(used, (1000 + index, 1000 + index))
    (tmp_path / "runs" / "unfetched" / agent.ENDED).write_text("{}")
    for run in ("live", "versioned", "unfetched"):
        os.utime(tmp_path / "runs" / run, (900, 900))  # least recently used of all
    evicted = building.evict("new")
    assert evicted["removed"] == ["runs/gone", "checkouts/idle", "runs/later"]
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["building", "live", "unfetched", "versioned"]
    for current in (live, versioned, building):
        end(current)


def test_a_stage_is_kept_from_eviction_until_its_run_starts_or_its_caller_withdraws(tmp_path):
    """`start` holds the run's directory while its caller stages it: eviction keeps it, and
    the runner owns it before the hold ends. A caller that closes without a plan withdraws:
    nothing starts, and the directory is a gone run's."""
    (tmp_path / "checkouts" / "new").mkdir(parents=True)
    _never_fits(tmp_path)

    def staging(run: str) -> subprocess.Popen:
        process = subprocess.Popen([sys.executable, str(AGENT), "start", "--root", str(tmp_path), "--run", run],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert json.loads(process.stdout.readline()) == {"staging": run}
        (tmp_path / "runs" / run / "stage-a").mkdir()
        os.utime(tmp_path / "runs" / run, (900, 900))  # least recently used of all
        return process

    started, withdrawn = staging("started"), staging("withdrawn")
    building = session(tmp_path, "building")
    assert claim(building, "source") == "new"
    assert building.evict("new")["removed"] == []
    assert (tmp_path / "runs" / "started" / "stage-a").is_dir()
    plan = {"info": {"who": "tester", "purpose": "p", "estimate": 60},
            "ops": [run_op([sys.executable, "-c", "print('ran')"], tmp_path, job="j")]}
    output, _ = started.communicate(json.dumps(plan))
    assert json.loads(output)["pid"] > 0 and ended(tmp_path, "started")["codes"] == {"j": 0}
    withdrawn.stdin.close()
    assert withdrawn.wait(10) == 0 and withdrawn.stdout.read() == ""
    assert not (tmp_path / "runs" / "withdrawn" / agent.PLAN).exists()
    assert building.evict("new")["removed"] == ["runs/withdrawn"]  # gone; the ended run is kept
    end(building)


def install(stage: Path, checkout: Path) -> dict:
    result = subprocess.run([sys.executable, str(AGENT), "install", "--from", str(stage), "--to", str(checkout)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def tree(directory: Path, files: dict[str, str]) -> Path:
    for path, text in files.items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text(text)
    return directory


def test_install_makes_the_checkout_equal_to_the_stage_rewriting_only_what_differs(tmp_path):
    checkout = tree(tmp_path / "checkout", {"same.rs": "same", "changed.rs": "old", "drop/c.rs": "c",
                                            "keep/extra.rs": "x", "was-file": "f", "was-dir/inner": "i"})
    (tmp_path / "checkout" / "empty").mkdir()
    old = 1_600_000_000
    for path in checkout.rglob("*"):
        os.utime(path, (old, old), follow_symlinks=False)
    stage = tree(tmp_path / "stage", {"same.rs": "same", "changed.rs": "new", "keep/kept.rs": "k",
                                      "was-file/inner": "i", "was-dir": "f"})
    (stage / "tool.sh").write_text("#!/bin/sh\n")
    (stage / "tool.sh").chmod(0o755)
    (stage / "link").symlink_to("same.rs")
    (stage / "dirlink").symlink_to("keep")
    assert install(stage, checkout) == {"written": 7, "removed": 4}
    files = sorted(p.relative_to(checkout).as_posix() for p in checkout.rglob("*"))
    assert files == ["changed.rs", "dirlink", "keep", "keep/kept.rs", "link", "same.rs", "tool.sh",
                     "was-dir", "was-file", "was-file/inner"]
    assert (checkout / "changed.rs").read_text() == "new" and (checkout / "was-dir").read_text() == "f"
    assert (checkout / "same.rs").stat().st_mtime == old  # same bytes: left as it was
    assert (checkout / "changed.rs").stat().st_mtime > old
    assert os.access(checkout / "tool.sh", os.X_OK)
    assert os.readlink(checkout / "link") == "same.rs" and os.readlink(checkout / "dirlink") == "keep"
    assert install(stage, checkout) == {"written": 0, "removed": 0}  # equal: nothing to do
    (stage / "same.rs").chmod(0o700)
    (stage / "link").unlink()
    (stage / "link").symlink_to("tool.sh")
    assert install(stage, checkout) == {"written": 1, "removed": 0}
    assert (checkout / "same.rs").stat().st_mode & 0o777 == 0o700  # the mode, in place
    assert (checkout / "same.rs").stat().st_mtime == old and os.readlink(checkout / "link") == "tool.sh"


def test_install_from_a_missing_stage_fails_and_keeps_the_checkout(tmp_path):
    checkout = tree(tmp_path / "checkout", {"a.rs": "a"})
    result = subprocess.run([sys.executable, str(AGENT), "install", "--from", str(tmp_path / "gone"),
                             "--to", str(checkout)], capture_output=True, text=True)
    assert result.returncode != 0 and (checkout / "a.rs").read_text() == "a"


# Runners: detached processes carrying out plans --------------------------------------------


def command(root: Path, verb: str, run: str, *extra: str, agent_path: Path = AGENT, **options):
    return subprocess.run([sys.executable, str(agent_path), verb, "--root", str(root), "--run", run, *extra],
                          capture_output=True, text=True, **options)


def start(root: Path, run: str, ops: list, *, agent_path: Path = AGENT, estimate: float = 60) -> dict:
    plan = {"info": {"who": "tester", "purpose": f"purpose of {run}", "estimate": estimate}, "ops": ops}
    started = command(root, "start", run, agent_path=agent_path, input=json.dumps(plan), check=True)
    staging, runner = started.stdout.splitlines()
    assert json.loads(staging) == {"staging": run}
    return json.loads(runner)


def events(root: Path, run: str) -> list[dict]:
    path = root / "runs" / run / agent.EVENTS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def until(root: Path, run: str, event: str, timeout=15.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        found = [e for e in events(root, run) if e["event"] == event]
        if found:
            return found[-1]
        assert time.monotonic() < deadline, f"{run} never had {event}: {events(root, run)}"
        time.sleep(0.05)


def ended(root: Path, run: str) -> dict:
    until(root, run, "ended")
    return json.loads((root / "runs" / run / agent.ENDED).read_text())


def run_op(argv: list, cwd: Path, **fields) -> dict:
    return {"op": "run", "argv": argv, "cwd": str(cwd), "env": {}, **fields}


def until_exists(path: Path) -> dict:
    """A step that runs until the test creates `path`: the run blocks there, holding what it holds."""
    return run_op(waiting.until_exists(path), path.parent)


def _orphaned(argv: list, ready: Path) -> int:
    """Start `argv` under a parent and, once `argv` has made `ready`, kill the parent outright:
    the orphan's pid."""
    spawn = "import subprocess, sys, time\nprint(subprocess.Popen(sys.argv[1:]).pid, flush=True)\ntime.sleep(600)"
    parent = subprocess.Popen([sys.executable, "-c", spawn, *argv], stdout=subprocess.PIPE, text=True)
    pid = int(parent.stdout.readline())
    deadline = time.monotonic() + 30
    while not ready.exists():
        assert time.monotonic() < deadline, f"{argv} never made {ready}"
        time.sleep(0.05)
    parent.kill()
    parent.wait()
    return pid


def _ends(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_waiting_step_ends_with_its_owner_whenever_it_starts_and_by_its_deadline(tmp_path):
    """The tests' wait for a path, the path never made, never outlives the process it names:
    started after that process is gone, it exits at once; started before, once it goes; and
    whoever lives, at its deadline."""
    bound = 5 * waiting.POLL_SECONDS + 2
    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()  # gone and reaped before the wait starts
    started = subprocess.Popen(waiting.until_exists(tmp_path / "never", owner=dead.pid))
    assert started.wait(timeout=bound) == 1
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    wait = subprocess.Popen(waiting.until_exists(tmp_path / "never", owner=owner.pid))
    time.sleep(2 * waiting.POLL_SECONDS)
    assert wait.poll() is None  # its owner lives
    owner.kill()
    owner.wait()
    assert wait.wait(timeout=bound) == 1
    bounded = subprocess.Popen(waiting.until_exists(tmp_path / "never", seconds=0.5))  # this test lives
    assert bounded.wait(timeout=bound) == 1


def test_a_recorded_job_ends_with_its_runner(tmp_path):
    """A runner killed outright ends none of its steps: the host's `record` ends its job and
    itself once its runner is gone."""
    job = [sys.executable, "-c", "import os, sys, time\nopen(sys.argv[1] + '.tmp', 'w').write(str(os.getpid()))\n"
           "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\ntime.sleep(600)", str(tmp_path / "job.pid")]
    record = [sys.executable, "-m", "magnitude_rig.host", "record", "--run-dir", str(tmp_path / "run"),
              "--alias", "here", "--", *job]
    pid = _orphaned(["env", f"PYTHONPATH={AGENT.parents[2]}", *record], tmp_path / "job.pid")
    job_pid = int((tmp_path / "job.pid").read_text())
    from magnitude_rig.host.__main__ import PARENT_POLL_SECONDS
    assert _ends(pid, PARENT_POLL_SECONDS + 10) and _ends(job_pid, 5)


def _room_plan(root: Path) -> list:
    """Claim a checkout, make room for its build under the install hold, then build."""
    return [{"op": "checkout", "side": "s", "source": "src", "tree": "t", "build": BUILD, "output": "o", "fallback": 0, "commit": "c0", "seconds": 60}, {"op": "hold", "kind": "install"},
            {"op": "evict", "side": "s", "host": "h"}, {"op": "release", "kind": "install"},
            run_op([sys.executable, "-c", "print('built')"], root, job="j")]


def test_a_run_short_of_room_that_live_runs_will_free_waits_for_them(tmp_path):
    """The volume holds one build's growth but not two: a run claiming beside a live run's
    building checkout waits for room, without its install hold and saying behind whom, and
    builds once that run ends; status puts a new run's wait at that run's remaining estimate."""
    size = int(0.6 * shutil.disk_usage(tmp_path).free)
    measured(tmp_path, size=size)
    other = session(tmp_path, "other", estimate=600)
    assert claim(other, "x") == "slot-0"
    assert 590 < engine.room_wait(status(tmp_path)["room"], size) <= 600
    start(tmp_path, "waiting", _room_plan(tmp_path))
    waiting = until(tmp_path, "waiting", "waiting")
    assert waiting["kind"] == "room" and [b["run"] for b in waiting["blocking"]] == ["other"]
    entries = {(e["run"], e["kind"]): e["state"] for e in status(tmp_path)["entries"]}
    assert entries[("waiting", "room")] == "waiting" and ("waiting", "install") not in entries
    assert not any(e["event"] == "ended" for e in events(tmp_path, "waiting"))
    end(other)
    result = ended(tmp_path, "waiting")
    assert result["codes"] == {"j": 0} and result["failed"] is None


def test_a_run_whose_growth_even_live_runs_ending_could_not_cover_fails_before_it_builds(tmp_path):
    size = int(1.5 * shutil.disk_usage(tmp_path).free)
    measured(tmp_path, size=size)
    other = session(tmp_path, "other")
    claim(other, "x")
    assert engine.room_wait(status(tmp_path)["room"], size) is None
    start(tmp_path, "refused", _room_plan(tmp_path))
    failed = ended(tmp_path, "refused")
    assert failed["codes"] == {} and "would not cover its own" in failed["failed"]["detail"]
    end(other)


def test_a_runner_carries_out_its_plan_and_records_its_end(tmp_path):
    ops = [
        {"op": "phase", "phase": "build"},
        {"op": "hold", "kind": "build"},
        run_op([sys.executable, "-c", "print('/the/export')"], tmp_path, capture="export"),
        run_op([sys.executable, "-c", "import sys; print(sys.argv[1])", "{capture:export}"], tmp_path,
               stdout_to=str(tmp_path / "out.txt")),
        {"op": "release", "kind": "build"},
        {"op": "phase", "phase": "job-j"},
        {"op": "hold", "kind": "device"},
        run_op([sys.executable, "-c", "import sys; print(sys.stdin.read().upper()); sys.exit(5)"], tmp_path,
               input="hi", job="j"),
    ]
    started = start(tmp_path, "run", ops)
    assert started["agent"] == [str(AGENT.resolve())] and started["pid"] > 0
    record = ended(tmp_path, "run")
    assert record["failed"] is None and record["codes"] == {"j": 5}
    assert [p["phase"] for p in record["phases"]] == ["build", "job-j"]
    assert (tmp_path / "out.txt").read_text() == "/the/export\n"
    assert "HI" in (tmp_path / "runs" / "run" / "logs" / "job-j.log").read_text()
    followed = command(tmp_path, "follow", "run", "--after", "0", check=True, timeout=30).stdout
    assert [json.loads(line)["event"] for line in followed.splitlines()][-1] == "ended"
    assert status(tmp_path)["entries"] == []  # its holds went with it


# A test binary that reports on SIGUSR1 as the engine's diagnostics do: its handler in place, it
# creates `stacks-<pid>.txt` in the rig's directory and appends every thread's stack there when
# asked, without stopping; it also names a foreign process's file there (argv[1], a process
# outside the job), which must never be signalled. Then it blocks, for as long as the test.
HUNG_TEST = """
import faulthandler, os, signal, sys, time
directory = os.environ["MAGNITUDE_STACKS"]
report = open(os.path.join(directory, "stacks-%d.txt" % os.getpid()), "a")
signal.signal(signal.SIGUSR1, lambda signum, frame: (faulthandler.dump_traceback(report, all_threads=True), report.flush()))
open(os.path.join(directory, "stacks-%s.txt" % sys.argv[1]), "a").close()
print("test hangs::blocks_forever ... ", flush=True)
def block_forever():
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(0.1)
block_forever()
"""


def test_a_job_past_its_estimate_reports_its_stacks_and_runs_on(tmp_path):
    """A job whose held time passes its estimate has each of its processes that reports on
    SIGUSR1 asked once for its stacks: they land in the run's logs, naming the frame it blocks
    in, the process is not stopped, and a process outside the job is never signalled."""
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        start(tmp_path, "hung", [run_op([sys.executable, "-c", HUNG_TEST, str(foreign.pid)], tmp_path, job="hang")],
              estimate=1)
        asked = until(tmp_path, "hung", "stacks", timeout=30)
        report = tmp_path / "runs" / "hung" / asked["path"]
        assert asked["path"] == f"logs/stacks/stacks-{asked['pid']}.txt" and "estimate" in asked["reason"]
        deadline = time.monotonic() + 15
        while "block_forever" not in (report.read_text() if report.exists() else ""):
            assert time.monotonic() < deadline, f"no stacks in {report}"
            time.sleep(0.1)
        os.kill(asked["pid"], 0)  # still running: asked, not stopped
        assert foreign.poll() is None  # its file named it, but it is not the job's
        assert [e["pid"] for e in events(tmp_path, "hung") if e["event"] == "stacks"] == [asked["pid"]]
    finally:
        command(tmp_path, "stop", "hung")
        foreign.kill()
        foreign.wait()
    ended(tmp_path, "hung")


def test_a_test_binary_whose_results_stop_is_a_hang_past_its_longest_recent_gap(tmp_path, monkeypatch):
    """Results that stop for `STALL_FACTOR` times the longest gap between recent ones, and at
    least `STALL_FLOOR`, are a hang to report; output that is not a test result moves nothing."""
    now = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: now[0])
    watch = agent.Watch(str(tmp_path), 0, lambda: 0.0, 3600.0, None)
    assert watch.reason() is None  # no result yet: only the estimate can say
    for at, line in ((1.0, "test a ... ok"), (3.0, "test b ... ok"), (11.0, "test c ... FAILED"),
                     (12.0, "running 3 tests")):
        now[0] = at
        watch.saw(line)
    now[0] = 11.0 + 79  # 8 s the longest gap: a hang at 80 s
    assert watch.reason() is None
    now[0] = 11.0 + 80
    assert watch.reason().startswith("no test result for 80s")
    quick = agent.Watch(str(tmp_path), 0, lambda: 0.0, 3600.0, None)
    for at in (100.0, 100.5):
        now[0] = at
        quick.saw("test q ... ok")
    now[0] = 100.5 + agent.STALL_FLOOR - 1  # 5 s by its gaps, but never under the floor
    assert quick.reason() is None
    watch.held = lambda: 3600.0
    assert "estimate" in watch.reason()


def test_a_jobs_long_line_is_kept_head_and_tail_and_its_end_is_its_commands_exit(tmp_path):
    """A test printed a 1.4 GB assertion on one line, and its
    caller, streaming it, showed the job running 11 minutes after it ended. A logged line keeps
    its head and its tail, what lies between counted; a step ends with its command's exit, however
    long a process that left its group holds its output open, and says so."""
    long_line = ("import sys\nsys.stdout.write('head')\nfor _ in range(500):\n    sys.stdout.write('x' * (1 << 20))\n"
                 "sys.stdout.write('tail\\n')\nsys.stdout.flush()\nprint('after', flush=True)")  # a 500 MiB line
    escaped = ("import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
               "start_new_session=True)\nprint('left a holder', flush=True)\nsys.exit(3)")
    start(tmp_path, "loud", [run_op([sys.executable, "-c", long_line], tmp_path, job="loud"),
                             run_op([sys.executable, "-c", escaped], tmp_path, job="escaped")])
    result = ended(tmp_path, "loud")
    assert result["codes"] == {"loud": 0, "escaped": 3}
    texts = [e["text"] for e in events(tmp_path, "loud") if e["event"] == "output"]
    (kept,) = [text for text in texts if text.startswith("head")]
    assert len(kept) < agent.LINE_HEAD + agent.LINE_TAIL + 100 and kept.endswith("xtail\n")
    elided = (500 << 20) + 9 - agent.LINE_HEAD - agent.LINE_TAIL  # head + 500 MiB + tail + newline, less those kept
    assert f"[... {elided} bytes of this line not kept by the rig ...]" in kept and "after\n" in texts
    # It costs neither the log nor the runner: the line is never held whole.
    assert (tmp_path / "runs" / "loud" / "logs" / "start.log").stat().st_size < 4 * agent.LINE_HEAD
    assert any("a process that left its group still holds its output open" in text for text in texts)
    assert sum(phase["seconds"] for phase in result["phases"]) < agent.OUTPUT_GRACE_SECONDS + 10  # not the holder's 60 s


def test_a_follow_reads_through_an_earlier_runners_giant_line_without_holding_it(tmp_path):
    """An earlier runner wrote a 1.4 GB output line as one event. A follow serving it reads such
    a line through in pieces, never holding it whole, and forwards that it was not forwarded."""
    import resource

    run = tmp_path / "runs" / "old"
    run.mkdir(parents=True)
    with open(run / agent.EVENTS, "w") as stream:
        stream.write(json.dumps({"seq": 1, "event": "phase", "at": 0.0, "phase": "job-test"}) + "\n")
        stream.write('{"seq": 2, "event": "output", "at": 0.0, "phase": "job-test", "stream": "stdout", "text": "')
        for _ in range(200):
            stream.write("x" * (1 << 20))
        stream.write('\\n"}\n')
        stream.write(json.dumps({"seq": 3, "event": "ended", "at": 0.0, "codes": {}, "phases": [], "failed": None}) + "\n")
    followed = command(tmp_path, "follow", "old", "--after", "0", check=True)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # bytes on macOS, KiB on Linux
    lines = [json.loads(line) for line in followed.stdout.splitlines()]
    assert [line["event"] for line in lines] == ["phase", "ended"]  # output before a follow began is in the log
    assert peak < (100 << 20 if sys.platform == "darwin" else 100 << 10)  # never the line's 200 MB
    events = list(map(agent._event_line, [open(run / agent.EVENTS, "rb")] * 3))
    assert json.loads(events[1])["text"] == f"agent: an output line of {len(open(run / agent.EVENTS, 'rb').readlines()[1])} bytes not forwarded\n"


def test_a_hang_with_no_process_to_ask_says_so_once(tmp_path):
    asked = []
    watch = agent.Watch(str(tmp_path), os.getpgid(0), lambda: 10.0, 1.0, lambda *ask: asked.append(ask))
    watch.check()
    watch.check()
    assert asked == [(None, None, "its held time passed its 1s estimate")]


def test_a_failing_step_ends_the_run_in_its_phase(tmp_path):
    marker = tmp_path / "never"
    ops = [{"op": "phase", "phase": "build-source"},
           run_op([sys.executable, "-c", "import sys; print('error: no'); sys.exit(3)"], tmp_path),
           run_op([sys.executable, "-c", "open(__import__('sys').argv[1], 'w')", str(marker)], tmp_path)]
    start(tmp_path, "run", ops)
    record = ended(tmp_path, "run")
    assert record["failed"]["phase"] == "build-source" and "exited 3" in record["failed"]["detail"]
    assert not marker.exists()


def test_a_run_outlives_the_connection_following_it(tmp_path):
    """The follower dies mid-run (a dropped link); the runner goes on, and a new follower
    takes the events up after the last one the old one saw: every one but the output written
    before it began, which the phase's log holds. (So the old follower, too, sees no output
    written before it began: the step's command line may be, so it reads the phase alone.)"""
    ops = [{"op": "phase", "phase": "job-j"}, until_exists(tmp_path / "go"),
           run_op([sys.executable, "-c", "print('done')"], tmp_path, job="j")]
    start(tmp_path, "run", ops)
    follower = subprocess.Popen([sys.executable, str(AGENT), "follow", "--root", str(tmp_path), "--run", "run"],
                                stdout=subprocess.PIPE, text=True)
    seen = [json.loads(follower.stdout.readline())]
    assert [e["event"] for e in seen] == ["phase"]
    follower.kill()
    follower.wait()
    (tmp_path / "go").touch()
    assert ended(tmp_path, "run")["codes"] == {"j": 0}
    rest = command(tmp_path, "follow", "run", "--after", str(seen[-1]["seq"]), check=True, timeout=30).stdout
    events_after = [json.loads(line) for line in rest.splitlines()]
    assert events_after[0]["seq"] > seen[-1]["seq"] and events_after[-1]["event"] == "ended"
    assert [e["event"] for e in events_after if e["event"] in ("exited", "ended")] == ["exited", "exited", "ended"]
    assert not any(e["event"] == "output" for e in events_after)
    assert "done\n" in (tmp_path / "runs" / "run" / "logs" / "job-j.log").read_text()


def test_a_retried_start_supersedes_one_still_waiting_for_its_plan(tmp_path):
    """The first attempt's connection went silent while it waited for the plan: the retry ends it
    and starts the run, rather than waiting for the dead connection to be noticed."""
    earlier = subprocess.Popen([sys.executable, str(AGENT), "start", "--root", str(tmp_path), "--run", "run"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert json.loads(earlier.stdout.readline()) == {"staging": "run"}
    started = start(tmp_path, "run", [run_op([sys.executable, "-c", "print('ran')"], tmp_path, job="j")])
    assert earlier.wait(10) != 0  # superseded
    assert ended(tmp_path, "run")["codes"] == {"j": 0} and started["pid"] > 0


def test_a_run_that_is_not_running_is_never_signalled(tmp_path):
    """Its runner has recorded its end: its pid may since be another process's, so no stop is
    sent, and the answer says why."""
    start(tmp_path, "run", [run_op([sys.executable, "-c", "print('done')"], tmp_path, job="j")])
    ended(tmp_path, "run")
    stopped = command(tmp_path, "stop", "run", check=True)
    assert json.loads(stopped.stdout) == {"stopped": False, "reason": "run run is not running"}


def test_a_stopped_run_kills_its_work_releases_its_holds_and_records_it(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    start(tmp_path, "run", [{"op": "phase", "phase": "job-j"}, {"op": "hold", "kind": "device"},
                            run_op([sys.executable, "-c", script], tmp_path, job="j")])
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert json.loads(command(tmp_path, "stop", "run", check=True).stdout) == {"stopped": True}
    record = ended(tmp_path, "run")
    assert record["failed"] == {"phase": "job-j", "detail": "stopped by mrig stop"}
    assert json.loads(command(tmp_path, "state", "run", check=True).stdout) == {"state": "ended"}
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 15
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "work outlived its stopped run"
        time.sleep(0.05)
    assert status(tmp_path)["entries"] == []


def test_a_runner_that_dies_holds_nothing_and_its_followers_learn_it(tmp_path):
    standing = lambda run: json.loads(command(tmp_path, "state", run, check=True).stdout)["state"]
    started = start(tmp_path, "dead", [{"op": "hold", "kind": "device"}, until_exists(tmp_path / "never")])
    until(tmp_path, "dead", "output")
    waiting = session(tmp_path, "waiting")
    assert blocked(waiting, "device") == ["dead"]
    assert standing("dead") == "running" and standing("never-started") == "absent"
    os.kill(started["pid"], signal.SIGKILL)
    deadline = time.monotonic() + 10
    while waiting.blocking("device"):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    followed = command(tmp_path, "follow", "dead", check=False, timeout=30).stdout.splitlines()
    assert json.loads(followed[-1])["event"] == "vanished"
    assert standing("dead") == "vanished"
    end(waiting)


def test_a_started_run_is_started_once(tmp_path):
    """Every start after the first, running or ended, names the runner that owns the run and
    stages nothing: no plan is read, no second runner starts."""
    first = start(tmp_path, "run", [until_exists(tmp_path / "go")])

    def again() -> dict:
        repeated = command(tmp_path, "start", "run", input=json.dumps({"ops": []}), check=True)
        (line,) = repeated.stdout.splitlines()
        return json.loads(line)

    until(tmp_path, "run", "output")
    assert again() == {"started": first}
    (tmp_path / "go").touch()
    ended(tmp_path, "run")
    assert again() == {"started": first}
    assert json.loads((tmp_path / "runs" / "run" / agent.PLAN).read_text())["ops"][0]["op"] == "run"


def test_a_run_directory_is_removed_only_once_ended_and_fetched(tmp_path):
    start(tmp_path, "run", [until_exists(tmp_path / "go")])
    until(tmp_path, "run", "output")
    assert command(tmp_path, "fetched", "run").returncode != 0  # still running
    assert (tmp_path / "runs" / "run").exists()
    (tmp_path / "go").touch()
    ended(tmp_path, "run")
    deadline = time.monotonic() + 10
    while command(tmp_path, "fetched", "run").returncode != 0:  # until the runner has exited
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert not (tmp_path / "runs" / "run").exists()


# A run of protocol 2, judging as its agent does: an entry of another protocol conflicts with
# every request, and a waiting entry ahead of a request blocks it. Driven by lines on stdin.
OLD_RUN = """
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import agent
rig, run = agent.Rig(sys.argv[2]), sys.argv[3]
entries = {}

def enter(kind):
    def write():
        live = rig.live()
        sequence = 1 + max((int(e["entry"].split("-")[0]) for e in live), default=0)
        path = os.path.join(rig.entries, "%012d-%d-%d.json" % (sequence, os.getpid(), len(entries)))
        lock = agent.FileLock(path, "w+b", agent.ENTRY_LOCK_OFFSET)
        assert lock.try_acquire()
        entries[kind] = (path, lock)
        state(kind, "waiting")
    rig.guarded(write)

def state(kind, value):
    lock = entries[kind][1]
    lock.file.seek(0); lock.file.truncate()
    lock.file.write(json.dumps({"run": run, "kind": kind, "state": value, "protocol": 2, "who": "old",
                                "purpose": "p", "estimate": 60, "started": time.time()}).encode())
    lock.file.flush()

def held(kind):
    def judge():
        mine, ahead = os.path.basename(entries[kind][0]), True
        for entry in rig.live():
            if entry["entry"] == mine:
                ahead = False
            elif entry["run"] != run and (entry.get("protocol") != 2 or agent.conflicts(entry["kind"], kind)) \\
                    and (entry["state"] == "held" or ahead):
                return False
        state(kind, "held")
        return True
    return rig.guarded(judge)

for line in sys.stdin:
    kind = line.strip()
    if kind == "end":
        break
    enter(kind)
    while not held(kind):
        time.sleep(0.05)
    print("held " + kind, flush=True)
"""


def test_runs_of_two_protocols_never_wait_on_each_other_in_a_cycle(tmp_path):
    """A protocol 2 run holding its checkout requests the device after a run of this protocol
    requested its checkout. Its agent counts every entry of another protocol as conflicting and
    a waiting one ahead as blocking, so had this run entered its request, each would wait on the
    other. This run enters nothing while the
    other is live, and a protocol 2 run that comes later waits on it whole."""
    old = subprocess.Popen([sys.executable, "-c", OLD_RUN, str(AGENT.parent), str(tmp_path), "old"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def old_holds(kind):
        old.stdin.write(kind + "\n")
        old.stdin.flush()
        assert old.stdout.readline().strip() == "held " + kind

    old_holds("checkout:G")
    start(tmp_path, "ours", [{"op": "checkout", "side": "s", "source": "U", "tree": "t", "build": BUILD, "output": "o", "fallback": 0, "commit": "c0", "seconds": 60}, {"op": "phase", "phase": "job-j"},
                             {"op": "hold", "kind": "device"},
                             run_op([sys.executable, "-c", "print('measured')"], tmp_path, job="j")])
    waiting = until(tmp_path, "ours", "waiting")
    assert [b["run"] for b in waiting["blocking"]] == ["old"]
    assert [e["run"] for e in status(tmp_path)["entries"]] == ["old"]  # ours entered nothing
    old_holds("device")  # nothing of ours is ahead of it
    old.stdin.write("end\n")
    old.stdin.flush()
    old.wait(10)
    assert ended(tmp_path, "ours")["codes"] == {"j": 0}

    # A protocol 2 run that comes while ours is live waits on it whole; ours never on it.
    start(tmp_path, "later-ours", [{"op": "checkout", "side": "s", "source": "U", "tree": "t", "build": BUILD, "output": "o", "fallback": 0, "commit": "c0", "seconds": 60}, until_exists(tmp_path / "go"),
                                   {"op": "hold", "kind": "device"}])
    until(tmp_path, "later-ours", "output")
    late = subprocess.Popen([sys.executable, "-c", OLD_RUN, str(AGENT.parent), str(tmp_path), "late"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    late.stdin.write("checkout:G\n")
    late.stdin.flush()
    (tmp_path / "go").touch()
    assert ended(tmp_path, "later-ours")["failed"] is None
    assert late.stdout.readline().strip() == "held checkout:G"  # once ours is gone
    late.stdin.write("end\n")
    late.stdin.flush()
    late.wait(10)


def test_a_run_shares_no_hold_with_a_run_of_another_protocol(tmp_path):
    other = tmp_path / "agent-next.py"
    current = int(re.search(r"\nPROTOCOL = (\d+)\n", AGENT.read_text())[1])
    other.write_text(AGENT.read_text().replace(f"\nPROTOCOL = {current}\n", f"\nPROTOCOL = {current + 1}\n"))
    root = tmp_path / "root"
    start(root, "newer", [{"op": "checkout", "side": "s", "source": "a", "tree": "t", "build": BUILD, "output": "o", "fallback": 0, "commit": "c0", "seconds": 60}, until_exists(root / "go")],
          agent_path=other)
    until(root, "newer", "output")
    ours = session(root, "ours")
    name, _, others = ours.claim("b", "t", BUILD, 0, "c0", 60)
    assert name is None and {e["run"] for e in others} == {"newer"}  # even a checkout waits
    assert not ours.entries
    (root / "go").touch()
    ended(root, "newer")
    deadline = time.monotonic() + 10
    while ours.claim("b", "t", BUILD, 0, "c0", 60)[0] is None:  # until the newer run has exited
        assert time.monotonic() < deadline
        time.sleep(0.05)
    end(ours)


needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="needs cargo")


def _crate(directory: Path, version: str) -> Path:
    """A one-crate workspace at `version` (a unit's hash covers it, so a new one is a new unit)."""
    (directory / "lib" / "src").mkdir(parents=True)
    (directory / "Cargo.toml").write_text('[workspace]\nmembers = ["lib"]\nresolver = "2"\n')
    (directory / "lib" / "Cargo.toml").write_text(
        f'[package]\nname = "my-lib"\nversion = "{version}"\nedition = "2021"\n')
    (directory / "lib" / "src" / "lib.rs").write_text("pub fn who() -> u32 { let unused = 1; 1 }\n")
    (directory / "lib" / "tests").mkdir()
    (directory / "lib" / "tests" / "here.rs").write_text(  # reads its package's directory as cargo gives it
        '#[test]\nfn here() { assert!(std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())'
        '.join("Cargo.toml").exists()); }\n')
    return directory


def _build_plan(stage: Path, tree: str, *after: dict) -> list:
    """Claim, install `stage`, build the crate's tests and record the build, then run them from
    the output: a run of tree `tree`."""
    src, env = "{checkout:s}/src", {"CARGO_TARGET_DIR": "{checkout:s}/target"}
    tests = run_op([sys.executable, "-m", "magnitude_rig.host", "tests", "--list", "{output:s}/tests.json"],
                   stage, job="tests")
    tests["env"] = {"PYTHONPATH": str(AGENT.parents[2])}
    return [*_claim_and_install(stage, tree),
            {"op": "phase", "phase": "build-s"},
            {"op": "run", "argv": ["cargo", "test", "--message-format=short", "--no-run", "--release", "--offline",
                                   "-p", "my-lib"], "cwd": src, "env": env},
            *_built(stage), tests, *after]


def _claim_and_install(stage: Path, tree: str) -> list:
    """Claim side `s`'s checkout for tree `tree`, make room and install `stage` there."""
    src = "{checkout:s}/src"
    return [{"op": "checkout", "side": "s", "source": "src", "tree": tree, "build": BUILD, "output": tree + "-b",
             "fallback": 0, "commit": "c0", "seconds": 60},
            {"op": "hold", "kind": "install"}, {"op": "evict", "side": "s", "host": "here"},
            run_op([sys.executable, str(AGENT), "install", "--from", str(stage), "--to", src], stage, install="s"),
            {"op": "release", "kind": "install"}]


def _built(stage: Path) -> list:
    """Side `s`'s build provenance (a record that keeps no profile executable: the crate has
    none) and the end of its build."""
    proven = str(stage.parent / "provenance.json")
    write = "import json, sys; json.dump({'kept': []}, open(sys.argv[1], 'w'))"
    return [run_op([sys.executable, "-c", write, proven], stage),
            {"op": "built", "side": "s", "release": True, "provenance": proven}]


def _hashes(target: Path) -> set:
    return {found.group(1) for path in (target / "release" / "deps").iterdir()
            if (found := agent.HASHED.search(path.name))}


@needs_cargo
def test_a_build_runs_from_its_output_releases_its_checkout_and_trims_what_no_kept_build_references(tmp_path):
    """A build records the units its cargo steps reference, its log as its own format shows it;
    its tests run from its output, which goes when no live run holds it; the checkout is released
    when the build ends, and the next build of another version in the same checkout leaves only
    its own units; a later run of a tree a live run holds finds the checkout warm and the output
    its own."""
    target = tmp_path / "checkouts" / "slot-0" / "target"
    start(tmp_path, "first", _build_plan(_crate(tmp_path / "v1", "0.1.0"), "t1"))
    assert ended(tmp_path, "first")["failed"] is None
    log = (tmp_path / "runs" / "first" / "logs" / "build-s.log").read_text()
    assert re.search(r"lib/src/lib\.rs:1:\d+: warning: unused variable", log)  # as the step's short format shows it
    assert '"reason"' not in log  # cargo's JSON messages are the runner's own
    first = _hashes(target)
    references = json.loads((target.parent / agent.REFERENCES).read_text())
    assert {unit.rsplit("/", 1)[1] for unit in references[BUILD]["units"]} == first
    assert ended(tmp_path, "first")["codes"] == {"tests": 0}  # its test ran from the output, and passed
    assert not (tmp_path / "outputs" / "t1-b").exists()  # and its output went with it
    stage = _crate(tmp_path / "v2", "0.2.0")
    start(tmp_path, "second", _build_plan(stage, "t2", until_exists(tmp_path / "go")))
    until(tmp_path, "second", "held", timeout=60)
    while not any("released checkout" in e.get("text", "") for e in events(tmp_path, "second")):
        assert "ended" not in {e["event"] for e in events(tmp_path, "second")}, events(tmp_path, "second")
        time.sleep(0.1)
    second = _hashes(target)
    assert second and not second & first  # the 0.1.0 units are gone
    output = tmp_path / "outputs" / "t2-b"
    listed = json.loads((output / agent.TESTS).read_text())  # its library's unit tests and its `here`
    assert len(listed) == 2 and {test["directory"] for test in listed} == {str(output / "src" / "lib")}
    assert all(test["executable"].startswith(str(output / "target" / "release" / "deps")) for test in listed)
    held = {e["kind"] for e in status(tmp_path)["entries"] if e["run"] == "second"}
    assert "output:t2-b" in held and not any(kind.startswith("checkout:") for kind in held)  # released at its build's end
    start(tmp_path, "again", _build_plan(stage, "t2"))
    result = ended(tmp_path, "again")
    assert result["failed"] is None and result["codes"] == {"tests": 0}
    log = (tmp_path / "runs" / "again" / "logs" / "build-s.log").read_text()
    assert "Compiling" not in log  # the checkout the second run released, warm for this tree
    assert output.exists()  # the second run still holds it
    (tmp_path / "go").write_text("")
    assert ended(tmp_path, "second")["failed"] is None
    assert not output.exists()


def test_a_run_sharing_a_checkout_neither_trims_nor_records_it_nor_makes_room(tmp_path):
    """A run sharing a checkout its holder installed records nothing of the build and trims
    nothing: its holder's build owns both. It completes the shared output and releases."""
    stage = tmp_path / "stage"
    (stage / "data").mkdir(parents=True)
    checkout = tmp_path / "checkouts" / "slot-0"
    start(tmp_path, "holder", [*_claim_and_install(stage, "t"), until_exists(tmp_path / "go")])
    while not (checkout / agent.TREE).exists():  # installed, and its output has the tree
        time.sleep(0.05)
    assert (tmp_path / agent.OUTPUTS / "t-b" / "src" / "data").is_dir()
    stale = checkout / "target" / "release" / "deps" / "libstale-0123456789abcdef.rlib"
    stale.parent.mkdir(parents=True)
    stale.write_text("x")
    keeps(checkout, "another build")
    kept = (checkout / agent.REFERENCES).read_text()
    start(tmp_path, "sharer", [*_claim_and_install(tmp_path / "unused", "t"), *_built(stage)])
    result = ended(tmp_path, "sharer")
    assert result["failed"] is None, events(tmp_path, "sharer")
    notes = [e.get("text", "") for e in events(tmp_path, "sharer")]
    assert any("shared with the live runs" in note for note in notes)
    assert any("skipped" in note and "holds this tree" in note for note in notes)
    assert stale.exists() and (checkout / agent.REFERENCES).read_text() == kept  # untrimmed, unrecorded
    assert not (checkout / "size").exists() and not (tmp_path / "rig" / agent.MEASURED).exists()
    assert (tmp_path / agent.OUTPUTS / "t-b" / agent.OUTPUT_DONE).exists()  # the holder still holds it
    (tmp_path / "go").write_text("")
    assert ended(tmp_path, "holder")["failed"] is None


def test_a_suite_run_from_its_output_fails_with_its_first_failure(tmp_path):
    """`tests` runs each listed executable from its directory with CARGO_MANIFEST_DIR naming it,
    and exits with the first failure."""
    host = lambda *argv: subprocess.run([sys.executable, "-m", "magnitude_rig.host", *argv], capture_output=True,
                                        text=True, env=dict(os.environ, PYTHONPATH=str(AGENT.parents[2])))
    (tmp_path / "pkg").mkdir()
    listed = []
    for name, code in (("passes", 0), ("fails", 7)):
        script = tmp_path / name
        script.write_text(f"#!{sys.executable}\nimport os, sys\nprint(os.getcwd() == os.environ['CARGO_MANIFEST_DIR'], "
                          f"sys.argv[1:])\nraise SystemExit({code})\n")
        script.chmod(0o755)
        listed.append({"executable": str(script), "directory": str(tmp_path / "pkg")})
    (tmp_path / "tests.json").write_text(json.dumps(listed))
    ran = host("tests", "--list", str(tmp_path / "tests.json"), "--", "filter")
    assert ran.returncode == 7 and ran.stdout.count("True ['filter']") == 2


def test_the_disk_owner_removes_every_build_output_no_live_run_holds(tmp_path):
    held, left = (tmp_path / agent.OUTPUTS / name for name in ("held", "left"))
    holding, building = session(tmp_path, "holding"), session(tmp_path, "building")
    holding.hold_output("held")
    left.mkdir(parents=True)
    assert building.reclaim()["removed"] == ["outputs/left"]
    assert held.is_dir() and not left.exists()
    assert holding.end_outputs() == ["held"] and not held.exists()
    end(holding)
    end(building)


def test_an_output_a_dead_run_left_incomplete_is_discarded_and_a_complete_one_reused(tmp_path):
    partial, complete = (tmp_path / agent.OUTPUTS / name for name in ("partial", "complete"))
    (partial / "src").mkdir(parents=True)
    (complete / "src").mkdir(parents=True)
    (complete / agent.OUTPUT_DONE).write_text("")
    holding = session(tmp_path, "holding")
    holding.hold_output("partial")
    holding.hold_output("complete")
    assert partial.is_dir() and not (partial / "src").exists()  # held anew, empty
    assert (complete / "src").is_dir()
    end(holding)
