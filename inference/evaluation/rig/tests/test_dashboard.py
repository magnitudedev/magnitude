"""The dashboard's host stream, end to end: the deployed host package's `watch` reports the
host's facts, its holds and a probe reading, and ends when the dashboard closes it."""

import threading
import time

from waiting import until_exists

from magnitude_rig.connection import HostRun, connect
from magnitude_rig.dashboard import Watch


def test_watch_reports_a_hosts_holds_and_readings_until_closed(rig):
    hosts, _, tmp_path = rig
    config = hosts.host("alpha")
    release = tmp_path / "release"
    plan = {
        "info": {"who": "other", "purpose": "p", "estimate": 3600},
        "ops": [
            {"op": "hold", "kind": "device"},
            {"op": "run", "argv": until_exists(release),
             "cwd": str(tmp_path), "env": {}},
        ],
    }
    with HostRun.staging(connect(config), "busy") as start:
        busy = start(plan)
    stopping = threading.Event()
    try:
        watch = Watch(config, 0.2, stopping)
        deadline = time.monotonic() + 30

        def held(view: dict) -> bool:
            entries = view["state"] == "live" and view["sample"]["status"]["entries"]
            return bool(entries) and entries[0]["state"] == "held"

        while not held(view := watch.view(history=True)):
            assert view["state"] != "unreachable", view["error"]
            assert time.monotonic() < deadline, view
            time.sleep(0.1)
        assert view["facts"]["alias"] == "alpha"
        [entry] = view["sample"]["status"]["entries"]
        assert (entry["run"], entry["kind"], entry["state"]) == ("busy", "device", "held")
        assert "system" in view["sample"]["reading"]["memory_bytes"]
        assert view["history"][-1]["received"] == view["received"]
        stopping.set()
        watch.close()
        watch.join()
        assert watch.view(history=True)["state"] == "live"  # closing is not a failure to report
    finally:
        release.write_text("")
        assert [event["event"] for event in busy.follow(0)][-1] == "ended"
