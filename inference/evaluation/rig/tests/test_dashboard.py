"""The dashboard's host stream, end to end: the deployed host package's `watch` reports the
host's facts, its holds and a probe reading, and ends when the dashboard closes it."""

import threading
import time

from magnitude_rig.connection import AgentSession, connect
from magnitude_rig.dashboard import Watch


def test_watch_reports_a_hosts_holds_and_readings_until_closed(rig):
    hosts, _, _ = rig
    config = hosts.host("alpha")
    busy = AgentSession(connect(config), who="other", run="busy", purpose="p", estimate=3600)
    stopping = threading.Event()
    try:
        busy.hold("device", lambda blocking: None)
        watch = Watch(config, 0.2, stopping)
        deadline = time.monotonic() + 30
        while (view := watch.view(history=True))["state"] != "live":
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
        busy.close()
