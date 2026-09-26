import json
import subprocess
from types import SimpleNamespace

import pytest

from magnitude_engine import host_info


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(host_info.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(host_info.platform, "node", lambda: "benchmark-mac")
    monkeypatch.setattr(host_info.platform, "mac_ver", lambda: ("15.7.9", (), "arm64"))
    monkeypatch.setattr(host_info.psutil, "cpu_count", lambda **kwargs: 14)
    monkeypatch.setattr(
        host_info.psutil, "virtual_memory", lambda: SimpleNamespace(total=48 << 30),
    )


def test_mac_hardware_normalizes_os_values_and_excludes_unrelated_details(mac, monkeypatch):
    def read(*command):
        if command[0].endswith("sysctl"):
            return "Mac16,11\nApple M4 Pro\n14\n51539607552"
        return json.dumps({"SPDisplaysDataType": [{
            "sppci_model": "Apple M4 Pro", "sppci_cores": "20",
            "spdisplays_ndrvs": [{"display_serial": "not benchmark evidence"}],
        }]})

    monkeypatch.setattr(host_info, "_read", read)
    record = host_info.capture_hardware().model_dump(mode="json")
    assert record == {
        "hostname": "benchmark-mac", "os": "Darwin", "os_version": "15.7.9",
        "model": "Mac16,11", "chip": "Apple M4 Pro", "cpu_cores": 14,
        "memory_bytes": 51539607552, "gpus": [{"name": "Apple M4 Pro", "cores": 20}],
        "errors": [],
    }
    assert host_info.HardwareInfo.model_validate_json(json.dumps(record)).memory_bytes == 48 << 30


@pytest.mark.parametrize("failure", [
    FileNotFoundError("probe unavailable"),
    subprocess.TimeoutExpired("probe", 10),
    ValueError("invalid probe output"),
])
def test_failed_probes_preserve_host_identity_and_available_hardware(mac, monkeypatch, failure):
    def read(*command):
        raise failure

    monkeypatch.setattr(host_info, "_read", read)
    info = host_info.capture_hardware()
    assert info.hostname == "benchmark-mac"
    assert info.memory_bytes == 48 << 30
    assert info.cpu_cores == 14
    assert info.gpus is None
    assert len(info.errors) == 2


def test_hardware_probe_is_bounded(monkeypatch):
    def run(command, **kwargs):
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 10}
        return SimpleNamespace(stdout="value\n")

    monkeypatch.setattr(host_info.subprocess, "run", run)
    assert host_info._read("probe") == "value"
