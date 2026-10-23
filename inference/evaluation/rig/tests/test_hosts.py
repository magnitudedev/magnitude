"""Hosts resolution, the sync manifest and the ssh command shape."""

import json
import shlex
from pathlib import Path

import pytest

from magnitude_rig import sync
from magnitude_rig.connection import Remote
from magnitude_rig.hosts import HostConfig, Hosts


def test_example_hosts_file():
    example = Path(__file__).resolve().parents[1] / "hosts.example.json"
    hosts = Hosts.load(example)
    pool = hosts.resolve("gpu-pool")
    assert [h.alias for h in pool] == ["host-a", "host-b"]
    assert all(h.root == "/path/to/scratch/magnitude-evaluation" for h in pool)
    assert hosts.resolve("unlisted") == (
        HostConfig("unlisted", "~/.cache/magnitude-evaluation", "~/.magnitude/models"),
    )
    assert hosts.resolve("host-b") == (pool[1],)


def test_missing_hosts_file_means_defaults_only(tmp_path):
    hosts = Hosts.load(tmp_path / "hosts.json")
    assert hosts.pools == {} and hosts.known() == ()
    assert hosts.resolve("any") == (
        HostConfig("any", "~/.cache/magnitude-evaluation", "~/.magnitude/models"),
    )


def test_pool_and_host_names_must_not_collide(tmp_path):
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({
        "defaults": {"root": "r", "models": "m"},
        "pools": {"a": ["b"]},
        "hosts": {"a": {"root": "x", "models": "y"}},
    }))
    with pytest.raises(ValueError, match="both a pool and a host"):
        Hosts.load(path)


def test_manifest_excludes_native_submodule_and_build_outputs():
    paths = sync.manifest()
    assert "Cargo.toml" in paths
    assert "evaluation/rig/hosts.example.json" in paths
    assert any(p.startswith("engine/templates/native/") for p in paths)
    assert not any(p == "native" or p.startswith("native/") for p in paths)
    assert not any("target" in p.split("/")[:-1] for p in paths)


def test_login_shell_command_quoting():
    remote = Remote("h", False, False, "Linux", "aarch64", "python3", "/r", "/m", "/r/agent/x.py",
                    {"T": "/home/u/t"}, ("cpu", "cuda"), True)
    line = remote.command(["python3", "/r/agent/x.py", "session", "--who", "a b"], login=True)[-1]
    assert line.startswith('exec "$SHELL" -lc ')
    inner = shlex.split(line)[-1]
    assert shlex.split(inner) == ["exec", "python3", "/r/agent/x.py", "session", "--who", "a b"]


def test_host_env_from_the_hosts_file(tmp_path):
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({
        "defaults": {"root": "r", "models": "m"},
        "hosts": {"gpu": {"env": {"SEISMIC_NVRTC_DIRECTORY": "~/nvrtc"}}},
    }))
    hosts = Hosts.load(path)
    assert hosts.host("gpu") == HostConfig("gpu", "r", "m", (("SEISMIC_NVRTC_DIRECTORY", "~/nvrtc"),))
    assert hosts.host("other").env == ()
    example = Hosts.load(Path(__file__).resolve().parents[1] / "hosts.example.json")
    assert dict(example.host("host-a").env) == {"SEISMIC_NVRTC_DIRECTORY": "/path/to/nvrtc"}


def test_this_machine_is_a_host_reached_without_ssh(tmp_path):
    hosts = Hosts.load(tmp_path / "hosts.json")
    assert hosts.host("local").local and not hosts.host("gpu").local
    path = tmp_path / "hosts.json"
    path.write_text(json.dumps({"defaults": {"root": "r", "models": "m"},
                                "hosts": {"mac": {"local": True}}, "pools": {"metal": ["mac", "m4"]}}))
    assert [h.local for h in Hosts.load(path).resolve("metal")] == [True, False]
