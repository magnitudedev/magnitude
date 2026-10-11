"""Hosts resolution, sync manifest, ssh command shape, and the local end-to-end flow."""

import json
import shlex
from pathlib import Path

import pytest

from magnitude_remote import flow, sync
from magnitude_remote.build import BuildIdentity
from magnitude_remote.connection import Remote
from magnitude_remote.hosts import HostConfig, Hosts


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
    assert "evaluation/remote/hosts.example.json" in paths
    assert "evaluation/remote/hosts.json" not in paths
    assert any(p.startswith("engine/templates/native/") for p in paths)
    assert not any(p == "native" or p.startswith("native/") for p in paths)
    assert not any("target" in p.split("/")[:-1] for p in paths)


def test_login_shell_command_quoting():
    remote = Remote("h", False, "Linux", "aarch64", "python3", "/r", "/m", "/r/agent/x.py", {"T": "/home/u/t"})
    line = remote.command(["python3", "/r/agent/x.py", "hold", "--who", "a b"], login=True)[-1]
    assert line.startswith('exec "$SHELL" -lc ')
    inner = shlex.split(line)[-1]
    assert shlex.split(inner) == ["exec", "python3", "/r/agent/x.py", "hold", "--who", "a b"]


def test_remote_steps_keep_caches_under_the_cache_root():
    class Session:
        remote = Remote("h", False, "Darwin", "arm64", "python3", "/r", "/m", "/r/agent/x.py", {"T": "/home/u/t"})
        seen = None

        def run(self, argv, *, cwd, env, input, on_output):
            self.seen = (cwd, env)
            return 0

    session = Session()
    flow.RemoteExecutor(session, lambda *a: None).run("job", ["true"], env={"X": "1"}, input=None)
    assert session.seen == ("/r/src", {
        "CARGO_TARGET_DIR": "/r/target",
        "UV_CACHE_DIR": "/r/cache/uv",
        "PIP_CACHE_DIR": "/r/cache/pip",
        "HF_HUB_CACHE": "/r/cache/huggingface/hub",
        "T": "/home/u/t",
        "X": "1",
    })


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


def test_local_steps_get_the_expanded_host_env(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path)
    monkeypatch.setattr(flow, "identify", lambda: BuildIdentity("c" * 40, None))
    defaults = HostConfig("", str(tmp_path / "root"), str(tmp_path / "models"), (("TOOLKIT", "~/kit"),))
    job = flow.Job(tool="t", command=("{python}", "-c", "import os; print(os.environ['TOOLKIT'])"),
                   gate=None)
    lines = []
    flow.execute(job, hosts=Hosts(defaults, {}, {}), sink=lambda s, _, t: lines.append((s, t)),
                 log=lambda m: None)
    assert ("job", str(Path.home() / "kit") + "\n") in lines


def test_placeholders():
    tokens = {"run_dir": "/runs/1", "python": "py"}
    assert flow._substitute(("{python}", "--out={run_dir}/x", "{\"k\": 1}"), tokens) == [
        "py", "--out=/runs/1/x", "{\"k\": 1}"
    ]


def test_local_flow_records_host_facts_trace_and_completes_result(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path)
    write_result = (
        "import json, sys; "
        "json.dump({'schema': 'evaluation-result/1', 'tool': 'harness.time', 'host': {}, "
        "'traces': {'tuning': 'tuning.jsonl'}}, open(sys.argv[1] + '/result.json', 'w')); "
        "print('job ran')"
    )
    job = flow.Job(tool="harness.time", command=("{python}", "-c", write_result, "{run_dir}"), gate=None)
    lines = []
    outcome = flow.execute(job, sink=lambda step, stream, text: lines.append((step, text)), log=lambda m: None)
    assert outcome.code == 0 and outcome.host is None
    assert outcome.results == tmp_path / outcome.run_id
    assert ("job", "job ran\n") in lines
    result = json.loads((outcome.results / "result.json").read_text())
    host = json.loads((outcome.results / "host.json").read_text())
    assert result["host"] == host and host["alias"] is None
    assert result["traces"] == {"tuning": "tuning.jsonl", "thermal": "thermal.jsonl"}
    assert (outcome.results / "thermal.jsonl").read_text().count("\n") >= 2
    assert json.loads((outcome.results / "thermal-summary.json").read_text())["trace"] == "thermal.jsonl"


def test_local_flow_substitutes_trees_root_and_env(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path / "results")
    tree = flow.Tree("extra", tmp_path / "extra", ("a.txt",))
    tree.local.mkdir()
    (tree.local / "a.txt").write_text("tree file")
    hosts = Hosts(HostConfig("", str(tmp_path / "root"), str(tmp_path / "models")), {}, {})
    show = "import os, sys; print(open(sys.argv[1]).read(), sys.argv[2], os.environ['VALUE'])"
    check_env = "import os, sys; assert os.environ['VALUE'] == sys.argv[1]"
    job = flow.Job(
        tool="t",
        command=("{python}", "-c", show, "{tree:extra}/a.txt", "{root}"),
        build=(("{python}", "-c", check_env, "{root}/v"),),
        env=(("VALUE", "{root}/v"),),
        trees=(tree,),
        gate=None,
    )
    lines = []
    outcome = flow.execute(
        job, hosts=hosts, sink=lambda step, stream, text: lines.append((step, text)), log=lambda m: None
    )
    assert outcome.code == 0
    root = str(tmp_path / "root")
    assert ("job", f"tree file {root} {root}/v\n") in lines


def test_local_flow_reports_failed_build(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path)
    job = flow.Job(tool="t", command=("true",), build=(("false",),), gate=None)
    with pytest.raises(flow.StepFailed, match="build exited 1"):
        flow.execute(job, sink=lambda *a: None, log=lambda m: None)


def test_uv_and_build_file_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path)
    identity = BuildIdentity("c" * 40, None)
    monkeypatch.setattr(flow, "identify", lambda: identity)
    job = flow.Job(tool="t", command=("{uv}", "--version"), build=(("cat", "{build_file}"),), gate=None)
    lines = []
    outcome = flow.execute(job, sink=lambda step, stream, text: lines.append((step, text)), log=lambda m: None)
    assert outcome.code == 0
    assert json.loads((outcome.results / "build.json").read_text()) == identity.to_json()
    assert json.loads("".join(t for s, t in lines if s == "build")) == identity.to_json()
    assert any(step == "job" and text.startswith("uv ") for step, text in lines)


def test_missing_probed_tool_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "RESULTS", tmp_path)
    monkeypatch.setattr(flow, "identify", lambda: BuildIdentity("c" * 40, None))
    monkeypatch.setenv("PATH", "/nonexistent")
    job = flow.Job(tool="t", command=("{uv}",), gate=None)
    with pytest.raises(flow.StepFailed, match="uv is not on the job PATH"):
        flow.execute(job, sink=lambda *a: None, log=lambda m: None)
