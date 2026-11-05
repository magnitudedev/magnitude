"""A fresh job reads its model's files from clones on macOS, as they are elsewhere, and leaves
no clone behind; on Linux it starts once every page of them is dropped from the cache."""

import json
import sys
from pathlib import Path

from magnitude_rig.host import fresh


def test_a_split_target_is_every_shard_and_a_file_is_itself():
    split = Path("/m/model-00001-of-00003.gguf")
    assert [path.name for path in fresh.target_files(split)] == [
        "model-00001-of-00003.gguf", "model-00002-of-00003.gguf", "model-00003-of-00003.gguf"]
    assert fresh.target_files(Path("/m/model.gguf")) == [Path("/m/model.gguf")]


def test_the_job_reads_its_target_from_clones_it_removes_after(tmp_path):
    for index in (1, 2):
        (tmp_path / f"m-0000{index}-of-00002.gguf").write_bytes(b"weights")
    target = tmp_path / "m-00001-of-00002.gguf"
    artifacts, output, clones = tmp_path / "a.json", tmp_path / "out.json", tmp_path / "clones"
    artifacts.write_text(json.dumps({"target": str(target), "drafter": None}))
    seen = tmp_path / "seen"
    probe = ("import json, os, sys; t = json.load(open(sys.argv[1]))['target']; "
             "open(sys.argv[2], 'w').write(json.dumps([t, sorted(os.listdir(os.path.dirname(t)))]))")
    code = fresh.run(artifacts, output, clones, [sys.executable, "-c", probe, str(output), str(seen)])
    assert code == 0
    read, listed = json.loads(seen.read_text())
    if sys.platform == "darwin":
        assert read == str(clones / target.name)
        assert listed == ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]
    else:
        assert read == str(target)
    assert not clones.exists()


def test_on_linux_the_job_starts_with_its_targets_pages_dropped(tmp_path, monkeypatch):
    for index in (1, 2):
        (tmp_path / f"m-0000{index}-of-00002.gguf").write_bytes(b"weights")
    target = tmp_path / "m-00001-of-00002.gguf"
    artifacts, output, clones = tmp_path / "a.json", tmp_path / "out.json", tmp_path / "clones"
    artifacts.write_text(json.dumps({"target": str(target), "drafter": None}))
    events = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(fresh, "drop_cached", lambda path: events.append(("dropped", path.name)))
    monkeypatch.setattr(fresh.subprocess, "Popen", lambda argv: events.append(("started",)) or _Ended())
    assert fresh.run(artifacts, output, clones, ["job"]) == 0
    assert events == [("dropped", "m-00001-of-00002.gguf"), ("dropped", "m-00002-of-00002.gguf"), ("started",)]
    assert json.loads(output.read_text())["target"] == str(target)


class _Ended:
    """A child that has ended well."""

    def wait(self):
        return 0

    def poll(self):
        return 0
