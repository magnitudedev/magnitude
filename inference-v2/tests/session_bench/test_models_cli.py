import json
import shlex
import subprocess
import sys

import pytest

from session_bench import models
from session_bench.cli import contexts, parser
from session_bench.results import public_command
from session_bench.runner import capacity


def test_aliases_are_local_and_paths_are_relative_to_file(tmp_path, monkeypatch):
    (tmp_path / "models.local.json").write_text(json.dumps({"mine": {"mlx": "models/test"}}))
    monkeypatch.chdir(tmp_path.parent)
    selected = models.select(tmp_path, ["mine"], ["magnitude", "omlx"], [])
    assert [t.reference for t in selected] == [str(tmp_path / "models/test")] * 2
    with pytest.raises(ValueError, match="no artifact"):
        models.select(tmp_path, ["mine"], ["llama.cpp"], [])
    with pytest.raises(ValueError, match="unknown model"):
        models.select(tmp_path, ["not_registered"], [], [])


def test_alias_edit_does_not_change_reproduction(tmp_path):
    path = tmp_path / "models.local.json"
    path.write_text(json.dumps({"mine": {"mlx": "model with spaces;$(false)"}}))
    selected = models.select(tmp_path, ["mine"], [], [])
    command = public_command(selected, ("context",), (1024,), ("simple-python",), 1)
    path.unlink()
    arguments = parser().parse_args(shlex.split(command)[4:])
    assert models.select(tmp_path, arguments.model, arguments.engine, arguments.target) == selected
    assert "mine" not in command


def test_no_low_output_or_project_or_generic_override_flags():
    for flag in ("--max-output-tokens", "--max-tokens", "--project", "--set"):
        with pytest.raises(SystemExit):
            parser().parse_args(["run", "--model", "mine", flag, "1"])
    with pytest.raises(SystemExit):
        parser().parse_args(["run", "--model", "mine", "--engine", "icn"])


def test_capacity_reserves_full_output_budget():
    assert capacity([{"a": 1000}, {"a": 1200}], [40000, 50000]) == 34048
    with pytest.raises(ValueError, match="headroom"):
        capacity([{"a": 1000}], [33000])
    with pytest.raises(ValueError):
        capacity([{"a": True}], [99999])


def test_hub_references_are_pinned():
    assert models.parse_hub("hf:owner/repo@" + "a" * 40 + "#model.gguf")[2] == "model.gguf"
    for value in ("hf:owner/repo@main", "hf:owner/repo@" + "a" * 40 + "#../other"):
        with pytest.raises(ValueError):
            models.parse_hub(value)


def test_artifact_identity_and_drift(artifact_path):
    target = models.Target(engine="magnitude", reference=str(artifact_path))
    artifact = models.prepare(target)
    assert artifact.context_limit == 262144
    assert len(artifact.files) == 2
    artifact.verify_unchanged()
    (artifact_path / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        artifact.verify_unchanged()


def test_discovery_imports_no_engine_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import session_bench.cli; "
            "assert 'mlx' not in sys.modules; assert 'mlx.core' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'magnitude_engine' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_context_units():
    assert contexts("16k,4096,4k") == (4096, 16384)
    with pytest.raises(ValueError):
        contexts("0")


def test_new_model_file_invalidates_prepared_artifact(artifact_path):
    artifact = models.prepare(models.Target(engine="magnitude", reference=str(artifact_path)))
    (artifact_path / "chat_template.jinja").write_text("changed template")
    with pytest.raises(ValueError, match="changed"):
        artifact.verify_unchanged()


def test_gguf_model_identity_comes_from_metadata(tmp_path):
    from gguf import GGUFWriter

    path = tmp_path / "tiny.gguf"
    writer = GGUFWriter(str(path), "llama")
    writer.add_context_length(65536)
    writer.add_file_type(1)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    artifact = models.prepare(models.Target(engine="llama.cpp", reference=str(path)))
    assert artifact.context_limit == 65536
    assert artifact.metadata == {"architecture": "llama", "file_type": 1}
    assert len(artifact.files[0].sha256) == 64
    # Hub snapshots symlink a .gguf name to an extensionless content-addressed blob.
    blob = tmp_path / "content-hash"
    path.rename(blob)
    path.symlink_to(blob)
    selected = models.select(tmp_path, [], [], [f"llama.cpp={path}"])[0]
    assert models.prepare(selected).files[0].sha256 == artifact.files[0].sha256
