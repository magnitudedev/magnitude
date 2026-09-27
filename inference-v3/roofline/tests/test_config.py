import hashlib
import json

import pytest

from roofline.config import Configuration
from roofline.contracts import Experiment, Model
from roofline.integrations.magnitude import verify_artifact


def test_locations_resolve_same_bytes_and_reject_wrong_copy(tmp_path):
    paths = [tmp_path / name for name in ("first.gguf", "renamed.gguf")]
    for path in paths:
        path.write_bytes(b"model artifact")
    model = Model(
        sha256=hashlib.sha256(b"model artifact").hexdigest(),
        locations=dict(zip(("local", "remote"), map(str, paths), strict=True)),
    )
    for target, path in zip(("local", "remote"), paths, strict=True):
        assert verify_artifact(model, target, "qwen:gguf:q4") == path
    paths[1].write_bytes(b"different model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_artifact(model, "remote", "qwen:gguf:q4")


def test_model_locations_are_validated_against_targets(tmp_path):
    folder = tmp_path / "roofline"
    folder.mkdir()
    models = {
        "models": {
            "qwen:gguf:q4": {
                "sha256": "a" * 64,
                "locations": {"local": "/models/qwen.gguf"},
            }
        }
    }
    targets = {
        "targets": {
            name: {"connection": {"kind": "local"}, "device": {"backend": "cpu"}}
            for name in ("local", "other")
        }
    }
    (folder / "models.json").write_text(json.dumps(models))
    (folder / "targets.json").write_text(json.dumps(targets))
    config = Configuration(tmp_path)
    config.validate(Experiment(model="qwen:gguf:q4"))
    with pytest.raises(ValueError, match="no artifact location"):
        config.validate(Experiment(model="qwen:gguf:q4", targets=("other",)))
    del targets["targets"]["local"]
    (folder / "targets.json").write_text(json.dumps(targets))
    with pytest.raises(ValueError, match="undefined targets"):
        Configuration(tmp_path)


def test_model_location_cannot_depend_on_worker_current_directory():
    with pytest.raises(ValueError, match="absolute file path"):
        Model(sha256="a" * 64, locations={"local": "model.gguf"})
