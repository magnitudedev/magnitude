import json

import pytest
from bench_support import tools_fixture
from magnitude_bench.models import Acquisition
from magnitude_bench.policy import LANES


@pytest.fixture
def fixtures():
    return [tools_fixture(lane) for lane in range(LANES + 1)]


@pytest.fixture
def artifact_path(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 262144, "model_type": "test"})
    )
    (path / "weights.safetensors").write_bytes(b"test artifact, never loaded by an engine")
    return path


@pytest.fixture
def acquisition(tmp_path):
    target = tmp_path / "target.gguf"
    target.write_bytes(b"never loaded")
    return Acquisition(
        model="qwen3.5-4b:gguf:q4",
        target=target,
        target_sha256="0" * 64,
        drafter={"method": "dflash", "path": tmp_path / "draft.gguf", "sha256": "1" * 64},
        projector=None,
    )


@pytest.fixture
def artifacts(tmp_path, acquisition):
    path = tmp_path / "artifacts.json"
    path.write_text(acquisition.model_dump_json())
    return path
