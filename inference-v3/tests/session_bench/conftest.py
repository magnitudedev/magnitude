import json

import pytest

from benchmark_fixtures.interactions import ExpectedCall, Interaction


@pytest.fixture
def interaction():
    return Interaction(
        id="simple_python_0",
        category="simple-python",
        messages=[{"role": "user", "content": "Call echo with value 7."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo an integer",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        expected=[ExpectedCall(name="echo", arguments={"value": [7]})],
        provenance={"commit": "a" * 40},
    )


@pytest.fixture
def artifact_path(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 262144, "model_type": "test"})
    )
    (path / "weights.safetensors").write_bytes(b"test artifact, never loaded by an engine")
    return path
