import os
from pathlib import Path

import pytest

from magnitude_engine.models.qwen35.description import AttentionWeights, RecurrentWeights
from magnitude_engine.models.qwen35.formats.gguf import inspect_dense
from magnitude_engine.platform.storage import FileSource
from magnitude_engine.weights.formats.gguf import Metadata, read_directory
from magnitude_engine.weights.identity import ArtifactIdentity

IDENTITY = ArtifactIdentity("b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7")


@pytest.fixture(scope="module")
def directory():
    path = os.environ.get("MAGNITUDE_TEST_GGUF")
    if path is None:
        pytest.skip("set MAGNITUDE_TEST_GGUF to inspect the pinned model artifact")
    with FileSource(Path(path)) as source:
        assert source.digest() == IDENTITY
        return read_directory(source)


@pytest.mark.model
def test_pinned_dense_geometry_and_complete_weight_roles(directory):
    model = inspect_dense(directory, IDENTITY)
    assert model.geometry.hidden == 2560
    assert model.geometry.vocabulary == 248320
    assert model.geometry.context_limit == 262144
    assert len(model.blocks) == 32
    assert sum(isinstance(b.mixer, AttentionWeights) for b in model.blocks) == 8
    assert sum(isinstance(b.mixer, RecurrentWeights) for b in model.blocks) == 24
    assert model.output is model.embedding
    assert model.model_validate_json(model.model_dump_json()) == model


@pytest.mark.model
@pytest.mark.parametrize(
    "key,value",
    [
        ("qwen35.attention.head_count", 15),
        ("qwen35.ssm.inner_size", 4097),
        ("qwen35.rope.dimension_sections", (11, 11, 9, 1)),
        ("qwen35.attention.layer_norm_rms_epsilon", float("inf")),
        ("qwen35.block_count", True),
    ],
)
def test_invalid_geometry_rejected_before_residency(directory, key, value):
    entries = tuple(
        Metadata(name=item.name, value=value) if item.name == key else item
        for item in directory.metadata
    )
    with pytest.raises(ValueError):
        inspect_dense(directory.model_copy(update={"metadata": entries}), IDENTITY)


@pytest.mark.model
def test_missing_wrong_shape_and_unbound_weights_rejected(directory):
    name = "blk.0.attn_qkv.weight"
    original = directory.tensor(name)
    without = tuple(t for t in directory.tensors if t.name != name)
    with pytest.raises(KeyError, match="not found"):
        inspect_dense(directory.model_copy(update={"tensors": without}), IDENTITY)
    wrong = original.model_copy(update={"shape": (8193, 2560)})
    with pytest.raises(ValueError, match="expected"):
        inspect_dense(directory.model_copy(update={"tensors": (*without, wrong)}), IDENTITY)
    extra = original.model_copy(update={"name": "blk.0.unmodeled_branch.weight"})
    with pytest.raises(ValueError, match="unbound"):
        inspect_dense(
            directory.model_copy(update={"tensors": (*directory.tensors, extra)}), IDENTITY
        )
