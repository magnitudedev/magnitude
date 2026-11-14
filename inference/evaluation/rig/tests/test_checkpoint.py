"""A whole GGUF set the host holds is acquired by the bytes of its shards: named by the SHA-256 of their bytes (hashed once and
recorded beside the set), refused where a shard is absent, or a shard's provenance record is malformed or names other
bytes than its own, never fetched or copied."""

import json

import pytest

from magnitude_rig.host import checkpoint, identity


def make_set(directory, count=3, stem="Fixture-FP8", payload=b"w"):
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (directory / f"{stem}-{index:05d}-of-{count:05d}.gguf").write_bytes(b"GGUF" + payload * index)
    return directory


def test_a_set_is_named_by_its_tensor_bytes_so_two_sets_of_one_header_are_two_models(tmp_path):
    first, second = make_set(tmp_path / "a"), make_set(tmp_path / "b")
    model, digest, paths = checkpoint.model_of(str(first))
    assert model == f"fixture-fp8:checkpoint-{digest[:16]}" and paths[0].endswith("-00001-of-00003.gguf")
    assert checkpoint.model_of(str(second))[1] == digest  # the same bytes under another directory: the same set
    shard = second / "Fixture-FP8-00002-of-00003.gguf"
    shard.write_bytes(shard.read_bytes()[:-1] + b"X")  # the same size, another byte
    assert checkpoint.model_of(str(second))[1] != digest


def test_a_lone_file_is_a_set_of_one_and_a_projector_is_no_part_of_a_set(tmp_path):
    directory = tmp_path / "slice"
    directory.mkdir()
    (directory / "GLM-layers.gguf").write_bytes(b"GGUF slice")
    (directory / "mmproj-GLM-BF16.gguf").write_bytes(b"GGUF projector")
    model, digest, paths = checkpoint.model_of(str(directory))
    assert [p.rsplit("/", 1)[1] for p in paths] == ["GLM-layers.gguf"] and model.startswith("glm-layers:checkpoint-")
    assert digest == checkpoint.identity(paths)
    assert checkpoint.model_of(str(directory))[2] == paths


def test_identity_is_the_bytes_hashed_once_and_remembered_and_a_corrupt_memory_is_refused(tmp_path, monkeypatch):
    directory = make_set(tmp_path / "set", count=2)
    first, second = sorted(directory.iterdir())
    hashed = []
    real = identity.sha256
    monkeypatch.setattr(identity, "sha256", lambda path: hashed.append(path) or real(path))
    digest = checkpoint.identity([str(first), str(second)])
    assert hashed == [str(first), str(second)]
    hashed.clear()
    assert checkpoint.identity([str(first), str(second)]) == digest and hashed == []  # remembered beside the set
    second.write_bytes(second.read_bytes() + b"more")  # a new size: read again, a new identity
    assert checkpoint.identity([str(first), str(second)]) != digest
    remembered = directory / identity.REMEMBERED
    for corrupt in ("{", "[]", json.dumps({first.name: {"size": 1}})):
        remembered.write_text(corrupt)
        with pytest.raises(identity.Refused, match=identity.REMEMBERED):
            checkpoint.identity([str(first), str(second)])


def test_a_shard_whose_provenance_record_is_malformed_or_names_other_bytes_refuses_the_set(tmp_path):
    """The record is catalog-tool's per-file one (`<file>.json`, its `sha256` the file's own), its one writer."""
    directory = tmp_path / "slice"
    directory.mkdir()
    target = directory / "GLM-layers.gguf"
    target.write_bytes(b"GGUF slice")
    record = directory / "GLM-layers.json"
    record.write_text(json.dumps({"main_layers": [0], "sha256": identity.sha256(target)}))  # a true claim
    assert checkpoint.model_of(str(directory))[1] == checkpoint.identity([str(target)])
    record.write_text(json.dumps({"sha256": "0" * 64}))
    with pytest.raises(checkpoint.Refused, match="provenance record names " + "0" * 64):
        checkpoint.model_of(str(directory))
    for malformed in ("{", json.dumps({"sha256": "XYZ"}), json.dumps({"main_layers": [0]})):
        record.write_text(malformed)
        with pytest.raises(checkpoint.Refused, match="GLM-layers.json"):
            checkpoint.model_of(str(directory))


def test_a_set_with_a_missing_shard_is_refused_naming_it(tmp_path):
    directory = make_set(tmp_path / "set")
    (directory / "Fixture-FP8-00002-of-00003.gguf").unlink()
    with pytest.raises(checkpoint.Refused, match=r"lacks shards \[2\] of 3"):
        checkpoint.shards(str(directory))
    with pytest.raises(checkpoint.Refused, match="not a directory"):
        checkpoint.shards(str(tmp_path / "absent"))


def test_the_acquisition_object_names_the_first_shard_and_the_set_as_a_model_of_its_own_content(tmp_path):
    directory = make_set(tmp_path / "set")
    artifacts = checkpoint.acquire(str(directory), str(tmp_path / "artifacts.json"))
    assert json.loads((tmp_path / "artifacts.json").read_text()) == artifacts
    assert artifacts["target"].endswith("Fixture-FP8-00001-of-00003.gguf")
    digest = checkpoint.model_of(str(directory))[1]  # the set's identity names the model
    assert artifacts["model"] == f"fixture-fp8:checkpoint-{digest[:16]}"
    assert artifacts["target_sha256"] == identity.sha256(artifacts["target"])  # the target file's own, as a catalog's
    assert artifacts["drafter"] is None and artifacts["projector"] is None
