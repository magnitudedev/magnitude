import json

import pytest

from engine.composition import Blueprint, blueprint
from engine.composition.graph import Catalog, dumps, loads
from engine.platform.backend import Backend


@blueprint
class Choice(Blueprint[object]):
    backend: Backend
    candidates: tuple[Backend, ...]
    optional: Backend | None

    @staticmethod
    def implementation():
        return object


def test_declared_enums_roundtrip_without_importing_types_from_data():
    recipe = Choice(
        backend=Backend.METAL, candidates=(Backend.METAL, Backend.LLVM), optional=Backend.CUDA
    )
    catalog = Catalog([Choice])
    restored = loads(dumps(recipe), catalog)
    assert isinstance(restored, Choice)
    assert restored.backend is Backend.METAL
    assert restored.candidates == recipe.candidates
    assert restored.optional is Backend.CUDA
    document = json.loads(dumps(recipe))
    document["nodes"][0]["fields"]["backend"]["enum"] = "untrusted.module.Class"
    with pytest.raises(ValueError, match="declared enum"):
        loads(json.dumps(document), catalog)
