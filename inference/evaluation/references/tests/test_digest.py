"""A reference's implementation identity is what computes in the sources it runs: text-only edits keep it, a module
it never reaches keeps it, a numeric edit moves it."""

import shutil

import pytest

from magnitude_references import digest

LLAMA = "magnitude_references.families.llama"
PLATFORM = {"device": "cpu", "blas": "fake", "machine": "fake", "rows_exact": True}


def family(module: str) -> type:
    return type("Family", (), {"__module__": module})


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A copy of the package's sources the identity reads, to edit."""
    copy = tmp_path / digest.PACKAGE.name
    shutil.copytree(digest.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(digest, "PACKAGE", copy)
    return copy


def identity(module: str = LLAMA) -> str:
    return digest.implementation(family(module), None, "cpu", PLATFORM)


def test_a_reference_runs_its_own_family_and_the_modules_that_run_every_reference():
    reached = digest.closure(LLAMA)
    assert {LLAMA, "magnitude_references.model", "magnitude_references.run"} <= set(reached)
    assert not any(name.startswith("magnitude_references.families.") and name != LLAMA for name in reached)
    assert not set(reached) & digest.OUTSIDE  # the registry, records, checks and performance options compute nothing
    assert "magnitude_references.families.lfm2" in digest.closure("magnitude_references.families.lfm3")  # it imports it
    assert "magnitude_references.families.glm5next_vision" not in digest.closure("magnitude_references.families.glm5next")


def test_comments_and_docstrings_leave_the_identity(package):
    before = identity()
    model = package / "model.py"
    source = model.read_text()
    edited = source.replace('"""Shared float32 building blocks', '"""Other words for the shared float32 building blocks', 1)
    edited = edited.replace("ROW_TILE = 128\n", "ROW_TILE = 128  # a comment\n# and a line of its own\n", 1)
    edited = edited.replace('    """A family reference:', '    """Another docstring for a family reference:', 1)
    assert edited != source
    model.write_text(edited)
    assert identity() == before


def test_a_module_no_reference_of_the_family_reaches_leaves_its_identity(package):
    before = identity()
    (package / "families" / "unrelated.py").write_text("SCALE = 3.0\n")
    glm = package / "families" / "glm5next.py"
    glm.write_text(glm.read_text() + "\nSCALE = 3.0\n")  # another family's edit
    assert identity() == before


def test_a_numeric_edit_moves_the_identity(package):
    before, glm = identity(), identity("magnitude_references.families.glm5next")
    model = package / "model.py"
    model.write_text(model.read_text().replace("ROW_TILE = 128\n", "ROW_TILE = 64\n", 1))
    assert identity() != before and identity("magnitude_references.families.glm5next") != glm
    llama = package / "families" / "llama.py"
    moved = identity()
    llama.write_text(llama.read_text() + "\nSCALE = 3.0\n")
    assert identity() != moved
