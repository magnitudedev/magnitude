import ast
from pathlib import Path


def test_ops_does_not_import_engine_and_tilelang_stays_internal():
    root = Path(__file__).parents[2] / "src" / "ops"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name.startswith("engine") for name in imports), path
        relative = path.relative_to(root)
        if relative.parts[0] not in {"kernels", "runtime"} and relative.as_posix() not in {
            "compiler/program.py", "compiler/streaming.py",
        }:
            assert not any(
                name == "tilelang" or name.startswith("tilelang.") for name in imports
            ), path
        assert not any(
            name == "tilelang.tvm" or name.startswith("tilelang.tvm.") for name in imports
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "exec"
            for node in ast.walk(tree)
        ), path


def test_package_root_contains_only_public_surface_and_representations():
    root = Path(__file__).parents[2] / "src" / "ops"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py", "representations.py", "formula.py", "operation.py", "binding.py", "isolation.py",
    }
