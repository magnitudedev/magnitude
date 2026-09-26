"""Load V3's reference modules without its GPU runtime facade."""
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
import sys


def activate(source: Path):
    path = source.resolve(strict=True) / "src/ops"
    if "ops" in sys.modules:
        raise RuntimeError("V3 reference package must be selected before importing ops")
    # Only omit ops/__init__.py, which eagerly imports TileLang kernels/runtime.
    # All numerical modules and their relative imports execute unmodified V3 source.
    spec = ModuleSpec("ops", loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    sys.modules["ops"] = module_from_spec(spec)
