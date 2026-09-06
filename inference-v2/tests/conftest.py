"""Optional reference imports remain outside the engine's dependency graph."""

import importlib
import os
import sys

import pytest


@pytest.fixture(scope="session")
def poc_module():
    reference = os.environ.get("MLX_POC_REFERENCE")
    if reference is None:
        pytest.skip("set MLX_POC_REFERENCE to the reviewed PoC engine directory")
    sys.path.insert(0, reference)
    try:
        yield lambda name: importlib.import_module(f"mlxengine.{name}")
    finally:
        sys.path.remove(reference)
