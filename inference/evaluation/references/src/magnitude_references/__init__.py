"""Independent float32 model-family references: the true outputs for rendered fixtures.

Python API: `build_reference(rendered, artifacts, output, Options(device))` writes the contract logits
output, `reference.json` (with the sanity gate) and `result.json`; the CLI is `magnitude-references`.
The package imports lazily: `digest`, `contract` and `sanity` load without torch.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {"Options": "magnitude_references.run", "build_reference": "magnitude_references.run",
            "Bounds": "magnitude_references.sanity"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'magnitude_references' has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)
