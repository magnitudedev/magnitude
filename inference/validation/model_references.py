"""The independent family references by GGUF architecture."""
from __future__ import annotations

from pathlib import Path

import torch

from reference_gguf import Package
from reference_model import Reference, Weights
from gemma4_reference import Gemma4Reference
from lfm2_reference import Lfm2Reference
from llama_reference import LlamaReference
from muse_reference import MuseReference
from nemotron_h_reference import NemotronHReference

REFERENCES: dict[str, type[Reference]] = {
    "llama": LlamaReference,
    "lfm2": Lfm2Reference,
    "lfm2moe": Lfm2Reference,
    "muse-glimmer": MuseReference,
    "gemma4": Gemma4Reference,
    "nemotron_h_moe": NemotronHReference,
}


def open_reference(model: Path | Package, device: torch.device, cache: bool, layers: int | None = None) -> Reference:
    package = model if isinstance(model, Package) else Package.local(Path(model))
    if package.architecture not in REFERENCES:
        raise ValueError(f"no reference for architecture {package.architecture!r}")
    return REFERENCES[package.architecture](package, Weights(package, device, cache), layers)
