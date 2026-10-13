"""The independent family references by GGUF architecture."""
from __future__ import annotations

from pathlib import Path

import torch

from magnitude_references.families.gemma4 import Gemma4Reference
from magnitude_references.families.lfm2 import Lfm2Reference
from magnitude_references.families.llama import LlamaReference
from magnitude_references.families.muse import MuseReference
from magnitude_references.families.nemotron_h import NemotronHReference
from magnitude_references.families.qwen35 import Qwen35Reference
from magnitude_references.families.qwen4exp import Qwen4ExpReference
from magnitude_references.gguf import Package
from magnitude_references.model import Reference, Weights

# GGUF `general.architecture` -> (family name, reference). The family name is the record's `family`
# and the logits producer `reference:<family>`.
REFERENCES: dict[str, tuple[str, type[Reference]]] = {
    "llama": ("llama", LlamaReference),
    "lfm2": ("lfm2", Lfm2Reference),
    "lfm2moe": ("lfm2", Lfm2Reference),
    "muse-glimmer": ("muse", MuseReference),
    "gemma4": ("gemma4", Gemma4Reference),
    "nemotron_h_moe": ("nemotron_h", NemotronHReference),
    "qwen35": ("qwen35", Qwen35Reference),
    "qwen35moe": ("qwen35", Qwen35Reference),
    "qwen4exp": ("qwen4exp", Qwen4ExpReference),
}


def family_of(package: Package) -> str:
    if package.architecture not in REFERENCES:
        raise ValueError(f"no reference for architecture {package.architecture!r}")
    return REFERENCES[package.architecture][0]


def open_reference(model: Path | Package, device: torch.device, cache: bool, layers: int | None = None) -> Reference:
    """The family reference of a GGUF. `cache=False` streams: every layer's weights are dequantized
    when the layer runs and dropped after."""
    package = model if isinstance(model, Package) else Package.local(Path(model))
    family_of(package)
    return REFERENCES[package.architecture][1](package, Weights(package, device, cache), layers)
