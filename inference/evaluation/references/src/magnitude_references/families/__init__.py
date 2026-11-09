"""The independent family references by GGUF architecture."""
from __future__ import annotations

from pathlib import Path

import torch

from magnitude_references.families.deepseek41 import DeepSeek41Reference
from magnitude_references.families.gemma4 import Gemma4Reference
from magnitude_references.families.lfm2 import Lfm2Reference
from magnitude_references.families.lfm3 import Lfm3Reference
from magnitude_references.families.llama import LlamaReference
from magnitude_references.families.muse import MuseReference
from magnitude_references.families.nemotron_h import NemotronHReference
from magnitude_references.families.qwen35 import Qwen35Reference
from magnitude_references.families.qwen4exp import Qwen4ExpReference
from magnitude_references.gguf import Package
from magnitude_references.model import Reference, Weights
from magnitude_references.perturbation import Perturbation

# GGUF `general.architecture` -> (family name, reference). The family name is the record's `family`
# and the logits producer `reference:<family>`.
REFERENCES: dict[str, tuple[str, type[Reference]]] = {
    "deepseek41": ("deepseek41", DeepSeek41Reference),
    "llama": ("llama", LlamaReference),
    "lfm2": ("lfm2", Lfm2Reference),
    "lfm2moe": ("lfm2", Lfm2Reference),
    "lfm3": ("lfm3", Lfm3Reference),
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


def open_reference(model: Path | Package, device: torch.device, cache: bool, layers: int | None = None,
                   variant: str | None = None) -> Reference:
    """The family reference of a GGUF, in its model numerics, the named measurement `variant`
    (`Reference.variants`), or its numerics over storage-resolution-perturbed stored values (`perturbed:<seed>`,
    `perturbation.py`). `cache=False` streams: every layer's weights are dequantized when the layer runs and
    dropped after."""
    package = model if isinstance(model, Package) else Package.local(Path(model))
    family_of(package)
    family = REFERENCES[package.architecture][1]
    perturbation = Perturbation.of(variant)
    if variant is not None and perturbation is None and variant not in family.variants:
        raise ValueError(f"{package.architecture} has no reference variant {variant!r}: "
                         f"{sorted(family.variants)} or perturbed:<seed>")
    options = family.variants[variant] if variant in family.variants else {}
    return family(package, Weights(package, device, cache, perturbation), layers, **options)
