"""The independent family references by GGUF architecture, and their vision references by projector type."""
from __future__ import annotations

from pathlib import Path

import torch

from magnitude_references.families.deepseek41 import DeepSeek41Reference
from magnitude_references.families.gemma4 import Gemma4Reference
from magnitude_references.families.glm5next import Glm5NextReference
from magnitude_references.families.glm5next_vision import Glm5NextVisionReference
from magnitude_references.families.lfm2 import Lfm2Reference
from magnitude_references.families.lfm3 import Lfm3Reference
from magnitude_references.families.llama import LlamaReference
from magnitude_references.families.muse import MuseReference
from magnitude_references.families.nemotron_h import NemotronHReference
from magnitude_references.families.qwen35 import Qwen35Reference
from magnitude_references.families.qwen4exp import Qwen4ExpReference
from magnitude_references.gguf import Package
from magnitude_references.model import Reference

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
    "glm5-next": ("glm5next", Glm5NextReference),
    "nemotron_h_moe": ("nemotron_h", NemotronHReference),
    "qwen35": ("qwen35", Qwen35Reference),
    "qwen35moe": ("qwen35", Qwen35Reference),
    "qwen4exp": ("qwen4exp", Qwen4ExpReference),
}


# GGUF `clip.projector_type` -> (family name, vision reference): a projector's image features.
PROJECTORS: dict[str, tuple[str, type[Glm5NextVisionReference]]] = {
    "glm5v": ("glm5next", Glm5NextVisionReference),
}


def open_projector(path: Path) -> Glm5NextVisionReference:
    """The vision reference of a projector GGUF (`mmproj-*.gguf`), by its projector type."""
    package = Package.local(path)
    kind = package.metadata["clip.projector_type"].value if "clip.projector_type" in package.metadata else None
    if kind not in PROJECTORS:
        raise ValueError(f"{path}: no vision reference for projector type {kind!r}")
    return PROJECTORS[kind][1](package)


def _entry(package: Package) -> tuple[str, type[Reference]]:
    if package.architecture not in REFERENCES:
        raise ValueError(f"no reference for architecture {package.architecture!r}")
    return REFERENCES[package.architecture]


def family_of(package: Package) -> str:
    """The family name of a GGUF's reference."""
    return _entry(package)[0]


def reference_of(package: Package) -> type[Reference]:
    """The reference class of a GGUF, by its architecture: what its identity is of (`digest.implementation`)."""
    return _entry(package)[1]


def open_reference(model: Path | Package, device: torch.device, cache: bool, layers: int | None = None,
                   variant: str | None = None) -> Reference:
    """The family reference of a GGUF (`Reference.open`)."""
    package = model if isinstance(model, Package) else Package.local(Path(model))
    return reference_of(package).open(package, device, cache, layers, variant)
