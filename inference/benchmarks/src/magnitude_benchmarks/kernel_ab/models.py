"""The regression model set: catalog models chosen so every kernel shape the catalog uses appears.

Revisions come from the catalog lock, so the set follows the shipped catalog. A model too large
for a host runs as a layer slice (`validation/layer_slice.py`): real weights, fewer layers, the
same attention and expert shapes.
"""

import json
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchModel:
    name: str
    catalog_id: str
    repository: str
    path: str
    covers: str
    draft: bool = False
    slice_layers: str | None = None

    @property
    def sliced(self) -> bool:
        return self.slice_layers is not None


MODELS = (
    BenchModel(
        "minicpm5-1b",
        "minicpm5-1b",
        "openbmb/MiniCPM5-1B-GGUF",
        "MiniCPM5-1B-Q4_K_M.gguf",
        "dense llama, G=8, head width 128",
    ),
    BenchModel(
        "qwen3.5-4b",
        "qwen3.5-4b",
        "unsloth/Qwen3.5-4B-MTP-GGUF",
        "Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "hybrid recurrent (gated delta), G=4, head width 256",
    ),
    BenchModel(
        "gemma-4-e2b",
        "gemma-4-e2b-it-qat",
        "unsloth/gemma-4-E2B-it-qat-GGUF",
        "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        "sliding window, G=8 with one KV head, head widths 256 and 512",
    ),
    BenchModel(
        "gemma-4-12b",
        "gemma-4-12b-it-qat",
        "unsloth/gemma-4-12B-it-qat-GGUF",
        "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        "G=16 global layers (head width 512), G=2 sliding layers (256)",
    ),
    BenchModel(
        "lfm2.5-8b-a1b",
        "lfm2.5-8b-a1b",
        "LiquidAI/LFM2.5-8B-A1B-GGUF",
        "LFM2.5-8B-A1B-Q4_K_M.gguf",
        "small mixture of experts (32 experts, top 4), short convolutions, G=4",
    ),
    BenchModel(
        "qwen3.6-35b-a3b",
        "qwen3.6-35b-a3b",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "mixture of experts (256 experts, top 8), G=8",
    ),
    BenchModel(
        "qwen3.8-27b-slice",
        "qwen3.8-27b",
        "unsloth/Qwen3.8-27B-GGUF",
        "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "first 4 layers of Qwen3.8 27B: G=6, attention projection at width 5120",
        slice_layers="0,1,2,3",
    ),
    BenchModel(
        "nemotron-3-super-slice",
        "nemotron-3-super-120b-a12b",
        "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
        "UD-Q4_K_XL/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-00001-of-00003.gguf",
        "one Mamba, one mixture-of-experts (512 experts, top 22) and one attention layer"
        " (G=16) of Nemotron 3 Super",
        slice_layers="0,1,7",
    ),
    BenchModel(
        "muse-glimmer-30b",
        "muse-glimmer-30b",
        "unsloth/Muse-Glimmer-30B-GGUF",
        "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        "dense, G=16 on every layer, head width 128",
    ),
)


def catalog_root(workspace: Path) -> Path:
    return workspace / "catalog"


def revision(workspace: Path, model: BenchModel) -> str:
    lock = json.loads((catalog_root(workspace) / "models.lock.json").read_text())
    return lock[model.catalog_id]["speculativeDraft" if model.draft else "target"]


def local_path(models_dir: Path, model: BenchModel) -> Path:
    return models_dir / model.name / Path(model.path).name


def select(names: str | None) -> tuple[BenchModel, ...]:
    if not names or names == "all":
        return MODELS
    known = {model.name: model for model in MODELS}
    selected = tuple(dict.fromkeys(names.split(",")))
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {', '.join(known)}")
    return tuple(known[name] for name in selected)


def fetch(workspace: Path, models_dir: Path, model: BenchModel, log) -> Path:
    """Download (or slice) the model at its locked revision; an existing file is reused."""
    target = local_path(models_dir, model)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    rev = revision(workspace, model)
    if model.sliced:
        log(f"slicing {model.name} layers {model.slice_layers}")
        subprocess.run(
            [
                "uv",
                "run",
                str(workspace / "validation" / "layer_slice.py"),
                "--repository",
                model.repository,
                "--revision",
                rev,
                "--path",
                model.path,
                "--output",
                str(target),
                "--layers",
                model.slice_layers,
            ],
            check=True,
        )
        return target
    url = f"https://huggingface.co/{model.repository}/resolve/{rev}/{model.path}"
    log(f"downloading {model.name} from {url}")
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file, 1 << 24)
    partial.replace(target)
    return target
