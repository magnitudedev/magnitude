"""Upstream llama.cpp at the release the converter is pinned to: its source, the tools `llama/CMakeLists.txt` builds
against it (CPU only), and what they answer.

- `verify-publish`: whether a converted file alone does what its users need (`verify.py`);
- `logits`: upstream's logits of every position of a prompt, the independent producer the references are checked
  against (`magnitude-references compare`).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import tomllib
from pathlib import Path

import numpy as np

_PROJECT = Path(__file__).resolve().parents[2]  # inference/conversion


def _pin() -> tuple[str, str]:
    """The llama.cpp repository and commit pyproject.toml pins the converter to: the one owner of the pin."""
    source = tomllib.loads((_PROJECT / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["llama-cpp-scripts"]
    return source["git"], source["rev"]


def source() -> Path:
    """A checkout of the pinned llama.cpp commit, fetched once into the user cache."""
    url, rev = _pin()
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "magnitude-conversion" / f"llama.cpp-{rev}"
    if not (root / "CMakeLists.txt").exists():
        root.mkdir(parents=True, exist_ok=True)
        for command in (["init", "-q"], ["fetch", "-q", "--depth", "1", url, rev], ["checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(root), *command], check=True)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    if head.stdout.strip() != rev:
        raise RuntimeError(f"{root} is at {head.stdout.strip()}, not the pinned {rev}")
    return root


def tool(target: str) -> Path:
    """A target of `llama/CMakeLists.txt`, built CPU-only against the pinned source."""
    pinned = source()
    # CMake binds a build directory to the absolute path of the source it was configured from, and
    # `llama/` moves with each checkout: one build directory per source path.
    project = _PROJECT / "llama"
    build = pinned / f"build-tools-{hashlib.sha256(str(project).encode()).hexdigest()[:16]}"
    subprocess.run(["cmake", "-S", str(project), "-B", str(build), f"-DLLAMA_SOURCE={pinned}",
                    "-DCMAKE_BUILD_TYPE=Release",
                    # No host compiler cache: ggml adopts any ccache or sccache on PATH, which ties the build
                    # to that daemon's state (sccache's socket path overflows inside a rig run).
                    "-DGGML_CCACHE=OFF"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(build), "--target", target, "-j", str(os.cpu_count() or 1)],
                   check=True, stdout=subprocess.DEVNULL)
    return build / target


def logits(gguf: Path, tokens: np.ndarray) -> np.ndarray:
    """Upstream's float32 logits [prompts, length, vocabulary] of `tokens` [prompts, length]: each prompt decoded in
    one forward from an empty context (`llama/logits.cpp`)."""
    if tokens.ndim != 2 or tokens.shape[1] == 0:
        raise ValueError(f"tokens must be [prompts, length] with length > 0, not {tokens.shape}")
    program = tool("logits")
    rows = []
    with tempfile.TemporaryDirectory(prefix="magnitude-logits-") as temporary:
        out = Path(temporary) / "logits.f32"
        for prompt in tokens:
            subprocess.run([str(program), str(gguf), str(out), *map(str, prompt.tolist())], check=True)
            values = np.fromfile(out, np.float32)
            if values.size % len(prompt):
                raise RuntimeError(f"logits wrote {values.size} values for {len(prompt)} positions")
            rows.append(values.reshape(len(prompt), -1))
    return np.stack(rows)
