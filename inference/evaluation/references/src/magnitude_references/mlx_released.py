"""Logits of an mlx_vlm language model over exported weights, for references whose released code
exists only in mlx_vlm (`checks.mlx_logits` runs it). It imports only mlx, mlx_vlm and numpy, so it
runs under an interpreter with mlx_vlm installed:

    python -I mlx_released.py EXPORT_DIR

`EXPORT_DIR` holds `config.json` ({"module", "text_config", "constants"}), `parameters.npz`
(float32, mlx parameter names) and `tokens.npy` (int64 [B, T]); `logits.npy` (float32 [B, T, V]) is
written beside them. Every model parameter must be supplied except the derived constants, which must
equal `constants` exactly.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


def main(directory: Path) -> None:
    config = json.loads((directory / "config.json").read_text())
    language = importlib.import_module(f"mlx_vlm.models.{config['module']}.language")
    settings = importlib.import_module(f"mlx_vlm.models.{config['module']}.config")
    model = language.LanguageModel(settings.TextConfig(**config["text_config"]))
    expected = dict(tree_flatten(model.parameters()))
    constants = config["constants"]
    for name, values in constants.items():
        derived = np.array(expected.pop(name)).tolist()
        if derived != values:
            raise SystemExit(f"{name}: mlx derives {derived}, the header stores {values}")
    supplied = dict(np.load(directory / "parameters.npz"))
    if set(supplied) != set(expected):
        raise SystemExit(f"mlx parameters not supplied: {sorted(set(expected) - set(supplied))}; "
                         f"unknown: {sorted(set(supplied) - set(expected))}")
    for name, value in supplied.items():
        if tuple(expected[name].shape) != value.shape:
            raise SystemExit(f"{name}: export {value.shape} vs mlx {tuple(expected[name].shape)}")
    model.update(tree_unflatten([(name, mx.array(value)) for name, value in supplied.items()]))
    tokens = mx.array(np.load(directory / "tokens.npy"))
    # One prefill from empty caches: the released attention reads its positions from a cache.
    logits = model.lm_head(model.model(tokens, cache=model.make_cache()))
    np.save(directory / "logits.npy", np.array(logits.astype(mx.float32)))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
