"""Command line shared by the family references.

uv run inference/validation/<family>_reference.py logits --model M.gguf --tokens 1,2,3
uv run inference/validation/<family>_reference.py check-hf --model M.gguf [--layers N] [--prompts 3 --length 48]
uv run inference/validation/<family>_reference.py check-weights --model M.gguf --checkpoint DIR_WITH_SAFETENSORS

Add `--device cuda` on a CUDA host. Matmuls run in full float32 (TF32 disabled).

`check-hf` builds the transformers model of the same family from the reference's config, loads the
reference's own float32 weights into it (every parameter must be supplied, none left over) and
compares logits on seeded random prompts: it checks the forward pass against the released modeling
code. `check-weights` compares the reference's weights, after undoing the converter's transforms,
with the original checkpoint tensors: it checks the GGUF mapping (the differences are the file's
quantization error; a mapping error shows as a relative error near 1).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from reference_gguf import Package
from reference_model import F32, Weights


def device_of(name: str) -> torch.device:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(name)


def open_reference(cls, model: Path, device: torch.device, cache: bool, layers: int | None):
    package = Package.local(model)
    return cls(package, Weights(package, device, cache), layers)


def prompts(count: int, length: int, vocab: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    return torch.from_numpy(generator.integers(0, vocab, (count, length), dtype=np.int64))


def hf_model(reference):
    """transformers model of the reference's family carrying the reference's weights exactly."""
    from transformers import AutoConfig
    config = AutoConfig.for_model(**reference.hf_config())
    with torch.device(reference.weights.device):
        model = reference.hf_build(config)
    parameters = reference.hf_parameters()
    expected = dict(model.state_dict())
    unused = reference.hf_unused_prefixes()
    missing = sorted(name for name in set(expected) - set(parameters) - set(reference.hf_derived())
                     if not name.startswith(unused))
    extra = sorted(set(parameters) - set(expected))
    if missing or extra:
        raise ValueError(f"transformers parameters not supplied: {missing}; unknown: {extra}")
    for name, value in parameters.items():
        if tuple(expected[name].shape) != tuple(value.shape):
            raise ValueError(f"{name}: reference {tuple(value.shape)} vs transformers {tuple(expected[name].shape)}")
    model.load_state_dict(parameters, strict=False)
    return model.eval()


def check_hf(reference, count: int, length: int, seed: int) -> dict:
    tokens = prompts(count, length, reference.vocab_size, seed).to(reference.weights.device)
    with torch.no_grad():
        ours = reference.forward(tokens)
        theirs = hf_model(reference)(input_ids=tokens, use_cache=False).logits.to(F32)
    difference = (ours - theirs).abs()
    scale = float(theirs.abs().max())
    report = {
        "prompts": count, "length": length, "seed": seed, "layers": reference.block_count,
        "max_abs_difference": float(difference.max()), "max_abs_logit": scale,
        "relative_max_difference": float(difference.max()) / scale,
        "mean_abs_difference": float(difference.mean()),
        "argmax_agreement": float((ours.argmax(-1) == theirs.argmax(-1)).to(F32).mean()),
    }
    report["pass"] = report["relative_max_difference"] < 1e-4 and report["argmax_agreement"] == 1.0
    return report


def check_weights(reference, checkpoint: Path) -> dict:
    from safetensors import safe_open
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise ValueError(f"no safetensors under {checkpoint}")
    location = {}
    for file in files:
        with safe_open(file, "pt") as stream:
            for key in stream.keys():
                location[key] = file

    def read(key: str) -> torch.Tensor:
        with safe_open(location[key], "pt") as stream:
            return stream.get_tensor(key).to(F32)

    rows = []
    for name, value in reference.hf_parameters().items():
        try:
            expected = reference.hf_checkpoint(name, read)
        except KeyError as missing:
            rows.append({"parameter": name, "status": f"absent from checkpoint: {missing}"})
            continue
        value = value.to("cpu")
        if tuple(expected.shape) != tuple(value.shape):
            rows.append({"parameter": name, "status": f"shape {tuple(value.shape)} vs {tuple(expected.shape)}"})
            continue
        relative = float((value - expected).norm() / expected.norm().clamp_min(1e-30))
        rows.append({"parameter": name, "relative_error": relative})
    worst = max((row.get("relative_error", math.inf) for row in rows), default=math.inf)
    return {"tensors": len(rows), "worst_relative_error": worst, "rows": rows}


def main(cls) -> None:
    parser = argparse.ArgumentParser(description=cls.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("logits", "check-hf", "check-weights"):
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--layers", type=int, help="run only the first N layers (then the final norm and head)")
        command.add_argument("--device", default="cpu")
        command.add_argument("--json", type=Path)
    commands.choices["logits"].add_argument("--tokens", required=True, help="comma-separated token ids")
    commands.choices["logits"].add_argument("--output", type=Path, help=".npy of float32 logits [T, V]")
    commands.choices["check-hf"].add_argument("--prompts", type=int, default=3)
    commands.choices["check-hf"].add_argument("--length", type=int, default=48)
    commands.choices["check-hf"].add_argument("--seed", type=int, default=20260927)
    commands.choices["check-weights"].add_argument("--checkpoint", type=Path, required=True)
    options = parser.parse_args()
    device = device_of(options.device)
    reference = open_reference(cls, options.model, device, options.command != "logits", options.layers)
    with torch.no_grad():
        if options.command == "logits":
            tokens = torch.tensor([[int(t) for t in options.tokens.split(",")]], device=device)
            logits = reference.forward(tokens)[0]
            if options.output:
                np.save(options.output, logits.cpu().numpy())
            report = {"tokens": tokens[0].tolist(), "argmax": logits.argmax(-1).tolist()}
        elif options.command == "check-hf":
            report = check_hf(reference, options.prompts, options.length, options.seed)
        else:
            report = check_weights(reference, options.checkpoint)
    report = {"architecture": reference.architecture, "model": str(options.model), **report}
    text = json.dumps(report, indent=1)
    print(text if options.command != "check-weights" else json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    if options.json:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(text + "\n")
