"""Validation of the references themselves against released code and checkpoints.

`check_hf` builds the transformers model of the same family from the reference's config, loads the
reference's own float32 weights into it (every parameter must be supplied, none left over) and
compares logits on seeded random prompts: it checks the forward pass against the released modeling
code. The prompts are arbitrary because both sides compute the same function of them; this is a
numerical equivalence check of two implementations, never a quality measurement. `deepseek41` is checked
the same way against DeepSeek's own inference code (`deepseek41_released.py`); a family transformers lacks
(`qwen4exp`, released in mlx_vlm) has no float64 run of its released math and is unjudgeable. `check_weights`
compares the reference's weights, after undoing the converter's transforms, with the original
checkpoint tensors: it checks the GGUF mapping (the differences are the file's quantization error; a
mapping error shows as a relative error near 1).

Each forward check is judged by `judge`: the reference and the released code both run in float64 (`Float64`), and
the released code also in float32, whose distance from its float64 run is the bar.

`compare_logits` compares the reference's logits with an independent producer's on the same tokens
(upstream llama.cpp's, `magnitude-references compare`), and `same_values` whether that producer's
file holds the values the reference computes with.

`check_hf` and `check_weights` need the `check` extra (transformers 5.17.0, safetensors).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from magnitude_references.model import F32
from magnitude_references.quality import Positions, compare_rows


def device_of(name: str) -> torch.device:
    """A torch device by name (`auto`: CUDA, else MPS, else CPU), with float32 matmuls in full
    precision (TF32 disabled)."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


NARROW = {torch.float32: torch.float64, torch.float16: torch.float64, torch.bfloat16: torch.float64,
          torch.complex64: torch.complex128}
NARROWING = {torch.Tensor.float, torch.Tensor.half, torch.Tensor.bfloat16}
IN_PLACE = {"__setitem__", "__iadd__", "__isub__", "__imul__", "__itruediv__", "__ipow__"}


class Float64(TorchFunctionMode):
    """Within it every floating-point value torch computes is float64 (complex128): every narrower dtype a call
    names is float64 (`.float()` is `.double()`), every narrower tensor a call reads is read as a float64 copy, and
    every narrower tensor it returns (a new tensor of the default dtype, its inputs being float64) is returned as
    one, so code written for float32 (the released code's explicit upcasts included) runs in float64. A call that
    would write a narrower tensor in place is an invariant violation: its copy's write would be lost. The mode is
    the calling thread's; the reference's pools carry it into their workers (`model.submit`)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in NARROWING:
            func = torch.Tensor.double
        name = getattr(func, "__name__", "")
        if name != "__get__":  # an attribute is read as stored
            in_place = name in IN_PLACE or name.endswith("_") and not name.startswith("__")
            target = kwargs.get("out", args[0] if in_place and args else None)
            if isinstance(target, torch.Tensor) and target.dtype in NARROW:
                raise AssertionError(f"{name} writes a {target.dtype} tensor in place in float64")
            args, kwargs = widened(args), {key: widened(value) for key, value in kwargs.items()}
        result = func(*args, **kwargs)
        return tuple(widened(value) for value in result) if type(result) is tuple else widened(result)


def widened(value):
    if isinstance(value, torch.dtype):
        return NARROW.get(value, value)
    if isinstance(value, torch.Tensor):
        return value.to(NARROW[value.dtype]) if value.dtype in NARROW else value
    if type(value) in (list, tuple):
        return type(value)(widened(item) for item in value)
    return value


def judge(values: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
          choices: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """The reference against the released code, by named output: (the reference in float64, the released code in
    float32, the released code in float64), and by named choice (argmax or proposal tokens): (the reference's,
    the released code's), both in float64; with `pass`.

    Both implementations run in float64, where each rounds 2^29 times finer than in float32, so they differ by
    their math alone: two implementations of the same function agree to float64's rounding, and any difference of
    the math larger than float32's rounding shows. The released code's own float32 run, against its float64 run,
    measures float32's rounding through this forward on these inputs, so it is the bar: the reference passes when
    its float64 output is no farther from the released code's float64 output than the released float32 output is
    (maximum difference relative to the released float64 output's largest magnitude), and when every choice
    agrees. No constant is involved. Precision a float32 order of operations loses, which a float64 comparison
    cannot see, is the storage-resolution KL's to judge (`kl.py`), downstream of this check."""
    report: dict = {"judgeable": True}
    passed = True
    for name, (ours, theirs, truth) in values.items():
        truth = truth.to("cpu", torch.float64)
        scale = float(truth.abs().max())
        ours_distance, bar = (float((value.to("cpu", torch.float64) - truth).abs().max()) / scale
                              for value in (ours, theirs))
        report[name] = {"relative_max_difference": ours_distance, "released_float32": bar}
        passed &= ours_distance <= bar
    for name, (ours, theirs) in choices.items():
        agree = ours.cpu() == theirs.cpu()
        report[name] = {"agreement": float(agree.double().mean())}
        passed &= bool(agree.all())
    return report | {"pass": bool(passed)}


def unjudgeable(reason: str) -> dict:
    """A check with no float64 truth to judge by: never a pass."""
    return {"judgeable": False, "reason": reason, "pass": False}


def without_float64(device: torch.device) -> dict | None:
    """The unjudgeable report of a check on `device` when it has no float64 (MPS), so neither side can run in it;
    None where it has."""
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
    except (TypeError, RuntimeError) as refusal:
        return unjudgeable(f"{device} has no float64 to run either implementation in: {refusal}")
    return None


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
    """The reference's logits against its family's released modeling code (`reference.released`): transformers, or
    DeepSeek's own; a family whose released code is mlx_vlm (transformers lacks it) is unjudgeable."""
    tokens = prompts(count, length, reference.vocab_size, seed).to(reference.weights.device)
    released = getattr(reference, "released", "transformers")
    refused = without_float64(reference.weights.device)
    if released == "mlx_vlm":  # no float64 run of its released math exists: mlx_vlm has none, transformers lacks it
        refused = unjudgeable(f"{reference.architecture}'s released code (mlx_vlm) has no float64 run")
    if refused is not None:
        return {"prompts": count, "length": length, "seed": seed, "layers": reference.block_count, **refused}

    def theirs(float64: bool) -> torch.Tensor:
        """The released code's logits, in float32 or (under `Float64`) float64. It takes its own copy of the
        weights, read uncached: the reference's decoded values would only double it."""
        with reference.weights.uncached():
            if released == "deepseek":
                from magnitude_references import deepseek41_released
                return deepseek41_released.logits(reference, tokens.cpu()).to(tokens.device)
            model = hf_model(reference)
            if float64:  # transformers' grouped expert GEMM has no float64 kernel; its eager loop is the same function
                model.set_experts_implementation("eager")
            return model(input_ids=tokens, use_cache=False).logits.to(F32)

    with torch.no_grad():
        released32 = theirs(False)
        with Float64():
            ours, truth = reference.forward(tokens), theirs(True)
    return {"prompts": count, "length": length, "seed": seed, "layers": reference.block_count,
            **judge({"logits": (ours, released32, truth)}, {"argmax": (ours.argmax(-1), truth.argmax(-1))})}


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


def logits_of(reference, tokens: torch.Tensor) -> np.ndarray:
    """The reference's float32 logits [prompts, length, vocabulary] of `tokens` [prompts, length]."""
    with torch.no_grad():
        return reference.forward(tokens.to(reference.weights.device)).to("cpu", F32).numpy()


def compare_logits(truth: np.ndarray, test: np.ndarray) -> tuple[Positions, float]:
    """KL(truth || test) and argmax agreement per position of two producers' logits [prompts, length, vocabulary]
    on the same tokens, each position named by its prompt and index, and the largest logit difference relative to
    the truth's largest."""
    if truth.shape != test.shape:
        raise ValueError(f"logits {test.shape} against {truth.shape}")
    prompts, length, vocabulary = truth.shape
    kl, agree = compare_rows(truth.reshape(-1, vocabulary), test.reshape(-1, vocabulary))
    rows = tuple({"prompt": prompt, "position": position} for prompt in range(prompts) for position in range(length))
    return Positions(rows, kl, agree), float(np.abs(truth - test).max() / np.abs(truth).max())


def same_values(weights, other) -> list[str]:
    """The tensors of `other` (another file's `Weights`) whose values, as the reference reads them, differ from
    `weights`' bit for bit: none when the two files store the same values. `weights`' file may hold more tensors,
    the companion scales it reads with their weights. A stacked expert tensor is compared an expert at a time."""
    names = other.package.names()
    if missing := sorted(set(names) - set(weights.package.names())):
        raise ValueError(f"tensors the reference's file lacks: {missing[:8]}")

    def equal(name: str) -> bool:
        shape = other.shape(name)
        if len(shape) != 3:
            return torch.equal(weights(name), other(name))
        return weights.shape(name) == shape and all(
            torch.equal(weights.expert(name, index), other.expert(name, index)) for index in range(shape[0]))
    return [name for name in names if not equal(name)]
