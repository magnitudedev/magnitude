"""Digests as the evaluation contracts define them (lowercase hex SHA-256; a file's is
`magnitude_rig.host.identity.sha256`)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def canonical(value) -> str:
    """SHA-256 of the canonical JSON encoding: sorted keys, no insignificant whitespace, UTF-8."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The CPU BLASes on which a reference row's bits are a function of its own prefix alone, whatever rows share its
# products (`model.ROW_TILE`): Accelerate (the Mac) and nvpl (GB10), whose GEMM rounds a row by the row and the
# call's shape. x86 MKL also rounds a GEMM row by its place in the call, and torch's vectorised silu and softplus an
# element by the tensor's length there, so its rows only stay within the storage-resolution bar of each other.
ROWS_EXACT = ("accelerate", "nvpl")


def platform(device: str) -> dict:
    """What a reference's bits depend on besides its sources (the reference fixes its own thread teams,
    `model.THREADS`): the operating system and machine, torch and the BLAS it calls, and the device (`type`, a
    CUDA device by name), which differ by platform; a reference records it (`run`) and its identity includes it.
    `rows_exact`: whether a row's bits are its prefix's alone on it (`ROWS_EXACT`), which judging a build's rows
    against the reference relies on (`cache.obtain`)."""
    import platform as host
    import re

    import torch
    blas = re.search(r"BLAS_INFO=([^,]+)", torch.__config__.show())
    assert blas is not None, "torch's build settings name no BLAS"
    return {"system": host.system(), "machine": host.machine(), "torch": torch.__version__, "blas": blas[1],
            "device": torch.cuda.get_device_name() if device == "cuda" else device,
            "rows_exact": device == "cpu" and blas[1] in ROWS_EXACT}


# The modules that run every reference, beside its family's (`run.build_reference`).
RUNS = ("model", "gguf", "perturbation", "contract", "sanity", "run")
# Package modules reachable from them whose sources decide no row or sanity value: the registry, which imports every
# family to choose one by architecture and computes none (`families`: the one chosen is the family the identity is
# of, and `Reference.open` opens it), this identity and the platform it records (`digest`), result records
# (`record`), the device's name and the validation of references against released code (`checks`), how a reference
# is evaluated (`performance`) and where its time goes (`profile`).
OUTSIDE = frozenset(f"{PACKAGE.name}.{name}"
                    for name in ("families", "digest", "record", "checks", "performance", "profile"))


def _path(module: str) -> Path | None:
    """The source of `module` within the package, or None where it names none of its modules (an attribute)."""
    parts = module.split(".")
    if parts[0] != PACKAGE.name:
        return None
    base = PACKAGE.joinpath(*parts[1:])
    return next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.is_file()), None)


def _imported(tree: ast.Module) -> set[str]:
    """The package modules `tree` imports anywhere in it (a function's own imports too)."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module is not None, f"{ast.unparse(node)}: the package imports absolutely"
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        else:
            continue
        found.update(name for name in names if _path(name) is not None)
    return found


def _computing(tree: ast.Module) -> str:
    """`tree` without its docstrings (comments are no part of a syntax tree), dumped without source positions."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(first := node.body[0], ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.dump(tree)


def closure(family: str) -> dict[str, ast.Module]:
    """The package modules a reference of `family` (its module) runs, each parsed, by name: the static import closure
    of it and `RUNS`, less `OUTSIDE`."""
    pending, trees = [family, *(f"{PACKAGE.name}.{name}" for name in RUNS)], {}
    while pending:
        module = pending.pop()
        if module in trees or module in OUTSIDE:
            continue
        path = _path(module)
        assert path is not None, f"{module} is no module of {PACKAGE.name}"
        trees[module] = tree = ast.parse(path.read_text(), str(path))
        pending.extend(_imported(tree))
    return trees


def _sources(family: str) -> str:
    """The digest of what computes in the sources a reference of `family` runs (`closure`): each module's syntax
    tree, without its comments and docstrings."""
    return canonical({module: _computing(tree) for module, tree in closure(family).items()})


def implementation(family: type, variant: str | None, device: str, on: dict | None = None) -> str:
    """The identity of a reference of `family` (its class, `families.reference_of`): the canonical digest of the
    sources it runs and `variant` and the `platform` it runs on (`device`, a device type; `on`, a recorded platform,
    to find what the same sources are on another), so no family, variant or platform shares an identity with
    another."""
    return canonical({"implementation": _sources(family.__module__), "variant": variant,
                      "platform": platform(device) if on is None else on})
