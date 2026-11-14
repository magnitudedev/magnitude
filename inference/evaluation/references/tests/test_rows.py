"""Every row operation the reference's forward calls keeps a row's bits its own, where the platform promises it
(`digest.ROWS_EXACT`): at sizes past one parallel chunk, a prefix's rows equal the whole call's rows.

The operations are read from the code (`called`): every torch function, torch.nn.functional function and tensor
method the families and model.py call, kept when it computes rows of a float tensor in its default form (an
elementwise op, or one along the last axis). torch cuts a parallel op into chunks by the tensor's length, and its
vectorised body and scalar tail round some ops differently, so such an op called on a whole tensor rounds a row by
how many rows the tensor has. An op that does is called only through `model.pointwise`, which calls it in
fixed-length row tiles."""
from __future__ import annotations

import ast
from functools import cache

import pytest
import torch

from conftest import ROWS_EXACT
from magnitude_references import digest, model

ROWS, WIDTH = 1200, 2048  # 2.4M elements: past one parallel chunk at any team size
PREFIXES = sorted({*range(1, ROWS, 11), 78, 92, 99})
TEAMS = (model.THREADS, 4, 1)


def sources() -> list:
    """The forward's sources: model.py and the families (the rest of what `digest.closure` reaches is NumPy and records)."""
    return [model.__file__, *sorted(str(path) for path in (digest.PACKAGE / "families").glob("*.py"))]


def called() -> dict[str, list[str]]:
    """Each torch callable the forward calls by name (`torch.x`, `F.x`, `torch.nn.functional.x`, a tensor's `.x`),
    with the literal keyword arguments of the call (`F.gelu(approximate='tanh')`), and the places it is called."""
    found: dict[str, list[str]] = {}
    for path in sources():
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            owner, name = ast.unparse(node.func.value), node.func.attr
            key = {"torch": f"torch.{name}", "F": f"F.{name}", "torch.nn.functional": f"F.{name}"}.get(owner,
                                                                                                    f"Tensor.{name}")
            literal = [k for k in node.keywords if k.arg is not None and isinstance(k.value, ast.Constant)]
            if literal:
                key += "(" + ", ".join(f"{k.arg}={k.value.value!r}" for k in literal) + ")"
            found.setdefault(key, []).append(f"{path.rsplit('/', 1)[-1]}:{node.lineno}")
    return found


def resolve(key: str):
    """The callable `key` names, its literal keyword arguments bound."""
    name, _, arguments = key.partition("(")
    owner, name = name.split(".", 1)
    function = getattr({"torch": torch, "F": torch.nn.functional, "Tensor": torch.Tensor}[owner], name, None)
    if function is None or not arguments:
        return function
    keywords = eval(f"dict({arguments.removesuffix(')')})")
    return lambda *args: function(*args, **keywords)


@cache
def row_op(key: str):
    """`key` as a function of one float tensor [1, T, W] that returns rows [1, T, ...], or None when it is not one
    (it takes other arguments, returns no tensor of rows, or writes in place)."""
    function = resolve(key)
    if function is None or key.partition("(")[0].endswith("_"):
        return None
    sample = torch.randn(1, 5, 16, generator=torch.Generator().manual_seed(0))
    for form in (lambda x: function(x), lambda x: function(x, dim=-1), lambda x: function(x, 2.0)):
        try:
            out = form(sample.clone())
        except Exception:
            continue
        if isinstance(out, torch.Tensor) and out.is_floating_point() and tuple(out.shape[:2]) == (1, 5):
            return form
    return None


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit for bit (an op's NaNs, of a negative's square root, included)."""
    bits = {torch.float32: torch.int32, torch.float64: torch.int64}
    return a.shape == b.shape and torch.equal(a.contiguous().view(bits[a.dtype]), b.contiguous().view(bits[b.dtype]))


def moved(function, threads: int) -> list[int]:
    """The prefix lengths whose rows differ from the same rows of the whole call, at a team of `threads`."""
    x = torch.randn(1, ROWS, WIDTH, generator=torch.Generator().manual_seed(1))
    torch.set_num_threads(threads)
    try:
        whole = function(x)
        return [p for p in PREFIXES if not same(function(x[:, :p].contiguous()), whole[:, :p])]
    finally:
        torch.set_num_threads(model.THREADS)


def row_ops() -> dict[str, list[str]]:
    return {key: places for key, places in called().items() if row_op(key) is not None}


def test_the_forward_calls_row_ops() -> None:
    """The scan finds the forward's row ops (a scan that found none would check nothing)."""
    assert {"torch.exp", "torch.rsqrt", "torch.softmax", "Tensor.sum"} <= set(row_ops())


@pytest.mark.skipif(not ROWS_EXACT, reason="this platform's BLAS rounds a row by its place in the call: no op is "
                                           "made row-exact here (`model.tiled`)")
def test_every_row_op_the_forward_calls_directly_keeps_a_rows_bits() -> None:
    """A row op the forward calls directly keeps a row's bits whatever the tensor's length and team; one that does
    not is called through `model.pointwise` only."""
    failing = {}
    for key, places in row_ops().items():
        lengths = {threads: moved(row_op(key), threads) for threads in TEAMS}
        if any(lengths.values()):
            failing[key] = (places, {threads: p[:5] for threads, p in lengths.items() if p})
    assert not failing, f"row ops that round a row by the tensor's length, called directly: {failing}"


@pytest.mark.skipif(not ROWS_EXACT, reason="this platform's BLAS rounds a row by its place in the call: no op is "
                                           "made row-exact here (`model.tiled`)")
@pytest.mark.parametrize("function", [torch.sigmoid, torch.nn.functional.silu, torch.nn.functional.softplus,
                                      lambda x: torch.nn.functional.gelu(x, approximate="tanh")],
                         ids=["sigmoid", "silu", "softplus", "gelu_tanh"])
def test_pointwise_keeps_a_rows_bits(function) -> None:
    """`model.pointwise` makes a row's bits its own for the ops that round by length on some platform."""
    for threads in TEAMS:
        assert not moved(lambda x: model.pointwise(function, x), threads), threads
