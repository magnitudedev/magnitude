"""An invocation binds named operands and outputs to a numerical Metal program."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files

import mlx.core as mx

type Parameter = int | bool | mx.Dtype
type Extent = tuple[int, int, int]


@cache
def identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"invalid Metal identifier: {name!r}")


@dataclass(frozen=True)
class Source:
    """Packaged source with explicit, ordered dependencies; paths are kernel-relative."""

    path: str
    dependencies: tuple[Source, ...] = ()
    text: str = field(init=False, repr=False)

    def __post_init__(self):
        if not self.path.endswith(".metal") or any(
            part in ("", ".", "..") for part in self.path.split("/")
        ):
            raise ValueError("Metal sources require a relative package path")
        object.__setattr__(
            self, "text", files("magnitude_engine.kernels").joinpath(self.path).read_text()
        )


@dataclass(frozen=True)
class Program:
    name: str
    body: Source

    def __post_init__(self):
        identifier(self.name)


@dataclass(frozen=True)
class Scalar:
    """A finite compile-time scalar, emitted without editing numerical source."""

    name: str
    value: int | float | bool = field(compare=False)
    literal: str = field(init=False, repr=False)

    def __post_init__(self):
        identifier(self.name)
        if not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError("kernel constants must be finite scalars")
        # The emitted representation distinguishes integer/float arithmetic and signed zero.
        literal = (
            str(int(self.value)) if isinstance(self.value, (bool, int)) else f"{self.value!r}f"
        )
        object.__setattr__(self, "literal", literal)


@dataclass(frozen=True)
class Input:
    name: str
    value: mx.array


@dataclass(frozen=True)
class Output:
    name: str
    shape: tuple[int, ...]
    dtype: mx.Dtype


@dataclass(frozen=True)
class Launch:
    grid: Extent
    threadgroup: Extent

    def __post_init__(self):
        for extent in (self.grid, self.threadgroup):
            if len(extent) != 3 or any(type(n) is not int or n <= 0 for n in extent):
                raise ValueError("kernel launch requires three positive integer extents")
        if math.prod(self.threadgroup) > 1024:
            raise ValueError("threadgroup exceeds Metal's 1024-thread limit")


@dataclass(frozen=True)
class KernelPlan:
    program: Program
    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    launch: Launch
    template: tuple[tuple[str, Parameter], ...] = ()
    constants: tuple[Scalar, ...] = ()

    def __post_init__(self):
        names = [x.name for x in (*self.inputs, *self.outputs, *self.constants)]
        names.extend(name for name, _ in self.template)
        for name in names:
            identifier(name)
        if len(names) != len(set(names)) or not self.outputs:
            raise ValueError("kernel bindings must be unique and include an output")
        if any(any(type(n) is not int or n < 0 for n in x.shape) for x in self.outputs):
            raise ValueError("kernel output shapes must be nonnegative integer extents")

    def run(self) -> tuple[mx.array, ...]:
        from .runtime import execute

        return execute(self)
