"""An invocation binds named operands and outputs to a numerical Metal program."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from pathlib import Path

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
        if not Path(self.path).is_absolute() and (
            not self.path.endswith(".metal")
            or any(part in ("", ".", "..") for part in self.path.split("/"))
        ):
            raise ValueError("Metal sources require a relative package path")
        object.__setattr__(
            self,
            "text",
            (
                Path(self.path)
                if Path(self.path).is_absolute()
                else files("magnitude_engine.kernels").joinpath(self.path)
            ).read_text(),
        )


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
class Launch:
    grid: Extent
    threadgroup: Extent

    def __post_init__(self):
        for extent in (self.grid, self.threadgroup):
            if len(extent) != 3 or any(type(n) is not int or n <= 0 for n in extent):
                raise ValueError("kernel launch requires three positive integer extents")
        if math.prod(self.threadgroup) > 1024:
            raise ValueError("threadgroup exceeds Metal's 1024-thread limit")
