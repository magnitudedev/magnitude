"""Portable generated Metal source, independently realized by each device owner."""

from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from magnitude_engine.data import Record
from magnitude_engine.platform.compiler import compiler_implementation
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.kernel_cache import ArtifactCache, ArtifactKey, encoded, fingerprint
from magnitude_engine.platform.measurement import clock_ns

if TYPE_CHECKING:
    from tilelang.jit import PrimFunc


class MetalLowering(Record):
    target: Literal["metal"] = "metal"
    unroll_max_step: int = 64
    unroll_max_extent: int = 4
    explicit_unroll: bool = True
    vectorize: bool = False
    legalize_safe_memory: bool = False


class MetalSource(Record):
    format: Literal["metal-source-v1"] = "metal-source-v1"
    source: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    signature: tuple[TensorSpec, ...]
    permutation: tuple[int, ...]
    groups: tuple[int, int, int]
    threads: tuple[int, int, int]

    @model_validator(mode="after")
    def launch_contract(self):
        if not self.signature or len(set(self.permutation)) != len(self.permutation):
            raise ValueError("invalid shader operand contract")
        if any(index < 0 or index >= len(self.signature) for index in self.permutation):
            raise ValueError("shader operand index outside signature")
        if any(extent <= 0 for extent in (*self.groups, *self.threads)):
            raise ValueError("shader launch extents must be positive")
        return self


class LoweringStatistics(Record):
    generated: int
    lowering_ns: int
    native_realizations: int
    native_realization_ns: int


@cache
def lowering_identity(policy: bytes) -> str:
    digest = bytearray(policy)
    for name in ("metal_compiler.py", "compiler.py", "kernel_cache.py"):
        digest.extend(name.encode() + b"\x00" + Path(__file__).with_name(name).read_bytes())
    return fingerprint(bytes(digest))


class MetalCompiler:
    def __init__(self, root: Path | None = None):
        self.policy = MetalLowering()
        self.artifacts = ArtifactCache(MetalSource, root)
        self.compiler_identity = fingerprint(encoded(compiler_implementation()))
        self.generated = self.lowering_ns = 0
        self.native_realizations = self.native_realization_ns = 0

    @property
    def statistics(self) -> LoweringStatistics:
        return LoweringStatistics(
            generated=self.generated,
            lowering_ns=self.lowering_ns,
            native_realizations=self.native_realizations,
            native_realization_ns=self.native_realization_ns,
        )

    def realized(self, duration_ns: int) -> None:
        self.native_realizations += 1
        self.native_realization_ns += duration_ns

    def compile(self, program: "PrimFunc") -> MetalSource:
        from tilelang import tvm
        from tilelang.engine import lower

        signature = tuple(
            TensorSpec(
                tuple(int(n) for n in program.buffer_map[p].shape),
                DType(str(program.buffer_map[p].dtype)),
            )
            for p in program.params
        )
        key = ArtifactKey(
            program=fingerprint(tvm.ir.save_json(program).encode()),
            compiler=self.compiler_identity,
            lowering=lowering_identity(encoded(self.policy)),
        )
        artifact = self.artifacts.read(key)
        if artifact is not None:
            if artifact.signature != signature:
                raise ValueError("cached Metal source differs from its kernel input contract")
            return artifact
        original = tuple(program.buffer_map[p].name for p in program.params)
        policy = self.policy
        target = tvm.target.Target(policy.target)
        started = clock_ns()
        with (
            target,
            tvm.transform.PassContext(
                config={
                    "tirx.disable_vectorize": not policy.vectorize,
                    "tl.disable_safe_memory_legalize": not policy.legalize_safe_memory,
                    "tl.UnrollLoop": {
                        "auto_max_step": policy.unroll_max_step,
                        "auto_max_extent": policy.unroll_max_extent,
                        "explicit_unroll": policy.explicit_unroll,
                    },
                }
            ),
        ):
            lowered = lower(program, target=target)
        self.generated += 1
        self.lowering_ns += clock_ns() - started
        functions = tuple(lowered.device_mod.functions.items())
        if len(functions) != 1:
            raise ValueError("a compiled Metal operation must have one entry point")
        symbol, function = functions[0]
        permutation = tuple(original.index(str(param.name)) for param in function.params)
        extents = {str(tag): int(value) for tag, value in function.attrs["thread_extent"].items()}

        def dimensions(prefix: str) -> tuple[int, int, int]:
            return (
                extents.get(prefix + ".x", 1),
                extents.get(prefix + ".y", 1),
                extents.get(prefix + ".z", 1),
            )

        artifact = MetalSource(
            source=lowered.kernel_source,
            entry=symbol.name_hint,
            signature=signature,
            permutation=permutation,
            groups=dimensions("blockIdx"),
            threads=dimensions("threadIdx"),
        )
        self.artifacts.write(key, artifact)
        return artifact
