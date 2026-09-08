"""The single MLX construction boundary; plans derive signatures and launch bindings."""

from functools import cache
from typing import Any

import mlx.core as mx

from .assembly import assemble
from .plan import KernelPlan, Program, Scalar


@cache
def compile_kernel(
    program: Program,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    constants: tuple[Scalar, ...],
) -> Any:
    source = assemble(program, constants)
    return mx.fast.metal_kernel(
        name=program.name,
        input_names=list(inputs),
        output_names=list(outputs),
        source=source.body,
        header=source.header,
    )


def execute(plan: KernelPlan) -> tuple[mx.array, ...]:
    kernel = compile_kernel(
        plan.program,
        tuple(x.name for x in plan.inputs),
        tuple(x.name for x in plan.outputs),
        plan.constants,
    )
    return tuple(
        kernel(
            inputs=[x.value for x in plan.inputs],
            template=list(plan.template),
            grid=plan.launch.grid,
            threadgroup=plan.launch.threadgroup,
            output_shapes=[x.shape for x in plan.outputs],
            output_dtypes=[x.dtype for x in plan.outputs],
        )
    )
