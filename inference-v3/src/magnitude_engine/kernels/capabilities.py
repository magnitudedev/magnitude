"""The backend axis, reduced to the endpoint facts a kernel may consult.

A ``Capability`` is produced once by the driver when a context opens. It is the
only backend fact any package outside ``platform/`` may read: no operation, no
weight owner, no model tests a ``Backend`` value.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    subgroup_width: int
    """Lanes that share ``T.warp_reduce_*``. One on the host."""

    threads_per_group: int
    """Largest ``T.Kernel(threads=)``. One on the host, where a grid is a loop."""

    shared_memory_bytes: int
    """``T.alloc_shared`` budget per group. Zero on the host."""

    matrix_instructions: bool
    """``T.gemm`` lowers to matrix hardware."""

    def __post_init__(self):
        if self.subgroup_width <= 0 or self.threads_per_group <= 0:
            raise ValueError("capability widths must be positive")
        if self.shared_memory_bytes < 0:
            raise ValueError("capability shared memory must not be negative")
        if type(self.matrix_instructions) is not bool:
            raise TypeError("matrix instruction support must be a boolean")


HOST = Capability(
    subgroup_width=1, threads_per_group=1, shared_memory_bytes=0, matrix_instructions=False
)


def capability_probe():
    """Smallest device program whose compiled pipeline reports execution limits.

    The compiler is imported here, not at module scope: a ``Capability`` is
    plain data that blueprint inspection must be able to read without loading a
    compiler or a device runtime.
    """
    import tilelang.language as T

    @T.prim_func
    def capabilities(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding(0)
            B[i] = A[i]

    return capabilities
