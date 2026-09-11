"""Smallest device program whose compiled pipeline reports execution limits."""

import tilelang.language as T


def capability_probe():
    @T.prim_func
    def capabilities(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding(0)
            B[i] = A[i]

    return capabilities
