"""Execution API families, independent of a particular runtime adapter."""

from enum import StrEnum


class Backend(StrEnum):
    LLVM = "llvm"
    METAL = "metal"
    CUDA = "cuda"
    HIP = "hip"
