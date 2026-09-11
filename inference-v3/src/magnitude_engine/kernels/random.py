"""Counter-addressed Philox4x32-10, using portable 32-bit integer arithmetic.

Algorithm and known-answer vectors: DEShawResearch/random123. No mutable RNG
state is shared between requests, physical rows, invocations, or tuning trials.
"""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial


@T.macro
def mulhilo(a, b):
    mask = T.uint32(65535)
    p0 = (a & mask) * (b & mask)
    p1 = (a >> 16) * (b & mask)
    p2 = (a & mask) * (b >> 16)
    p3 = (a >> 16) * (b >> 16)
    middle = (p0 >> 16) + (p1 & mask) + (p2 & mask)
    high = p3 + (p1 >> 16) + (p2 >> 16) + (middle >> 16)
    low = (middle << 16) | (p0 & mask)
    return high, low


@T.macro
def philox(c0, c1, c2, c3, k0, k1):
    counter = T.alloc_local((4,), "uint32")
    key = T.alloc_local((2,), "uint32")
    next_counter = T.alloc_local((4,), "uint32")
    counter[0], counter[1], counter[2], counter[3] = c0, c1, c2, c3
    key[0], key[1] = k0, k1
    for _ in T.serial(10):
        hi0, lo0 = mulhilo(T.uint32(0xD2511F53), counter[0])
        hi1, lo1 = mulhilo(T.uint32(0xCD9E8D57), counter[2])
        next_counter[0] = hi1 ^ counter[1] ^ key[0]
        next_counter[1] = lo1
        next_counter[2] = hi0 ^ counter[3] ^ key[1]
        next_counter[3] = lo0
        for j in T.unroll(4):
            counter[j] = next_counter[j]
        key[0] += T.uint32(0x9E3779B9)
        key[1] += T.uint32(0xBB67AE85)
    return counter[0], counter[1], counter[2], counter[3]


def words(rows: int, *, capability: Capability, threads: int = 128):
    """Explicit counter/key transform, also independently qualified by KATs."""
    if min(rows, threads) <= 0:
        raise ValueError("random transform requires positive geometry")
    cpu = serial(capability)

    @T.macro
    def apply(Addresses, Output, row):
        a, b, c, d = philox(
            Addresses[row, 0],
            Addresses[row, 1],
            Addresses[row, 2],
            Addresses[row, 3],
            Addresses[row, 4],
            Addresses[row, 5],
        )
        Output[row, 0], Output[row, 1], Output[row, 2], Output[row, 3] = a, b, c, d

    @T.prim_func
    def main(Addresses: T.Tensor((rows, 6), "uint32"), Output: T.Tensor((rows, 4), "uint32")):
        if cpu:
            for row in T.Parallel(rows):
                apply(Addresses, Output, row)
        else:
            with T.Kernel(T.ceildiv(rows, threads), threads=threads) as block:
                row = block * threads + T.get_thread_binding(0)
                if row < rows:
                    apply(Addresses, Output, row)

    return main
