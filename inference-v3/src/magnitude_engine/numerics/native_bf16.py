"""Native BF16 operation boundaries, ported from the TileLang POC."""

import struct

import tilelang.language as T


@T.macro
def native_sigmoid(x, precise=True):
    exponential = (
        (T.exp(T.abs(x)) if precise else T.__exp(T.abs(x))).astype("bfloat16").astype("float32")
    )
    denominator = (1 + exponential).astype("bfloat16").astype("float32")
    reciprocal = (1 / denominator).astype("bfloat16").astype("float32")
    return T.if_then_else(x < 0, reciprocal, 1 - reciprocal).astype("bfloat16")


@T.macro
def native_decay(input, bias, log_rate):
    x = (input + bias).astype("bfloat16").astype("float32")
    exponential = T.exp(-T.abs(x)).astype("bfloat16").astype("float32")
    logged = (
        T.if_then_else(exponential < 1e-4, exponential, T.log(1 + exponential))
        .astype("bfloat16")
        .astype("float32")
    )
    softplus = (T.max(x, 0) + logged).astype("bfloat16").astype("float32")
    return T.exp(-T.exp(log_rate) * softplus)


def norm(H, D, gain=1.0, ROWS=1, STRIDE=None, OFFSET=0, epsilon=1e-6):
    STRIDE = H * D if STRIDE is None else STRIDE
    threads = min(D // 4, 1024)
    gain_bits = struct.unpack("I", struct.pack("f", gain))[0]

    @T.prim_func
    def main(
        A: T.Tensor((ROWS, STRIDE), "bfloat16"),
        B: T.Tensor((D,), "float32"),
        C: T.Tensor((ROWS, H, D), "bfloat16"),
    ):
        with T.Kernel(H, ROWS, threads=threads) as (h, row):
            lane = T.get_thread_binding()
            partial = T.alloc_shared((32,), "float32")
            squares = T.alloc_local((1,), "float32")
            squares[0] = 0
            for chunk in T.serial(T.ceildiv(D, threads * 4)):
                for j in T.serial(4):
                    d = chunk * threads * 4 + lane * 4 + j
                    if d < D:
                        x = A[row, OFFSET + h * D + d].astype("float32")
                        squares[0] += x * x
            if lane < 32:
                partial[lane] = 0
            T.sync_threads()
            subtotal = T.warp_reduce_sum(squares[0])
            if lane % 32 == 0:
                partial[lane // 32] = subtotal
            T.sync_threads()
            if lane < 32:
                total = T.warp_reduce_sum(partial[lane])
                if lane == 0:
                    partial[0] = total
            T.sync_threads()
            inverse = T.rsqrt(partial[0] / D + epsilon)
            for chunk in T.serial(T.ceildiv(D, threads * 4)):
                for j in T.serial(4):
                    d = chunk * threads * 4 + lane * 4 + j
                    if d < D:
                        scaled = (
                            (A[row, OFFSET + h * D + d].astype("float32") * inverse)
                            .astype("bfloat16")
                            .astype("float32")
                        )
                        weighted = (
                            (scaled * B[d].astype("float32")).astype("bfloat16").astype("float32")
                        )
                        C[row, h, d] = weighted * T.reinterpret(T.uint32(gain_bits), "float32")

    return main
