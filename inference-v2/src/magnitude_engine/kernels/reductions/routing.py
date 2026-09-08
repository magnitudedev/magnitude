"""Qwen routing epilogue: rounded softmax, stable top-k, normalization and shared gate."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

ROUTING = Program("magnitude_qwen_routes", Source("reductions/routing.metal"))


def select(logits: mx.array, top_k: int, normalize: bool) -> tuple[mx.array, mx.array, mx.array]:
    experts = logits.shape[-1] - 1
    if not 1 <= top_k <= min(16, experts) or not 1 <= experts <= 1024:
        raise ValueError("routing epilogue requires up to 1024 experts and 16 selected routes")
    threads = max(32, ((experts + 127) // 128) * 32)
    rows = logits.size // logits.shape[-1]
    indices, scores, shared = tuple(
        KernelPlan(
            program=ROUTING,
            inputs=(Input("logits", logits),),
            outputs=(
                Output("indices", (*logits.shape[:-1], top_k), mx.uint32),
                Output("scores", (*logits.shape[:-1], top_k), logits.dtype),
                Output("shared", (*logits.shape[:-1], 1), logits.dtype),
            ),
            launch=Launch((threads, rows, 1), (threads, 1, 1)),
            template=(
                ("T", logits.dtype),
                ("EXPERTS", experts),
                ("TOPK", top_k),
                ("GROUPS", threads // 32),
                ("NORMALIZE", normalize),
            ),
        ).run()
    )

    return indices, scores, shared
