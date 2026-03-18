"""Serve: start the vllm-mlx OpenAI-compatible inference server."""

import os
import sys


def main():
    model = os.environ.get("INFER_MODEL", "mlx-community/Qwen3.5-27B-6bit")
    host = os.environ.get("INFER_HOST", "127.0.0.1")
    port = os.environ.get("INFER_PORT", "8012")

    # NOTE: --continuous-batching and --use-paged-cache are disabled because
    # Qwen3.5's hybrid architecture (ArraysCache for linear-attention layers)
    # is incompatible with vllm-mlx's paged cache and batched engine.
    # Tracking issues:
    #   https://github.com/waybarrios/vllm-mlx/issues/159  (continuous batching incompatible with ArraysCache)
    #   https://github.com/waybarrios/vllm-mlx/issues/145  (ArraysCache 'trim' AttributeError)
    #   https://github.com/waybarrios/vllm-mlx/issues/142  (prefix cache tensor slicing fails for Qwen3.5)
    #   https://github.com/waybarrios/vllm-mlx/issues/136  (prefix cache fails on MoE models)
    #   https://github.com/QwenLM/Qwen3.5/issues/37        (KV cache prefill failure in mlx-lm)
    args = [
        "vllm-mlx", "serve", model,
        "--host", host,
        "--port", port,
        #"--continuous-batching",
        #"--use-paged-cache",
        #"--reasoning-parser", "qwen3",
    ]

    extra = os.environ.get("INFER_EXTRA_ARGS", "")
    if extra:
        args.extend(extra.split())

    print(f"Starting vllm-mlx for {model}")
    print(f"Base URL: http://{host}:{port}/v1")
    print()

    os.execvp("vllm-mlx", args)


if __name__ == "__main__":
    main()