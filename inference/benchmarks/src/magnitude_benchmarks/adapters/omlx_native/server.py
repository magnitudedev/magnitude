from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path
from typing import Any

from .config import AdapterConfiguration, write_configuration
from .instrumentation import install_instrumentation


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Magnitude's managed oMLX benchmark adapter")
    value.add_argument("--model", type=Path, required=True)
    value.add_argument("--served-model", required=True)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, required=True)
    value.add_argument("--base-path", type=Path, required=True)
    value.add_argument("--max-concurrent-requests", type=int, required=True)
    value.add_argument("--context-capacity", type=int, required=True)
    return value


def configuration(args: argparse.Namespace) -> AdapterConfiguration:
    return AdapterConfiguration(
        model=args.model.resolve(),
        served_model=args.served_model,
        base_path=args.base_path.resolve(),
        context_capacity=args.context_capacity,
        max_concurrent_requests=args.max_concurrent_requests,
    )


def _engine_parallel_capacity(engine: Any) -> int | None:
    scheduler_config = getattr(engine, "_scheduler_config", None)
    if scheduler_config is not None:
        value = getattr(scheduler_config, "max_num_seqs", None)
        if value is not None:
            return int(value)
    scheduler = getattr(
        getattr(getattr(engine, "_engine", None), "engine", None), "scheduler", None
    )
    value = getattr(getattr(scheduler, "config", None), "max_num_seqs", None)
    return int(value) if value is not None else None


def main() -> None:
    args = parser().parse_args()
    config = configuration(args)
    write_configuration(config)
    # Upstream publishes its SSH interpreter even with distributed inference off.
    # Keep that startup artifact inside this disposable benchmark installation.
    from omlx.cluster import worker_shim

    worker_shim.ensure_cluster_python_shim = partial(
        worker_shim.ensure_cluster_python_shim, home=config.base_path
    )
    install_instrumentation()

    import omlx.server as omlx_server
    import uvicorn
    from fastapi import HTTPException
    from omlx.settings import burst_decode_env, init_settings

    settings = init_settings(base_path=str(config.base_path))
    settings.model.model_dirs = [str((config.base_path / "models").resolve())]
    settings.model.model_fallback = False
    settings.model.hide_helper_models = True
    settings.scheduler.max_concurrent_requests = config.max_concurrent_requests
    settings.scheduler.chunked_prefill = False
    settings.cache.enabled = False
    settings.memory.prefill_memory_guard = False
    settings.huggingface.hf_cache_enabled = False
    settings.sampling.max_context_window = config.context_capacity
    settings.server.host = args.host
    settings.server.port = args.port
    settings.server.burst_decode_mode = "off"
    settings.server.preserve_mid_system_cache = False
    settings.server.distributed_inference_enabled = False
    for key, value in burst_decode_env("off").items():
        os.environ[key] = value
    settings.auth.api_key = None
    settings.auth.secret_key = "magnitude-private-benchmark-adapter"
    settings.ensure_directories()

    omlx_server.init_server(
        model_dirs=[str((config.base_path / "models").resolve())],
        scheduler_config=settings.to_scheduler_config(),
        api_key=None,
        global_settings=settings,
    )

    @omlx_server.app.get("/magnitude/benchmark/readiness")
    async def benchmark_readiness() -> dict[str, Any]:
        state = omlx_server._server_state
        pool = state.engine_pool
        if pool is None or not state.pinned_preload_complete:
            raise HTTPException(status_code=503, detail="model preload is incomplete")
        model_ids = pool.get_model_ids()
        if model_ids != [config.served_model]:
            raise HTTPException(
                status_code=500, detail=f"unexpected discovered models: {model_ids}"
            )
        entry = pool.get_entry(config.served_model)
        engine = getattr(entry, "engine", None) if entry is not None else None
        if engine is None:
            raise HTTPException(status_code=503, detail="model engine is not loaded")
        native_context = getattr(entry, "model_context_length", None)
        if native_context is None or int(native_context) < config.context_capacity:
            raise HTTPException(
                status_code=500,
                detail=f"model native context {native_context} is below {config.context_capacity}",
            )
        effective_context = omlx_server.get_max_context_window(config.served_model)
        if effective_context != config.context_capacity:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"effective context is {effective_context}, expected {config.context_capacity}"
                ),
            )
        actual_capacity = _engine_parallel_capacity(engine)
        if actual_capacity != config.max_concurrent_requests:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"engine concurrency is {actual_capacity}, "
                    f"expected {config.max_concurrent_requests}"
                ),
            )
        from omlx.patches.mlx_lm_mtp.batch_generator import _model_mtp_decode_enabled

        if type(engine).__name__ == "DFlashEngine" or _model_mtp_decode_enabled(
            getattr(engine, "_model", None)
        ):
            raise HTTPException(status_code=500, detail="speculation enabled in plain baseline")
        return {
            "ready": True,
            "discovered_model": config.served_model,
            "served_model": config.served_model,
            "loaded": True,
            "context_capacity": effective_context,
            "max_concurrent_requests": actual_capacity,
            "speculative_backend": "none",
        }

    @omlx_server.app.post("/session-bench/count")
    async def count_prompt(body: dict) -> dict:
        # Use the loaded engine's native chat renderer, without generating output.
        from omlx.engine.batched import BatchedEngine
        from omlx.engine.vlm import VLMBatchedEngine

        pool = omlx_server._server_state.engine_pool
        entry = None if pool is None else pool.get_entry(config.served_model)
        engine = None if entry is None else entry.engine
        if not isinstance(engine, (BatchedEngine, VLMBatchedEngine)):
            raise HTTPException(status_code=503, detail="text generation engine is not loaded")
        request = omlx_server.ChatCompletionRequest(**body)
        tools = omlx_server.convert_tools_for_template(request.tools)
        if tools and "gemma" in config.served_model.lower():
            tools = omlx_server.enrich_tool_params_for_gemma4(tools)
        kwargs = {"enable_thinking": False}
        is_vlm = isinstance(engine, omlx_server.VLMBatchedEngine)
        if getattr(engine, "message_extractor", None) is not None:
            raise HTTPException(status_code=500, detail="custom message extractor is unsupported")
        extractor = (
            omlx_server.extract_multimodal_content if is_vlm else omlx_server.extract_text_content
        )
        messages = extractor(
            request.messages, None, engine.tokenizer, consolidate_system_messages=False
        )
        messages = omlx_server.prepare_system_messages_for_template(
            messages,
            engine.tokenizer,
            tools=tools,
            chat_template_kwargs=kwargs,
            is_partial=False,
            merge_consecutive_roles=not is_vlm,
            unsupported_mid_system_policy=omlx_server._unsupported_mid_system_policy(),
        )
        return {
            "prompt_tokens": engine.count_chat_tokens(
                messages, tools, chat_template_kwargs=kwargs, is_partial=False
            )
        }

    uvicorn.run(omlx_server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
