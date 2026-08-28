from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import AdapterConfiguration, detected_backend, write_configuration
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
    value.add_argument("--cache-policy", choices=["disabled"], required=True)
    value.add_argument("--memory-guard", choices=["off"], required=True)
    value.add_argument(
        "--speculative-method", choices=["none", "mtp", "dflash", "dspark"], required=True
    )
    value.add_argument("--max-draft-tokens", type=int)
    value.add_argument("--dflash-draft", type=Path)
    value.add_argument("--dflash-block-size", type=int)
    return value


def configuration(args: argparse.Namespace) -> AdapterConfiguration:
    return AdapterConfiguration(
        model=args.model.resolve(),
        served_model=args.served_model,
        base_path=args.base_path.resolve(),
        context_capacity=args.context_capacity,
        max_concurrent_requests=args.max_concurrent_requests,
        speculative_method=args.speculative_method,
        max_draft_tokens=args.max_draft_tokens,
        dflash_draft=args.dflash_draft.resolve() if args.dflash_draft else None,
        dflash_block_size=args.dflash_block_size,
    )


def _terminal(events: str) -> dict[str, Any]:
    terminal: dict[str, Any] | None = None
    saw_done = False
    tool_name = ""
    tool_arguments = ""
    for block in events.replace("\r\n", "\n").split("\n\n"):
        lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
        if not lines:
            continue
        data = "\n".join(lines)
        if data == "[DONE]":
            saw_done = True
            continue
        payload = json.loads(data)
        for choice in payload.get("choices", []):
            for call in choice.get("delta", {}).get("tool_calls", []):
                function = call.get("function", {})
                tool_name += str(function.get("name", ""))
                tool_arguments += str(function.get("arguments", ""))
        if payload.get("choices") == [] and "timings" in payload:
            terminal = payload
    if not saw_done or terminal is None:
        raise RuntimeError("qualification response did not contain terminal evidence and [DONE]")
    try:
        parsed_arguments = json.loads(tool_arguments)
    except json.JSONDecodeError as exc:
        raise RuntimeError("qualification tool arguments are invalid JSON") from exc
    if tool_name != "benchmark_ready" or parsed_arguments != {"ready": True}:
        raise RuntimeError("qualification response did not parse as benchmark_ready(ready=true)")
    return terminal


async def _qualify_running_server(host: str, port: int, served_model: str) -> dict[str, Any]:
    import httpx

    tool = {
        "type": "function",
        "function": {
            "name": "benchmark_ready",
            "description": "Confirm readiness",
            "parameters": {
                "type": "object",
                "properties": {"ready": {"type": "boolean"}},
                "required": ["ready"],
            },
        },
    }
    body = {
        "model": served_model,
        "messages": [
            {
                "role": "user",
                "content": "Call benchmark_ready with ready=true. Do not answer with text.",
            }
        ],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "benchmark_ready"}},
        "temperature": 0,
        "max_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(f"http://{host}:{port}/v1/chat/completions", json=body)
        response.raise_for_status()
    terminal = _terminal(response.text)
    if terminal["usage"]["completion_tokens"] <= 0:
        raise RuntimeError("qualification request generated no tokens")
    return terminal


def _selected_backend(engine: Any, expected: str) -> str:
    if expected == "none":
        return "none"
    if expected == "dflash":
        return "dflash" if type(engine).__name__ == "DFlashEngine" else "none"
    try:
        from omlx.patches.mlx_lm_mtp.batch_generator import (
            _model_has_mtp_module,
            _model_mtp_decode_enabled,
        )

        model = getattr(engine, "_model", None)
        if (
            model is None
            or not _model_has_mtp_module(model)
            or not _model_mtp_decode_enabled(model)
        ):
            return "none"
    except Exception:
        return "none"
    if expected == "dspark":
        try:
            from omlx.utils.model_loading import _has_dspark_heads

            model_config = json.loads((Path(engine._model_name) / "config.json").read_text())
            if not _has_dspark_heads(model_config):
                return "none"
        except Exception:
            return "none"
    return expected


def _engine_parallel_capacity(engine: Any) -> int | None:
    scheduler_config = getattr(engine, "_scheduler_config", None)
    if scheduler_config is not None:
        value = getattr(scheduler_config, "max_num_seqs", None)
        if value is not None:
            return int(value)
    scheduler = getattr(getattr(getattr(engine, "_engine", None), "engine", None), "scheduler", None)
    value = getattr(getattr(scheduler, "config", None), "max_num_seqs", None)
    return int(value) if value is not None else None


def main() -> None:
    args = parser().parse_args()
    config = configuration(args)
    write_configuration(config)
    backend = detected_backend(config)
    install_instrumentation(backend)

    from fastapi import HTTPException
    from omlx.settings import burst_decode_env, init_settings
    import omlx.server as omlx_server
    import uvicorn

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

    qualification: dict[str, Any] | None = None
    qualification_lock = asyncio.Lock()

    @omlx_server.app.get("/magnitude/benchmark/readiness")
    async def benchmark_readiness() -> dict[str, Any]:
        nonlocal qualification
        state = omlx_server._server_state
        pool = state.engine_pool
        if pool is None or not state.pinned_preload_complete:
            raise HTTPException(status_code=503, detail="model preload is incomplete")
        model_ids = pool.get_model_ids()
        if model_ids != [config.served_model]:
            raise HTTPException(status_code=503, detail=f"unexpected discovered models: {model_ids}")
        entry = pool.get_entry(config.served_model)
        engine = getattr(entry, "engine", None) if entry is not None else None
        if engine is None:
            raise HTTPException(status_code=503, detail="model engine is not loaded")
        native_context = getattr(entry, "model_context_length", None)
        if native_context is None or int(native_context) < config.context_capacity:
            raise HTTPException(
                status_code=503,
                detail=f"model native context {native_context} is below {config.context_capacity}",
            )
        effective_context = omlx_server.get_max_context_window(config.served_model)
        if effective_context != config.context_capacity:
            raise HTTPException(
                status_code=503,
                detail=f"effective context is {effective_context}, expected {config.context_capacity}",
            )
        actual_capacity = _engine_parallel_capacity(engine)
        if actual_capacity != config.max_concurrent_requests:
            raise HTTPException(
                status_code=503,
                detail=f"engine concurrency is {actual_capacity}, expected {config.max_concurrent_requests}",
            )
        selected = _selected_backend(engine, backend)
        if selected != backend:
            raise HTTPException(
                status_code=503, detail=f"expected backend {backend}, selected {selected}"
            )
        async with qualification_lock:
            if qualification is None:
                try:
                    terminal = await _qualify_running_server(args.host, args.port, config.served_model)
                    timings = terminal["timings"]
                    if backend != "none":
                        if timings.get("speculative_backend") != backend:
                            raise RuntimeError("qualification backend evidence does not match")
                        if int(timings.get("draft_n", 0)) <= 0:
                            raise RuntimeError("qualification did not demonstrate drafting")
                    qualification = {"terminal": terminal}
                    (config.base_path / "qualification.json").write_text(
                        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
                    )
                except Exception as exc:
                    logging.getLogger(__name__).exception("qualification request failed")
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "ready": True,
            "discovered_model": config.served_model,
            "served_model": config.served_model,
            "loaded": True,
            "context_capacity": effective_context,
            "max_concurrent_requests": actual_capacity,
            "speculative_backend": selected,
            "qualification_completed": True,
        }

    uvicorn.run(omlx_server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
