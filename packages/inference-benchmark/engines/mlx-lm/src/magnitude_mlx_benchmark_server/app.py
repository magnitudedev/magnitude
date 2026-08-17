from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import threading
import time
import uuid

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from . import __version__
from .generation import GeneratedEvent, MlxGenerationWorker, NativeEvidence
from .protocol import (
    RequestValidationError,
    completion_chunk,
    parse_chat_request,
    sse,
    terminal_chunk,
)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_app(
    *,
    model_path: str,
    served_model: str,
    model_revision: str,
    artifact_manifest: str,
    prefill_step_size: int = 2048,
    prompt_cache_entries: int = 32,
) -> Starlette:
    manifest_digest = file_sha256(artifact_manifest)
    worker = MlxGenerationWorker(
        model_path,
        prefill_step_size=prefill_step_size,
        prompt_cache_entries=prompt_cache_entries,
    )

    async def health(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "adapter": f"magnitude-mlx-benchmark-server/{__version__}",
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "model": served_model,
                "model_revision": model_revision,
                "artifact_manifest_sha256": manifest_digest,
            }
        )

    async def models(_: Request) -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": served_model, "object": "model", "owned_by": "magnitude"}
                ],
            }
        )

    async def chat(request: Request):
        try:
            body = await request.json()
            parsed = parse_chat_request(body, served_model)
        except (json.JSONDecodeError, RequestValidationError) as error:
            return JSONResponse(
                {"error": {"message": str(error), "type": "invalid_request_error"}},
                status_code=400,
            )

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[GeneratedEvent | Exception | None] = asyncio.Queue()
        cancelled = threading.Event()

        worker.submit(
            parsed,
            cancelled,
            lambda event: loop.call_soon_threadsafe(queue.put_nowait, event),
        )

        async def stream():
            evidence: NativeEvidence | None = None
            finish_reason = "stop"
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        yield sse(
                            {
                                "id": request_id,
                                "object": "error",
                                "model": served_model,
                                "choices": [],
                                "error": {"message": str(item)},
                            }
                        )
                        return
                    if item.kind == "content":
                        yield sse(
                            completion_chunk(
                                request_id,
                                served_model,
                                created,
                                content=str(item.value),
                            )
                        )
                    elif item.kind == "tool_calls":
                        yield sse(
                            completion_chunk(
                                request_id, served_model, created, tool_calls=item.value
                            )
                        )
                    elif item.kind == "finish":
                        finish_reason = str(item.value)
                    elif item.kind == "evidence":
                        evidence = item.value
                if evidence is None:
                    return
                yield sse(
                    completion_chunk(
                        request_id, served_model, created, finish_reason=finish_reason
                    )
                )
                yield sse(
                    terminal_chunk(
                        request_id,
                        served_model,
                        created,
                        prompt_tokens=evidence.prompt_tokens,
                        cached_tokens=evidence.cached_tokens,
                        completion_tokens=evidence.completion_tokens,
                        prompt_ms=evidence.prompt_ms,
                        generation_ms=evidence.generation_ms,
                        peak_memory_bytes=evidence.peak_memory_bytes,
                    )
                )
                yield sse("[DONE]")
            finally:
                cancelled.set()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
        ],
        on_shutdown=[worker.close],
    )
