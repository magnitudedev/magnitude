"""HTTP adaptation; model execution remains in the private worker."""

from contextlib import aclosing, asynccontextmanager
from functools import partial
from typing import cast

from anyio import to_thread
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Lifespan

from magnitude_engine.worker.host import WorkerUnavailable

from .requests import ChatRequest
from .responses import ChatResponse, sse
from .session import ChatFinished, ChatService


def error_payload(message: str, kind: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": kind}}


def create_app(
    service: ChatService | None = None, *, lifespan: Lifespan[FastAPI] | None = None
) -> FastAPI:
    @asynccontextmanager
    async def service_lifetime(app):
        try:
            yield
        finally:
            if service is not None:
                await to_thread.run_sync(service.close)

    app = FastAPI(title="Magnitude inference", lifespan=lifespan or service_lifetime)
    app.state.service = service

    def ready() -> ChatService:
        current = cast(ChatService | None, app.state.service)
        if current is None or not current.host.available:
            raise WorkerUnavailable("model worker is unavailable")
        return current

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError):
        return JSONResponse(error_payload(str(error)), status_code=422)

    @app.exception_handler(WorkerUnavailable)
    async def unavailable(request: Request, error: WorkerUnavailable):
        return JSONResponse(error_payload(str(error), "server_error"), status_code=503)

    @app.get("/health")
    async def health():
        current = ready()
        return {
            "status": "ready",
            "model": current.model,
            **{key: value for key, value in current.host.properties.items() if key != "type"},
            "speculative_backend": current.host.properties["speculative_backend"] or "none",
        }

    @app.get("/v1/models")
    async def models():
        current = ready()
        return {
            "object": "list",
            "data": [
                {"id": current.model, "object": "model", "created": 0, "owned_by": "magnitude"}
            ],
        }

    @app.post("/v1/chat/completions")
    async def completions(body: ChatRequest, connection: Request):
        current = ready()
        if body.model != current.model:
            return JSONResponse(error_payload("requested model is not loaded"), status_code=404)
        try:
            prompt = await to_thread.run_sync(partial(current.prepare, body))
        except (ValueError, TypeError, KeyError) as error:
            return JSONResponse(error_payload(str(error)), status_code=400)
        response = ChatResponse(body.model, current.host.properties["speculative_backend"])

        async def stream():
            try:
                yield sse(response.chunk({"role": "assistant"}))
                async with aclosing(
                    current.events(body, prompt, connection.is_disconnected)
                ) as events:
                    async for event in events:
                        if isinstance(event, ChatFinished):
                            yield sse(response.chunk({}, event.reason))
                            if body.stream_options.include_usage:
                                yield sse(response.terminal(event))
                            yield sse("[DONE]")
                        else:
                            yield sse(response.semantic(event, retain=False))
            except (
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                OverflowError,
                TimeoutError,
            ) as error:
                yield sse(error_payload(str(error), "server_error"))
                yield sse("[DONE]")

        if body.stream:
            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            async with aclosing(current.events(body, prompt, connection.is_disconnected)) as events:
                async for event in events:
                    if isinstance(event, ChatFinished):
                        return JSONResponse(response.complete(event))
                    response.semantic(event, retain=True)
        except (ValueError, TypeError, KeyError) as error:
            return JSONResponse(error_payload(str(error)), status_code=400)
        except OverflowError as error:
            return JSONResponse(error_payload(str(error), "server_error"), status_code=429)
        except WorkerUnavailable:
            raise
        except (RuntimeError, TimeoutError) as error:
            return JSONResponse(error_payload(str(error), "server_error"), status_code=500)
        return JSONResponse(error_payload("client disconnected", "server_error"), status_code=499)

    return app
