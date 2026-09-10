"""HTTP routes adapt validated chat work; the worker owns all model execution."""

import asyncio
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import TemplateError

from magnitude_engine.platform.worker import WorkerUnavailable
from magnitude_engine.serving.requests import ChatRequest
from magnitude_engine.serving.responses import ChatResponse, sse
from magnitude_engine.serving.runtime import Config
from magnitude_engine.serving.session import ChatFinished, ChatService


def error_payload(message: str, kind="invalid_request_error") -> dict:
    return dict(error=dict(message=message, type=kind))


def create_app(config: Config) -> FastAPI:
    @asynccontextmanager
    async def lifetime(app):
        service = await ChatService.open(config)
        app.state.service = service
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="Magnitude inference", lifespan=lifetime)

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request, error):
        return JSONResponse(error_payload(str(error)), status_code=422)

    @app.exception_handler(WorkerUnavailable)
    async def unavailable(request, error):
        return JSONResponse(error_payload(str(error), "server_error"), status_code=503)

    def ready() -> ChatService:
        return app.state.service

    @app.get("/health")
    async def health():
        service = ready()
        await asyncio.wrap_future(service.worker.call(lambda owner: None))
        return service.properties.model_dump(mode="json")

    @app.get("/v1/models")
    async def models():
        return dict(
            object="list",
            data=[dict(id=ready().model, object="model", created=0, owned_by="magnitude")],
        )

    @app.post("/v1/chat/completions")
    async def completions(body: ChatRequest):
        service = ready()
        if body.model != service.model:
            return JSONResponse(error_payload("requested model is not loaded"), status_code=404)
        try:
            prompt = await asyncio.to_thread(service.prepare, body)
        except (ValueError, TypeError, TemplateError) as error:
            return JSONResponse(error_payload(str(error)), status_code=400)
        response = ChatResponse(body.model)

        async def stream():
            try:
                yield sse(response.chunk({"role": "assistant"}))
                async with aclosing(service.events(body, prompt)) as events:
                    async for event in events:
                        if isinstance(event, ChatFinished):
                            yield sse(response.chunk({}, event.reason))
                            if body.stream_options.include_usage:
                                yield sse(response.terminal(event))
                            yield sse("[DONE]")
                        else:
                            yield sse(response.semantic(event, retain=False))
            except (ValueError, TypeError, RuntimeError) as error:
                yield sse(error_payload(str(error), "server_error"))
                yield sse("[DONE]")

        if body.stream:
            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            async with aclosing(service.events(body, prompt)) as events:
                async for event in events:
                    if isinstance(event, ChatFinished):
                        return JSONResponse(response.complete(event))
                    response.semantic(event, retain=True)
        except (ValueError, TypeError) as error:
            return JSONResponse(error_payload(str(error)), status_code=400)
        except RuntimeError as error:
            return JSONResponse(error_payload(str(error), "server_error"), status_code=500)
        return JSONResponse(error_payload("generation ended without completion"), status_code=500)

    return app
