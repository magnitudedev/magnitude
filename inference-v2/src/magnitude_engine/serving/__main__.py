"""Explicit local development serving composition."""

import argparse
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import anyio
import uvicorn
from anyio import to_thread

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.worker.host import Worker
from magnitude_engine.worker.launch import add_engine_arguments, engine_from_arguments

from .app import create_app
from .session import ChatService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_arguments(parser)
    parser.add_argument("--model", required=True, help="served model ID")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    engine = engine_from_arguments(args, parser)

    @asynccontextmanager
    async def lifespan(app):
        host = await to_thread.run_sync(partial(Worker.start, engine=engine))
        service = None
        try:
            artifact = await to_thread.run_sync(
                partial(TokenizerArtifact.load, Path(host.properties["target_path"]))
            )
            service = await to_thread.run_sync(partial(ChatService, host, artifact, args.model))
            app.state.service = service
            yield
        finally:
            with anyio.CancelScope(shield=True):
                if service is not None:
                    await to_thread.run_sync(service.close)
                await to_thread.run_sync(host.close)

    uvicorn.run(create_app(lifespan=lifespan), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
