"""Explicit local development serving composition."""

import argparse
from contextlib import asynccontextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio
import uvicorn
from anyio import to_thread

from magnitude_engine import blueprints as bp
from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.worker.host import Worker

from .app import create_app
from .session import ChatService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target", type=Path)
    source.add_argument("--engine-blueprint", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--model", required=True, help="served model ID")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-active", type=int)
    parser.add_argument("--prefill-tokens", type=int)
    parser.add_argument("--output-capacity", type=int)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--memory-gib", type=float)
    memory.add_argument("--memory-bytes", type=int)
    parser.add_argument("--retained-prefixes", type=int)
    parser.add_argument("--max-draft-tokens", type=int)
    args = parser.parse_args()
    if args.engine_blueprint is not None:
        overrides = (
            "head",
            "max_draft_tokens",
            "context_tokens",
            "max_active",
            "prefill_tokens",
            "output_capacity",
            "memory_gib",
            "memory_bytes",
            "retained_prefixes",
        )
        if any(getattr(args, name) is not None for name in overrides):
            parser.error("declare engine dependencies and limits inside --engine-blueprint")
        engine = bp.loads(args.engine_blueprint.read_text())
        if not isinstance(engine, bp.engine.Engine):
            raise ValueError("serving requires an engine blueprint")
    else:
        artifact = bp.model.artifacts.Local(path=str(args.target.expanduser().resolve()))
        target = bp.model.auto(artifact)
        method = bp.generation.methods.Plain()
        if args.head is not None:
            program = bp.model.programs.qwen35.Program(artifact=artifact)
            target = bp.model.Executor(program=program, state=bp.model.state.PagedHybrid())
            head = bp.model.programs.mtp.Head(
                artifact=bp.model.artifacts.Local(path=str(args.head.expanduser().resolve())),
                target=program,
            )
            method = bp.generation.methods.MTP(
                drafter=bp.model.Executor(program=head, state=bp.model.state.Native(source=head)),
                max_draft_tokens=args.max_draft_tokens,
            )
        elif args.max_draft_tokens is not None:
            raise ValueError("a draft allowance requires a drafter")
        memory_policy = bp.engine.memory.Budgeted()
        if args.memory_gib is not None:
            memory_policy = replace(memory_policy, limit_bytes=int(args.memory_gib * (1 << 30)))
        elif args.memory_bytes is not None:
            memory_policy = replace(memory_policy, limit_bytes=args.memory_bytes)
        engine = bp.engine.Engine(
            generation=bp.generation.Generation(target=target, method=method),
            scheduler=bp.engine.scheduling.TimeShared(
                **{
                    name: getattr(args, name)
                    for name in ("max_active", "prefill_tokens")
                    if getattr(args, name) is not None
                },
            ),
            memory=memory_policy,
            prefixes=bp.engine.prefixes.Radix(
                retention=bp.engine.prefixes.LeastRecentlyUsed(
                    **(
                        {"max_entries": args.retained_prefixes}
                        if args.retained_prefixes is not None
                        else {}
                    ),
                ),
            ),
            **{
                name: getattr(args, name)
                for name in ("context_tokens", "output_capacity")
                if getattr(args, name) is not None
            },
        )

    @asynccontextmanager
    async def lifespan(app):
        host = await to_thread.run_sync(partial(Worker.start, engine=engine))
        try:
            artifact = await to_thread.run_sync(
                partial(TokenizerArtifact.load, Path(host.properties["target_path"]))
            )
            app.state.service = ChatService(host, artifact, args.model)
            yield
        finally:
            with anyio.CancelScope(shield=True):
                await to_thread.run_sync(host.close)

    uvicorn.run(create_app(lifespan=lifespan), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
