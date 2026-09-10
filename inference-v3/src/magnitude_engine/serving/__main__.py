"""Owned model serving entry point, including the verbatim session-bench flags."""

import argparse

import uvicorn

from magnitude_engine.platform.backend import Backend
from magnitude_engine.serving.app import create_app
from magnitude_engine.serving.runtime import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Magnitude TileLang server")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default="magnitude")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--memory-bytes", type=int, required=True)
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--max-queued", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--output-capacity", type=int, default=16)
    parser.add_argument("--retained-prefixes", type=int, choices=(0,), default=0)
    parser.add_argument("--backend", type=Backend, choices=tuple(Backend))
    parser.add_argument("--ordinal", type=int, default=0)
    args = parser.parse_args()
    config = Config(
        target=args.target,
        model=args.model,
        memory_bytes=args.memory_bytes,
        context_tokens=args.context_tokens,
        parallel_sequences=args.max_active,
        max_queued=args.max_queued,
        prefill_tokens=args.prefill_tokens,
        output_capacity=args.output_capacity,
        retained_prefixes=args.retained_prefixes,
        backend=args.backend,
        ordinal=args.ordinal,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
