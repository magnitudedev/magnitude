"""Owned model serving entry point, including the verbatim session-bench flags."""

import argparse
from pathlib import Path

import uvicorn

from engine.blueprints.execution import ScheduleProfile
from engine.inputs.formats.templates import template_file
from engine.platform.backend import Backend
from engine.serving.app import create_app
from engine.serving.runtime import Config


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
    parser.add_argument("--forced-quantum", type=int, default=32)
    parser.add_argument("--retained-prefixes", type=int, choices=(0,), default=0)
    parser.add_argument("--backend", type=Backend, choices=tuple(Backend))
    parser.add_argument("--ordinal", type=int, default=0)
    templates = parser.add_mutually_exclusive_group()
    templates.add_argument("--chat-template", type=Path, help="operator template source override")
    templates.add_argument("--chat-template-variant", help="artifact template variant override")
    parser.add_argument(
        "--schedule-profile",
        type=Path,
        help="JSON selection-store directory and validation identity; "
        "requires complete calibration",
    )
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
        forced_quantum=args.forced_quantum,
        retained_prefixes=args.retained_prefixes,
        backend=args.backend,
        ordinal=args.ordinal,
        template_variant=args.chat_template_variant,
        template_override=(
            template_file(args.chat_template, name="operator")
            if args.chat_template is not None
            else None
        ),
        schedules=(
            ScheduleProfile.model_validate_json(args.schedule_profile.read_text())
            if args.schedule_profile is not None
            else None
        ),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
