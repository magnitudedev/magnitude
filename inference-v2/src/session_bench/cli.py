"""Direct CLI selection; no experiment modules or saved-result execution format."""

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from benchmark_fixtures import bfcl as corpus
from benchmark_fixtures import prose as prose_source
from benchmark_fixtures.prose_history import Prose
from benchmark_fixtures.ruler import RulerFixture

from . import models
from .policy import (
    DEFAULT_CONTEXTS,
    ENGINES,
    MAX_OUTPUT_TOKENS,
    PROSE_OUTPUT_TOKENS,
    RETRIEVAL_OUTPUT_TOKENS,
    project_root,
)
from .results import inspect_run, public_command
from .suites import SECTIONS


def choices(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(value.split(",")))
    if selected == ("all",):
        return allowed
    if not selected or any(item not in allowed for item in selected):
        raise ValueError(f"choose from {', '.join(allowed)}")
    return selected


def contexts(value: str, *, preserve_order: bool = False) -> tuple[int, ...]:
    result = []
    for part in value.lower().split(","):
        number = int(part[:-1]) * 1024 if part.endswith("k") else int(part)
        if number <= 0:
            raise ValueError("context targets must be positive")
        result.append(number)
    return tuple(result) if preserve_order else tuple(sorted(set(result)))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="session-bench", description="Simulated sessions for inference serving"
    )
    commands = root.add_subparsers(dest="command", required=True)
    execute = commands.add_parser(
        "run", help="run selected sessions and record results automatically"
    )
    execute.add_argument("--model", action="append", default=[])
    execute.add_argument("--engine", action="append", choices=ENGINES, default=[])
    execute.add_argument("--target", action="append", default=[], metavar="ENGINE=ARTIFACT")
    execute.add_argument("--suite", default="context", help=", ".join(SECTIONS))
    execute.add_argument(
        "--context",
        default=",".join(map(str, DEFAULT_CONTEXTS)),
        help="input checkpoints, e.g. 4k,16k",
    )
    workload = execute.add_mutually_exclusive_group()
    workload.add_argument("--prose", action="store_true", help="continue Moby Dick; no tools")
    workload.add_argument("--retrieval", action="store_true", help="score RULER-derived retrieval")
    execute.add_argument("--retrieval-variant", choices=("single", "multiquery"))
    execute.add_argument("--retrieval-seed", type=int)
    execute.add_argument("--retrieval-queries", type=int)
    execute.add_argument(
        "--needle-depth", type=float, help="fraction of distractor records before facts"
    )
    execute.add_argument("--category", default=None, help=", ".join(corpus.CATEGORIES))
    execute.add_argument(
        "--case", help="BFCL decision ID; canonical background history is retained"
    )
    execute.add_argument("--repeat", type=int, default=1, help="repeat the whole balanced schedule")
    execute.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect corpus and schedule selection without loading models",
    )
    execute.add_argument("--json", action="store_true")
    for name in ("models", "engines", "suites", "runs"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--json", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true")
    return root


async def dry_run(
    root,
    targets,
    sections,
    checkpoints,
    categories,
    repeat,
    case,
    prose=False,
    retrieval: RulerFixture | None = None,
    needle_depth: float = 0.5,
):
    if retrieval is not None:
        corpus_digest = retrieval.identity
    elif prose:
        text, provenance = await prose_source.prepare()
        corpus_digest = Prose(text, provenance).identity
    else:
        fixtures, corpus_digest = await corpus.prepare(categories)
        if case is not None and not any(f.id == case for f in fixtures):
            raise ValueError(f"unknown or excluded BFCL case: {case}")
    return {
        "command": public_command(
            targets, sections, checkpoints, categories, repeat, case, prose, retrieval, needle_depth
        ),
        "targets": [target.model_dump() for target in targets],
        "corpus_digest": corpus_digest,
        "sections": sections,
        "contexts": checkpoints,
        "max_output_tokens": (
            RETRIEVAL_OUTPUT_TOKENS
            if retrieval
            else PROSE_OUTPUT_TOKENS
            if prose
            else MAX_OUTPUT_TOKENS
        ),
        "workload": "retrieval" if retrieval else "prose" if prose else "tools",
        "retrieval": retrieval.model_dump(mode="json") if retrieval else None,
        "needle_depth": needle_depth if retrieval else None,
        "preparation": "pending first-target tokenizer binding",
        "cache_policy": "disabled",
    }


async def managed_run(*args):
    from .runner import run

    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    # SIGINT is handled by asyncio.run; SIGTERM must take the same cleanup path.
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await run(*args)
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = project_root()
    try:
        if args.command == "run":
            selected = models.select(root, args.model, args.engine, args.target)
            sections = choices(args.suite, tuple(SECTIONS))
            checkpoints = contexts(args.context, preserve_order=args.retrieval)
            if (args.prose or args.retrieval) and (
                args.category is not None or args.case is not None
            ):
                raise ValueError("--prose/--retrieval cannot be combined with --category or --case")
            retrieval_options = (
                args.retrieval_variant,
                args.retrieval_seed,
                args.retrieval_queries,
                args.needle_depth,
            )
            if not args.retrieval and any(value is not None for value in retrieval_options):
                raise ValueError("retrieval options require --retrieval")
            retrieval = None
            needle_depth = 0.5 if args.needle_depth is None else args.needle_depth
            if not 0 <= needle_depth <= 1:
                raise ValueError("--needle-depth must be between zero and one")
            if args.retrieval:
                variant = args.retrieval_variant or "single"
                retrieval = RulerFixture(
                    seed=42 if args.retrieval_seed is None else args.retrieval_seed,
                    variant=variant,
                    queries=(
                        args.retrieval_queries
                        if args.retrieval_queries is not None
                        else 4
                        if variant == "multiquery"
                        else 1
                    ),
                )
            categories = (
                ()
                if args.prose or args.retrieval
                else choices(args.category or "all", tuple(corpus.CATEGORIES))
            )
            if args.repeat < 1:
                raise ValueError("--repeat must be positive")
            if args.dry_run:
                value = asyncio.run(
                    dry_run(
                        root,
                        selected,
                        sections,
                        checkpoints,
                        categories,
                        args.repeat,
                        args.case,
                        args.prose,
                        retrieval,
                        needle_depth,
                    )
                )
            else:
                value = asyncio.run(
                    managed_run(
                        root,
                        selected,
                        sections,
                        checkpoints,
                        categories,
                        args.repeat,
                        args.case,
                        lambda message: print(message, file=sys.stderr, flush=True),
                        args.prose,
                        retrieval,
                        needle_depth,
                    )
                )
            print(
                json.dumps(value, indent=2)
                if args.json or args.dry_run
                else f"{value['status']}: {value['path']}/report.md\n{value.get('error') or ''}"
            )
            return (
                0
                if args.dry_run or value["status"] == "completed"
                else (130 if value["status"] == "cancelled" else 1)
            )
        if args.command == "models":
            value = {
                key: alias.model_dump(exclude_none=True)
                for key, alias in models.aliases(root).items()
            }
            if not value and not args.json:
                print(f"No model aliases. Add MLX/GGUF references to {root / 'models.local.json'}")
                return 0
        elif args.command == "engines":
            value = {
                "magnitude": "Magnitude Python engine",
                "mlx-vlm": "Stock MLX-VLM serving",
                "omlx": "Pinned oMLX with native timing instrumentation",
                "llama.cpp": "Upstream llama-server on PATH (GGUF)",
            }
        elif args.command == "suites":
            value = {
                name: description + " Retained prefixes disabled."
                for name, description in SECTIONS.items()
            }
        elif args.command == "runs":
            directory = root / "runs" / "session-bench"
            value = [
                inspect_run(path)
                for path in sorted(directory.glob("*"), reverse=True)
                if (path / "run.json").is_file()
            ]
        else:
            if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
                raise ValueError("show requires a run ID, not a filesystem path")
            path = root / "runs" / "session-bench" / args.run_id
            value = inspect_run(path)
            if not args.json and (path / "report.md").is_file():
                print((path / "report.md").read_text())
                return 0
        if args.json or not isinstance(value, dict):
            print(json.dumps(value, indent=2))
        else:
            for key, item in value.items():
                print(f"{key}: {item}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"session-bench: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
