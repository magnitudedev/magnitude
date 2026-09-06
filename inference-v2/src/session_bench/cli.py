"""Direct CLI selection; no experiment modules or saved-result execution format."""

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from . import corpus, models
from .policy import DEFAULT_CONTEXTS, ENGINES, MAX_OUTPUT_TOKENS, project_root
from .results import inspect_run, public_command
from .runner import run
from .suites import SECTIONS, compile_plan


def choices(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(value.split(",")))
    if selected == ("all",):
        return allowed
    if not selected or any(item not in allowed for item in selected):
        raise ValueError(f"choose from {', '.join(allowed)}")
    return selected


def contexts(value: str) -> tuple[int, ...]:
    result = []
    for part in value.lower().split(","):
        number = int(part[:-1]) * 1024 if part.endswith("k") else int(part)
        if number <= 0:
            raise ValueError("context targets must be positive")
        result.append(number)
    return tuple(sorted(set(result)))


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
    execute.add_argument("--category", default="all", help=", ".join(corpus.CATEGORIES))
    execute.add_argument(
        "--case", help="BFCL decision ID; canonical background history is retained"
    )
    execute.add_argument("--repeat", type=int, default=1, help="repeat the whole balanced schedule")
    execute.add_argument(
        "--dry-run", action="store_true", help="build sessions without loading models"
    )
    execute.add_argument("--json", action="store_true")
    for name in ("models", "engines", "suites", "runs"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--json", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true")
    return root


async def dry_run(root, targets, sections, checkpoints, categories, repeat, case):
    fixtures, corpus_digest = await corpus.prepare(root, categories)
    plan = compile_plan(fixtures, corpus_digest, sections, checkpoints, case)
    return {
        "command": public_command(targets, sections, checkpoints, categories, repeat, case),
        "targets": [target.model_dump() for target in targets],
        "plan_digest": plan.identity,
        "requests_per_target_pass": len(plan.requests),
        "contexts": checkpoints,
        "parallel_sequences": plan.parallel_sequences,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "capacity": "pending artifact and tokenizer qualification",
        "cache_policy": plan.cache_policy,
    }


async def managed_run(*args):
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
            checkpoints = contexts(args.context)
            categories = choices(args.category, tuple(corpus.CATEGORIES))
            if args.repeat < 1:
                raise ValueError("--repeat must be positive")
            if args.dry_run:
                value = asyncio.run(
                    dry_run(
                        root, selected, sections, checkpoints, categories, args.repeat, args.case
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
