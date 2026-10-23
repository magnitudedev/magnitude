"""magnitude-bench: the serving benchmark, over HTTP, on a catalog model at context lengths.

    magnitude-bench job --mode tools --model qwen3.5-4b:gguf:q4 --context 4k,16k \\
        [--engine magnitude --engine llama.cpp ...] [--section context,parallel] \\
        [--tuning-seconds 60] [--fresh-tuning] --artifacts FILE --output DIR --build-identity FILE

`job` measures inside a run directory on the executing host; `mrig bench` takes the same
measurement options and runs it there (building our engine and the harness, acquiring the
model, holding the device and passing the thermal gate). Paths given (binaries, model stores,
profiles) are paths on that host. A binary given is used as is and not built:
`--magnitude-binary PATH --magnitude-build '{"commit": …, "dirty_digest": null}'
--no-tuning-trace` measures another build's engine (one predating the tuning options).
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import get_args

from .models import Acquisition, catalog_reference, targets
from .options import (
    EngineOptions,
    KvCacheType,
    LlamaOptions,
    MagnitudeOptions,
    OllamaOptions,
    Schedule,
    SourceBuild,
    Watchdog,
)
from .plan import SECTIONS
from .policy import ENGINES, MODES
from .store import RunStore

#: Engines whose prompts the harness renders host-only (GGUF): their runs need `--harness`.
HOST_RENDERED = ("magnitude", "llama.cpp", "ollama", "ollama-registry")


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
            raise ValueError("context lengths must be positive")
        result.append(number)
    return tuple(sorted(set(result)))


def measurement_options(parser: argparse.ArgumentParser) -> None:
    """The options of a measurement, apart from its model: `job`'s, and `mrig bench`'s."""
    parser.add_argument("--mode", required=True, choices=MODES, help="fixture family served")
    parser.add_argument("--context", required=True, help="context lengths, e.g. 4k,16k")
    parser.add_argument("--engine", action="append", choices=ENGINES, default=[],
                        help="repeatable; default magnitude")
    parser.add_argument(
        "--artifact", action="append", default=[], metavar="ENGINE=REF",
        help="the MLX snapshot (hf:owner/repo@commit or a directory) or Ollama tag "
        "(ollama:name:tag) of the catalog model for an engine the catalog has no form for",
    )
    parser.add_argument("--section", default="context", help=", ".join(SECTIONS))
    parser.add_argument("--drafting", choices=("on", "off"), default="on",
                        help="serve with the catalog drafter where the engine supports it")
    parser.add_argument("--passes", type=int,
                        help="measured passes per engine instead of the balanced schedule")
    parser.add_argument("--stall-seconds", type=float,
                        help="fail after this long without request, stream or engine progress")
    parser.add_argument("--request-seconds", type=float,
                        help="fail when one request runs this long")
    magnitude = parser.add_argument_group("magnitude (our engine)")
    magnitude.add_argument("--tuning-seconds", type=float,
                           help="first-load tuning budget (the engine's standard when unset)")
    magnitude.add_argument("--fresh-tuning", action="store_true",
                           help="tune the first launch afresh, ignoring stored results")
    magnitude.add_argument("--no-tuning-trace", action="store_true",
                           help="no --tuning-trace, for a binary that predates it")
    magnitude.add_argument("--magnitude-binary", type=Path,
                           help="any magnitude-engine (default: this tree's, built by the run)")
    magnitude.add_argument("--magnitude-build", type=SourceBuild.model_validate_json,
                           help='the binary\'s source build, {"commit", "dirty_digest"} JSON')
    magnitude.add_argument("--harness", type=Path, help="magnitude-harness, for rendering")
    magnitude.add_argument("--magnitude-device", default="auto")
    magnitude.add_argument("--magnitude-cache-dir", type=Path,
                           help="the engine's kernel cache, exactly this directory")
    magnitude.add_argument("--magnitude-kernel-cache-root", type=Path,
                           help="persistent per-model, per-binary kernel caches under this root")
    magnitude.add_argument("--magnitude-prefill-tokens", type=int)
    magnitude.add_argument("--magnitude-mtp-proposals", type=int)
    magnitude.add_argument("--magnitude-draft-budget",
                           help="exact draft-budget JSON, or `adaptive`")
    magnitude.add_argument("--magnitude-alpha", type=float,
                           help="the numerical profile's α; the engine's standard when unset")
    magnitude.add_argument("--magnitude-method", choices=("mtp",),
                           help="draft with the target's own head in place of the catalog drafter")
    llama = parser.add_argument_group("llama.cpp")
    llama.add_argument("--llama-binary", type=Path, help="llama-server (default: PATH)")
    llama.add_argument("--llama-draft-proposals", type=int, default=3)
    llama.add_argument("--llama-gpu-layers", type=int, default=99)
    llama.add_argument("--llama-cache-type-k", choices=get_args(KvCacheType), default="f16")
    llama.add_argument("--llama-cache-type-v", choices=get_args(KvCacheType), default="f16")
    llama.add_argument("--llama-context-headroom", type=int, default=0,
                       help="tokens per slot beyond the benchmark context")
    ollama = parser.add_argument_group("Ollama (ollama, ollama-mlx, ollama-registry)")
    ollama.add_argument("--ollama-binary", type=Path, help="ollama (default: PATH)")
    ollama.add_argument("--ollama-models", type=Path, help="model store (default: OLLAMA_MODELS)")
    ollama.add_argument("--ollama-kv-cache-type", choices=("f16", "q8_0", "q4_0"))
    ollama.add_argument("--ollama-flash-attention", choices=("auto", "on", "off"), default="auto")
    ollama.add_argument("--ollama-speculation", choices=("off", "default"), default="off",
                        help="Ollama's own drafting (never the catalog drafter)")
    ollama.add_argument("--ollama-context-headroom", type=int, default=0)
    ollama.add_argument("--ollama-answer-prefill", action="store_true",
                        help="Muse Glimmer only: open the reply at its answer header")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="magnitude-bench", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = root.add_subparsers(dest="command", required=True)
    job = commands.add_parser("job", help="measure inside a run directory (run by mrig bench)")
    job.add_argument("--model", required=True, help="catalog reference <model>:<variant>")
    measurement_options(job)
    job.add_argument("--artifacts", type=Path, required=True, help="catalog acquisition JSON")
    job.add_argument("--output", type=Path, required=True, help="the run directory")
    job.add_argument("--build-identity", type=Path, required=True,
                     help='build identity JSON file ({"commit", "dirty_digest"})')
    return root


def artifact_pairs(values: list[str]) -> dict[str, str]:
    pairs = {}
    for value in values:
        engine, separator, reference = value.partition("=")
        if not separator or engine not in ENGINES or not reference:
            raise ValueError("--artifact takes ENGINE=REF")
        if engine in pairs:
            raise ValueError(f"--artifact names {engine} twice")
        pairs[engine] = reference
    return pairs


def engine_options(args: argparse.Namespace) -> EngineOptions:
    magnitude = MagnitudeOptions(
        device=args.magnitude_device,
        cache_dir=args.magnitude_cache_dir,
        kernel_cache_root=args.magnitude_kernel_cache_root,
        prefill_tokens=args.magnitude_prefill_tokens,
        mtp_proposals=args.magnitude_mtp_proposals,
        draft_budget=args.magnitude_draft_budget,
        method=args.magnitude_method,
        alpha=args.magnitude_alpha,
        tuning_seconds=args.tuning_seconds,
        fresh_tuning=args.fresh_tuning,
        tuning_trace=not args.no_tuning_trace,
        build=args.magnitude_build,
        **{
            key: value
            for key, value in (("binary", args.magnitude_binary), ("harness", args.harness))
            if value is not None
        },
    )
    return EngineOptions(
        drafting=args.drafting == "on",
        magnitude=magnitude,
        llama=LlamaOptions(
            binary=args.llama_binary,
            draft_proposals=args.llama_draft_proposals,
            gpu_layers=args.llama_gpu_layers,
            cache_type_k=args.llama_cache_type_k,
            cache_type_v=args.llama_cache_type_v,
            context_headroom=args.llama_context_headroom,
        ),
        ollama=OllamaOptions(
            binary=args.ollama_binary,
            models=args.ollama_models,
            kv_cache_type=args.ollama_kv_cache_type,
            flash_attention=args.ollama_flash_attention,
            speculation=args.ollama_speculation,
            context_headroom=args.ollama_context_headroom,
            answer_prefill=args.ollama_answer_prefill,
        ),
        watchdog=Watchdog(stall_seconds=args.stall_seconds, request_seconds=args.request_seconds),
        schedule=Schedule(passes=args.passes),
    )


def arguments(args: argparse.Namespace) -> list[str]:
    """The measurement options `args` holds (parsed by `measurement_options`) as `job` takes
    them: every option whose value is not its default."""
    declared = argparse.ArgumentParser(add_help=False)
    measurement_options(declared)

    def text(value) -> str:
        return value.model_dump_json() if isinstance(value, SourceBuild) else str(value)

    result = []
    for action in declared._actions:
        value = getattr(args, action.dest)
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            result += [flag] if value else []
        elif isinstance(action, argparse._AppendAction):
            result += [token for item in value for token in (flag, text(item))]
        elif value != action.default:
            result += [flag, text(value)]
    return result


async def managed(coroutine):
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    # SIGINT is handled by asyncio.run; SIGTERM must take the same cleanup path.
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        return await coroutine
    finally:
        loop.remove_signal_handler(signal.SIGTERM)


def job(argv: list[str], args: argparse.Namespace) -> int:
    from .runner import run as measure

    acquisition = Acquisition.model_validate_json(args.artifacts.read_text())
    if acquisition.model != args.model:
        raise ValueError(f"--artifacts acquired {acquisition.model}, not {args.model}")
    options = engine_options(args)
    if args.ollama_models is not None:
        # Artifact preparation and the adapter read the same store.
        os.environ["OLLAMA_MODELS"] = str(args.ollama_models.expanduser().absolute())
    selected = targets(args.engine or ["magnitude"], artifact_pairs(args.artifact), acquisition)
    summary = asyncio.run(
        managed(
            measure(
                RunStore(args.output),
                args.mode,
                selected,
                choices(args.section, tuple(SECTIONS)),
                contexts(args.context),
                acquisition,
                args.artifacts,
                json.loads(args.build_identity.read_text()),
                options,
                ["magnitude-bench", *argv],
                lambda message: print(message, file=sys.stderr, flush=True),
            )
        )
    )
    print(f"{summary['status']}: {summary['path']}/report.md\n{summary.get('error') or ''}")
    return 0 if summary["status"] == "completed" else 130 if summary["status"] == "cancelled" else 1


def check(args: argparse.Namespace) -> None:
    """Every measurement option's meaning, checked before anything runs; raises ValueError."""
    catalog_reference(args.model)
    choices(args.section, tuple(SECTIONS))
    contexts(args.context)
    artifact_pairs(args.artifact)
    engine_options(args)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    try:
        check(args)
        return job(argv, args)
    except (ValueError, OSError) as error:
        print(f"magnitude-bench: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
