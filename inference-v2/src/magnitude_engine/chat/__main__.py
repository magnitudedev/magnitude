"""Stream a text conversation with engine progress and per-turn diagnostics."""

import argparse
import sys
from contextlib import closing
from pathlib import Path
from time import perf_counter_ns

from magnitude_engine.artifacts.tokenizer import TokenizerArtifact
from magnitude_engine.engine.delivery import Finished, PrefillProgress
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.worker.host import Worker
from magnitude_engine.worker.launch import add_engine_arguments, engine_from_arguments

from .session import ChatSession
from .terminal import Terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_arguments(parser)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--system", default="")
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=None,
        help="set the model template's enable_thinking option; otherwise use its default",
    )
    parser.add_argument(
        "--prompt", help="run one turn and exit instead of reading interactive input"
    )
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    engine = engine_from_arguments(args, parser)
    sampling = SamplingPolicy(temperature=args.temperature, seed=args.seed)
    terminal = Terminal(sys.stdout)
    terminal.status("Loading model…")
    with Worker.start(engine=engine) as worker:
        artifact = TokenizerArtifact.load(Path(worker.properties["target_path"]))
        session = ChatSession(worker, artifact, args.system, thinking=args.thinking)
        terminal.clear_status()
        terminal.write(
            f"Model: {worker.properties['target_path']}\n"
            f"Execution: {worker.properties['program_implementation']}"
            f" · speculation: {worker.properties['speculative_backend'] or 'none'}\n"
            f"Context: {worker.properties['context_tokens']:,}"
            f" · output limit: {args.max_tokens:,} · temperature: {args.temperature}\n"
            "Prefill progress counts bulk prompt tokens; the last prompt token produces the first "
            "output. Rates use active worker service; counts include special/EOS tokens.\n"
            "Commands: /reset clears conversation history, /exit quits. Ctrl-C cancels a turn.\n\n"
        )
        while True:
            try:
                text = args.prompt if args.prompt is not None else input("You: ")
            except (EOFError, KeyboardInterrupt):
                terminal.write("\n")
                break
            if text.strip() == "/exit":
                break
            if text.strip() == "/reset":
                session.reset()
                terminal.write("Conversation cleared.\n\n")
                if args.prompt is not None:
                    break
                continue
            if not text.strip():
                if args.prompt is not None:
                    break
                continue
            terminal.begin()
            start = perf_counter_ns()
            try:
                with closing(session.respond(text, sampling, args.max_tokens)) as events:
                    for event in events:
                        if isinstance(event, PrefillProgress):
                            terminal.progress(event)
                        elif isinstance(event, Finished):
                            terminal.finish(event, perf_counter_ns() - start)
                        else:
                            terminal.text(event)
            except KeyboardInterrupt:
                terminal.clear_status()
                terminal.write("\nCancelled; conversation history unchanged.\n\n")
            except (ValueError, RuntimeError, TimeoutError, OverflowError) as error:
                terminal.clear_status()
                terminal.write(f"\nError: {error}\n\n")
                if args.prompt is not None or not worker.available:
                    return 1
            if args.prompt is not None:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
