"""Matched numerical work with and without a permissive output constraint."""

import argparse
import asyncio
import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from engine.generation.constraints import ConstraintPlan
from engine.serving.requests import ChatRequest
from engine.serving.responses import ChatResponse
from engine.serving.runtime import Config
from engine.serving.session import ChatFinished, ChatService


async def main(args):
    started = perf_counter()
    service = await ChatService.open(
        Config(
            target=str(args.model),
            model="qualification",
            memory_bytes=16 * 1024**3,
            context_tokens=1024,
            parallel_sequences=1,
            prefill_tokens=128,
            output_capacity=64,
            forced_quantum=0,
        )
    )
    result = {
        "platform": platform.platform(),
        "startup_seconds": perf_counter() - started,
        "properties": service.properties.model_dump(mode="json"),
        "runs": [],
        "purpose": "same prompt and generated token IDs; constraint vs none",
        "constraint_kind": args.case,
    }
    cases = (
        (("required_tool", ""),)
        if args.case == "tool"
        else (("short_prefill", ""), ("long_prefill", "Background information. " * 100))
    )
    tool_fields = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": "required",
    }
    try:
        for name, context in cases:
            expected = None
            order = (False, True) + tuple(
                flag
                for pair in range(args.pairs)
                for flag in ((False, True) if pair % 2 == 0 else (True, False))
            )
            for index, constrained in enumerate(order):
                body = ChatRequest.model_validate(
                    {
                        "model": "qualification",
                        "max_tokens": 64,
                        "temperature": 0,
                        "reasoning_effort": "none",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Call weather for the city Paris. Do not explain."
                                if args.case == "tool"
                                else context
                                + "List the integers from 1 to 200 separated by spaces. Output only the list.",
                            }
                        ],
                        **(tool_fields if args.case == "tool" else {}),
                    }
                )
                started = perf_counter()
                prompt = service.prepare(body)
                if args.case != "tool":
                    assert prompt.constraint is None
                if args.case == "tool":
                    assert prompt.constraint is not None
                    if not constrained:
                        prompt = replace(prompt, constraint=None)
                elif constrained:
                    prompt = replace(
                        prompt,
                        constraint=ConstraintPlan(
                            artifact_identity=service.properties.artifact_identity,
                            template_identity=prompt.profile.template_identity,
                            grammar=r'root ::= ([^\n] | "\n")*',
                        ),
                    )
                prepared = perf_counter()
                response = ChatResponse("qualification")
                terminal = None
                tokens = ()
                elapsed = None
                async for event in service.events(body, prompt):
                    if isinstance(event, ChatFinished):
                        # Read the accepted history only after the measured terminal
                        # boundary, before normal request cleanup removes its owner.
                        elapsed = perf_counter() - started
                        terminal = event
                        identity = event.native.identity
                        tokens = await asyncio.wrap_future(
                            service.worker.call(
                                lambda owner: tuple(
                                    owner.engine.requests[identity].generation.generated
                                )
                            )
                        )
                    else:
                        response.semantic(event, retain=True)
                assert terminal is not None and elapsed is not None
                observed = (prompt.tokens, tokens)
                if expected is None:
                    expected = observed
                assert observed == expected, (
                    "mask altered the numerical work; comparison is invalid"
                )
                assert terminal.native.forced_tokens == 0
                complete = response.complete(terminal)
                if args.case == "tool":
                    assert terminal.reason == "tool_calls"
                    calls = complete["choices"][0]["message"]["tool_calls"]
                    assert len(calls) == 1 and calls[0]["function"]["name"] == "weather"
                    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
                run = {
                    "case": name,
                    "constrained": constrained,
                    "warmup": index < 2,
                    "through_terminal_seconds": elapsed,
                    "prepare_seconds": prepared - started,
                    "prompt_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                    "prompt_tokens": list(prompt.tokens),
                    "generated_tokens": list(tokens),
                    "response": complete,
                }
                result["runs"].append(run)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "case": name,
                            "constrained": constrained,
                            "seconds": elapsed,
                            "generated": len(tokens),
                        }
                    ),
                    flush=True,
                )
    finally:
        await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--case", choices=("prose", "tool"), default="prose")
    args = parser.parse_args()
    if args.pairs < 4:
        parser.error("at least four measured pairs required")
    asyncio.run(main(args))
