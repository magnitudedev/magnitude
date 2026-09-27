"""Full-model serving qualification; one worker, serial paired generation options."""

import argparse
import asyncio
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

from engine.serving.requests import ChatRequest
from engine.serving.responses import ChatResponse
from engine.serving.runtime import Config
from engine.serving.session import ChatFinished, ChatService


async def main(args):
    config = Config(
        target=str(args.model),
        model="qualification",
        memory_bytes=args.memory_bytes,
        context_tokens=1024,
        parallel_sequences=1,
        prefill_tokens=128,
        output_capacity=64,
    )
    start = perf_counter()
    service = await ChatService.open(config)
    result = {
        "platform": platform.platform(),
        "startup_seconds": perf_counter() - start,
        "properties": service.properties.model_dump(mode="json"),
        "runs": [],
    }
    cases = {
        "prose": {"messages": [{"role": "user", "content": "Reply with just OK."}]},
        "json_const": {
            "messages": [{"role": "user", "content": "Return the requested JSON object."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "const": "The quick brown fox jumps over the lazy dog. This exact sentence is required."
                            }
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        "required_tool": {
            "messages": [
                {"role": "user", "content": "Call weather for the city Paris. Do not explain."}
            ],
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
        },
    }
    try:
        for name, fields in cases.items():
            order = (0, 32) + tuple(
                quantum
                for pair in range(args.pairs)
                for quantum in ((0, 32) if pair % 2 == 0 else (32, 0))
            )
            for index, quantum in enumerate(order):
                # The service/model stay resident. Record these generation options
                # separately from the unchanged execution composition identity.
                service.properties = service.properties.model_copy(
                    update={"forced_quantum": quantum}
                )
                body = ChatRequest.model_validate(
                    {
                        "model": "qualification",
                        "max_tokens": 128,
                        "temperature": 0.0,
                        "reasoning_effort": "none",
                        **fields,
                    }
                )
                start = perf_counter()
                prompt = service.prepare(body)
                prepared = perf_counter()
                response = ChatResponse("qualification")
                finished = None
                async for event in service.events(body, prompt):
                    if isinstance(event, ChatFinished):
                        finished = event
                    else:
                        response.semantic(event, retain=True)
                elapsed = perf_counter() - start
                if finished is None:
                    raise AssertionError("missing terminal event")
                complete = response.complete(finished)
                message = complete["choices"][0]["message"]
                if name == "json_const":
                    assert json.loads(message["content"]) == {
                        "text": fields["response_format"]["json_schema"]["schema"]["properties"][
                            "text"
                        ]["const"]
                    }
                elif name == "required_tool":
                    assert finished.reason == "tool_calls"
                    assert len(message["tool_calls"]) == 1
                    call = message["tool_calls"][0]["function"]
                    assert call["name"] == "weather" and isinstance(
                        json.loads(call["arguments"])["city"], str
                    )
                else:
                    assert message["content"] and finished.reason == "stop"
                run = {
                    "case": name,
                    "warmup": index < 2,
                    "forced_quantum": quantum,
                    "prepare_seconds": prepared - start,
                    "total_seconds": elapsed,
                    "prompt_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                    "template_identity": prompt.profile.template_identity,
                    "response": complete,
                }
                result["runs"].append(run)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "case": name,
                            "quantum": quantum,
                            "seconds": elapsed,
                            "generated": finished.native.generated_tokens,
                            "forced": finished.native.forced_tokens,
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
    parser.add_argument("--memory-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--pairs", type=int, default=4)
    args = parser.parse_args()
    if args.pairs < 4:
        parser.error("qualification requires at least four measured pairs")
    asyncio.run(main(args))
