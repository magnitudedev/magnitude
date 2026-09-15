"""Run as a file in the PoC venv; no dependency on Magnitude's numerical runtime.

The synchronous listener and Model share their owner thread. Only the explicit
non-counting mode imports the PoC, loads weights, or performs GPU work.
"""

import argparse
import json
import time
import traceback
import uuid
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from transformers import AutoTokenizer


def render(tokenizer, body):
    if body.get("tools"):
        raise ValueError("PoC benchmark bridge does not implement tool responses")
    text = tokenizer.apply_chat_template(
        body["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer.encode(text, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--warmup-requests", type=Path)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "config.json").read_text())
    configured_eos = config.get("eos_token_id", tokenizer.eos_token_id)
    eos = set(configured_eos if isinstance(configured_eos, list) else [configured_eos])
    eos.add(tokenizer.eos_token_id)
    readiness = {
        "ready": True,
        "served_model": "session-bench",
        "context_capacity": args.context,
        "parallel_sequences": 1,
        "count_only": args.count_only,
        "prefix_cache": False,
    }

    with ExitStack() as stack:
        session = None
        if not args.count_only:
            if args.warmup_requests is None:
                raise ValueError("Measurement mode requires full-shape warmup requests")
            import tilelang
            from tilelang_poc.model import Model

            tilelang.set_log_level("WARNING")
            resident = stack.enter_context(Model(args.model))
            session = stack.enter_context(resident.create(args.context))
            # Arena capacity participates in kernel specialization. Grow it once
            # before warming short and long requests, not between those shapes.
            session.state.reserve(args.context)
            session.reset()
            readiness["prefill_block_tokens"] = resident.prefills.block_tokens

        def generate(body, emit):
            tokens = render(tokenizer, body)
            limit = body.get("max_tokens")
            if type(limit) is not int or not 1 <= limit <= 256:
                raise ValueError("PoC bridge requires 1..256 output tokens")
            if body.get("temperature", 0) != 0 or body.get("n", 1) != 1:
                raise ValueError("PoC bridge supports one greedy completion only")
            if not tokens or len(tokens) + limit > args.context:
                raise ValueError("Prompt and output allowance exceed context capacity")
            session.reset()
            session.state.reserve(len(tokens) + limit)
            session.execution.complete()
            compilers = (session.execution.compiler, resident.prefills.execution.compiler)

            def compilation_counts():
                return (
                    sum(compiler.builds for compiler in compilers),
                    sum(compiler.loads for compiler in compilers),
                )

            before = compilation_counts()
            tick = time.perf_counter()
            logits = session.prefill(tokens)
            session.execution.complete()
            prompt_ms = (time.perf_counter() - tick) * 1000
            generated, decode_ms, finish = [], 0.0, "length"
            for index in range(limit):
                tick = time.perf_counter()
                if index:
                    logits = session.step(generated[-1])
                token = session.select(logits)
                decode_ms += (time.perf_counter() - tick) * 1000
                generated.append(token)
                if token in eos:
                    finish = "stop"
                    break
                emit(generated)
            session.execution.complete()
            after = compilation_counts()
            return {
                "prompt_tokens": len(tokens),
                "generated_tokens": generated,
                "prompt_ms": prompt_ms,
                "predicted_ms": decode_ms,
                "finish_reason": finish,
                "timed_kernel_builds": after[0] - before[0],
                "timed_kernel_loads": after[1] - before[1],
            }

        if session is not None:
            warmups = []
            for body in json.loads(args.warmup_requests.read_text()):
                print("Warming prepared request", flush=True)
                warmups.append(generate(body, lambda tokens: None))
            session.reset()
            session.execution.complete()
            (args.evidence / "warmup.json").write_text(json.dumps(warmups, indent=2))
            print("Full-shape warmup complete", flush=True)

        class Handler(BaseHTTPRequestHandler):
            def respond(self, code, value):
                content = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                self.respond(200, readiness) if self.path == "/health" else self.respond(404, {})

            def do_POST(self):
                streaming = False
                try:
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 8 * 1024**2:
                        raise ValueError("Invalid request size")
                    body = json.loads(self.rfile.read(size))
                    if self.path == "/count":
                        self.respond(200, {"prompt_tokens": len(render(tokenizer, body))})
                        return
                    if self.path != "/v1/chat/completions" or session is None:
                        self.respond(404, {})
                        return
                    if body.get("stream") is not True:
                        raise ValueError("Only streaming requests are supported")
                    identity = "chatcmpl-" + uuid.uuid4().hex
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    streaming = True

                    def event(payload):
                        value = json.dumps({"id": identity, **payload})
                        self.wfile.write(f"data: {value}\n\n".encode())
                        self.wfile.flush()

                    emitted = ""

                    def emit(tokens):
                        nonlocal emitted
                        text = tokenizer.decode(tokens, skip_special_tokens=True)
                        # Wait for complete byte-fallback Unicode sequences.
                        if text.endswith("\ufffd"):
                            return
                        if not text.startswith(emitted):
                            raise ValueError("Tokenizer rewrote already emitted text")
                        suffix, emitted = text[len(emitted) :], text
                        if suffix:
                            event(
                                {
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": suffix},
                                            "finish_reason": None,
                                        }
                                    ]
                                }
                            )

                    result = generate(body, emit)
                    emit(result["generated_tokens"])
                    with (args.evidence / "native-requests.jsonl").open("a") as evidence:
                        evidence.write(json.dumps({"id": identity, **result}) + "\n")
                    if result["timed_kernel_builds"] or result["timed_kernel_loads"]:
                        raise RuntimeError(
                            "Warm measurement encountered kernel compilation/loading"
                        )
                    event(
                        {
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}
                            ]
                        }
                    )
                    prompt, output = result["prompt_tokens"], len(result["generated_tokens"])
                    event(
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt,
                                "completion_tokens": output,
                                "total_tokens": prompt + output,
                                "prompt_tokens_details": {"cached_tokens": 0},
                            },
                            "timings": {
                                "cache_n": 0,
                                "prompt_n": prompt,
                                "predicted_n": output,
                                "prompt_ms": result["prompt_ms"],
                                "predicted_ms": result["predicted_ms"],
                            },
                        }
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as exc:
                    traceback.print_exc()
                    if streaming:
                        self.wfile.write(
                            ("data: " + json.dumps({"error": str(exc)}) + "\n\n").encode()
                        )
                        self.wfile.flush()
                    else:
                        self.respond(400, {"error": str(exc)})

        server = stack.enter_context(HTTPServer(("127.0.0.1", args.port), Handler))
        server.serve_forever()


if __name__ == "__main__":
    main()
