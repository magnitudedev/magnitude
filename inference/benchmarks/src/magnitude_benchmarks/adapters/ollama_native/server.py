"""Magnitude's managed Ollama benchmark adapter.

Owns one ``ollama serve`` process with a private home directory, loads exactly one model at
the benchmark context, and serves the benchmark's wire protocol on top of Ollama's native
API so that prefill and decode times are the ones Ollama's runner reports.

Ollama cannot disable its prompt cache. The model is therefore unloaded and loaded again
after every response, before the response's ``[DONE]``, so each request starts with an empty
cache and time to first token never includes a load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import translate

STARTUP_SECONDS = 120
STARTUP_ATTEMPTS = 3
# Ollama's scheduler logs the runner it starts for a model; that line is the evidence.
RUNNER_MARKERS = {
    b"using llama-server for model": "llama-server",
    b"starting mlx runner subprocess": "mlx",
}
LOAD_SECONDS = 900
REQUEST_SECONDS = 1800


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    value.add_argument("--ollama", type=Path, required=True)
    value.add_argument("--store", type=Path, required=True)
    source = value.add_mutually_exclusive_group(required=True)
    source.add_argument("--gguf", type=Path, help="import this GGUF file (no requantisation)")
    source.add_argument("--registry-model", help="serve this already pulled registry model")
    value.add_argument("--served-model", required=True)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, required=True)
    value.add_argument("--base-path", type=Path, required=True)
    value.add_argument("--max-concurrent-requests", type=int, required=True)
    value.add_argument("--context-capacity", type=int, required=True)
    value.add_argument("--kv-cache-type", default=None)
    value.add_argument("--flash-attention", choices=("auto", "on", "off"), default="auto")
    value.add_argument("--speculation", choices=("off", "default"), default="off")
    value.add_argument(
        "--context-headroom",
        type=int,
        default=0,
        help="tokens allocated beyond --context-capacity",
    )
    value.add_argument(
        "--answer-prefill",
        action="store_true",
        help="Muse Glimmer only: raw completion of Ollama's rendered prompt plus the answer header",
    )
    value.add_argument(
        "--expected-prompt-tokens",
        type=Path,
        help="JSON map of prompt identity to the prompt tokens counted during preparation",
    )
    return value


def imported_name(path: Path) -> str:
    """A store name bound to the file's identity, so a changed file is never served stale."""
    stat = path.stat()
    identity = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return "session-bench-" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def server_environment(args: argparse.Namespace, port: int) -> dict[str, str]:
    env = os.environ.copy()
    home = args.base_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        HOME=str(home),
        OLLAMA_HOST=f"127.0.0.1:{port}",
        OLLAMA_MODELS=str(args.store),
        OLLAMA_NUM_PARALLEL=str(args.max_concurrent_requests),
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_KEEP_ALIVE="-1",
        OLLAMA_NOPRUNE="1",
        OLLAMA_NO_CLOUD="1",
    )
    # Ambient settings must not change the measured configuration.
    for key in ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION"):
        env.pop(key, None)
    if args.kv_cache_type:
        env["OLLAMA_KV_CACHE_TYPE"] = args.kv_cache_type
    if args.flash_attention != "auto":
        env["OLLAMA_FLASH_ATTENTION"] = "1" if args.flash_attention == "on" else "0"
    return env


class Ollama:
    """The owned Ollama server and its single loaded model."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # The context Ollama is asked to allocate: the benchmark's plus any headroom.
        self.allocation = args.context_capacity + args.context_headroom
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.env = server_environment(args, self.port)
        self.process: subprocess.Popen | None = None
        self.model = ""
        self.evidence: dict[str, Any] | None = None
        self.context: int | None = None
        self.runner: str | None = None
        self.draft_loaded = False
        # The llama.cpp runner's command line; Ollama adds its speculation flags from the model.
        self.runner_command: str | None = None
        # Speculation statistics the runner logged, newest last.
        self.drafted: list[tuple[int, int]] = []
        # Full size of the last prompt Ollama truncated to fit the context.
        self.truncated: int | None = None
        # What the runner logged while loading: layers on the GPU per model, or the MLX device.
        self.gpu_layers: list[tuple[int, int]] = []
        self.mlx_device: str | None = None
        expected = args.expected_prompt_tokens
        self.expected: dict[str, int] = json.loads(expected.read_text()) if expected else {}
        # One request, or one reload, at a time: a request never observes a half-loaded model.
        self.turn = threading.Lock()

    def call(self, path: str, body: dict | None = None, timeout: float = 60) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.endpoint + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"null")

    def serve(self) -> None:
        """Start ``ollama serve`` and wait for it to answer.

        Startup scans the store; a store another Ollama is writing to at that moment (a pull
        in progress) can make it exit, so a failed start is retried.
        """
        for attempt in range(STARTUP_ATTEMPTS):
            self.process = subprocess.Popen(
                [str(self.args.ollama), "serve"],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            threading.Thread(target=self.forward, args=(self.process,), daemon=True).start()
            deadline = time.monotonic() + STARTUP_SECONDS
            while self.process.poll() is None:
                try:
                    self.version = self.call("/api/version", timeout=5)["version"]
                    return
                except (urllib.error.URLError, OSError):
                    if time.monotonic() > deadline:
                        self.stop()
                        raise RuntimeError("ollama serve did not start") from None
                    time.sleep(0.25)
            print(
                f"adapter: ollama serve exited during startup (attempt {attempt + 1})", flush=True
            )
            time.sleep(2)
        raise RuntimeError("ollama serve exited during startup")

    def forward(self, process: subprocess.Popen) -> None:
        """Relay Ollama's log to the engine log, noting which runner it starts."""
        assert process.stdout is not None
        for line in process.stdout:
            for marker, runner in RUNNER_MARKERS.items():
                if marker in line:
                    self.runner = runner
            if b'msg="Loaded draft model"' in line:
                self.draft_loaded = True
            truncated = translate.truncated_prompt_tokens(line)
            if truncated is not None:
                self.truncated = truncated
            layers = translate.offloaded_layers(line)
            if layers is not None:
                self.gpu_layers.append(layers)
            device = translate.mlx_device(line)
            if device is not None:
                self.mlx_device = device
            launch = translate.llama_launch(line)
            if launch is not None:
                self.runner_command = launch
                self.draft_loaded = "--spec-type" in launch
            stats = translate.speculation_stats(line)
            if stats is not None:
                self.drafted.append(stats)
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    def start(self) -> None:
        self.serve()
        if self.args.gguf is not None:
            self.model = imported_name(self.args.gguf)
            modelfile = self.args.base_path / "Modelfile"
            modelfile.write_text(f"FROM {self.args.gguf.resolve()}\n")
            subprocess.run(
                [str(self.args.ollama), "create", self.model, "-f", str(modelfile)],
                env=self.env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        else:
            # Pulling is preparation, never part of a run: the store must already hold it.
            self.model = self.args.registry_model
        self.show = self.call("/api/show", {"model": self.model})
        family = (self.show.get("details") or {}).get("family")
        if self.args.answer_prefill and family != translate.GLIMMER_FAMILY:
            raise RuntimeError(f"answer prefill is for {translate.GLIMMER_FAMILY}, not {family}")
        self.load()
        if self.args.answer_prefill:
            self.verify_rendering()
        entry = self.loaded()
        # The runner's load lines arrive through the log relay, shortly after the load returns.
        deadline = time.monotonic() + 5
        while not (self.gpu_layers or self.mlx_device) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.evidence = {
            "ready": True,
            "served_model": self.args.served_model,
            "ollama_version": self.version,
            "ollama_model": self.model,
            "runner": self.runner,
            "draft_model_loaded": self.draft_loaded,
            "runner_command": self.runner_command,
            "speculation": self.args.speculation,
            "model_format": (self.show.get("details") or {}).get("format"),
            "quantization": (self.show.get("details") or {}).get("quantization_level"),
            "renderer": self.show.get("renderer"),
            "parser": self.show.get("parser"),
            "context_capacity": self.args.context_capacity,
            "context_headroom": self.args.context_headroom,
            "answer_prefill": self.args.answer_prefill,
            "allocated_context": entry.get("context_length"),
            "max_concurrent_requests": self.args.max_concurrent_requests,
            "gpu_layers": [list(layers) for layers in self.gpu_layers],
            "mlx_device": self.mlx_device,
            "size_bytes": entry.get("size"),
            "size_vram_bytes": entry.get("size_vram"),
            "running": entry,
        }

    def verify_rendering(self) -> None:
        """Refuse to run unless the reproduced prompt is the one Ollama renders itself.

        A fixed conversation is evaluated once through ``/api/chat`` with ``think`` false and
        once through the raw route without the answer header; both must evaluate the same
        number of prompt tokens, and the header must add exactly its own tokens.
        """
        messages = [
            {"role": "system", "content": "Rendering check."},
            {"role": "user", "content": "Reply with the word ready."},
        ]
        options = {"num_ctx": self.allocation, "num_predict": 1, "temperature": 0}
        chat = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "options": options,
        }
        raw = translate.prefilled_request(chat) | {"stream": False}
        bare = raw | {"prompt": raw["prompt"].removesuffix(translate.GLIMMER_ANSWER_HEADER)}
        counts = []
        for path, body in (("/api/chat", chat), ("/api/generate", bare), ("/api/generate", raw)):
            counts.append(self.call(path, body, timeout=LOAD_SECONDS)["prompt_eval_count"])
            self.reload()
        rendered, reproduced, prefilled = counts
        if rendered != reproduced or prefilled <= reproduced:
            raise RuntimeError(
                "answer prefill does not reproduce Ollama's prompt: chat evaluated "
                f"{rendered} tokens, raw {reproduced}, raw with the answer header {prefilled}"
            )
        self.rendering = {
            "chat_prompt_tokens": rendered,
            "raw_prompt_tokens": reproduced,
            "answer_header_tokens": prefilled - reproduced,
        }
        print(f"adapter: answer prefill rendering verified: {self.rendering}", flush=True)

    def loaded(self) -> dict[str, Any]:
        models = self.call("/api/ps")["models"]
        if len(models) != 1 or models[0].get("model") not in (self.model, self.model + ":latest"):
            raise RuntimeError(f"unexpected running Ollama models: {models}")
        return models[0]

    def load(self) -> None:
        self.gpu_layers.clear()
        self.call(
            "/api/generate",
            {
                "model": self.model,
                "keep_alive": -1,
                "options": {"num_ctx": self.allocation},
            },
            timeout=LOAD_SECONDS,
        )
        self.context = self.loaded().get("context_length")
        if self.context != self.allocation:
            raise RuntimeError(f"Ollama reports context {self.context}, expected {self.allocation}")

    def reload(self) -> float:
        """Unload and load the model; returns the elapsed milliseconds."""
        started = time.monotonic()
        self.call("/api/generate", {"model": self.model, "keep_alive": 0}, timeout=LOAD_SECONDS)
        deadline = time.monotonic() + LOAD_SECONDS
        while self.call("/api/ps")["models"]:
            if time.monotonic() > deadline:
                raise RuntimeError("Ollama did not unload the model")
            time.sleep(0.05)
        self.load()
        return (time.monotonic() - started) * 1000

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def handler(ollama: Ollama) -> type[BaseHTTPRequestHandler]:
    args = ollama.args

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *values: Any) -> None:
            print("adapter: " + format % values, flush=True)

        def reply(self, status: int, value: Any) -> None:
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def body(self) -> dict[str, Any]:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))

        def do_GET(self) -> None:
            if self.path != "/magnitude/benchmark/readiness":
                return self.reply(404, {"error": "not found"})
            if ollama.evidence is None:
                return self.reply(503, {"error": "model load is incomplete"})
            self.reply(200, ollama.evidence)

        def do_POST(self) -> None:
            if ollama.evidence is None:
                return self.reply(503, {"error": "model load is incomplete"})
            if self.path == "/session-bench/count":
                return self.count()
            if self.path == "/v1/chat/completions":
                return self.chat()
            self.reply(404, {"error": "not found"})

        def native(self, request: dict[str, Any]):
            path = "/api/chat"
            if args.answer_prefill:
                path, request = "/api/generate", translate.prefilled_request(request)
            data = json.dumps(request).encode()
            return urllib.request.urlopen(
                urllib.request.Request(
                    ollama.endpoint + path,
                    data=data,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=REQUEST_SECONDS,
            )

        def count(self) -> None:
            """Prompt tokens as Ollama's runner renders and counts them.

            Ollama has no tokenise route. The prompt is evaluated for one output token and its
            reported prompt count is returned (the count includes tokens reused from the prompt
            cache, so counting launches keep the cache). A prompt the llama.cpp runner rejects
            as larger than the context is counted from that rejection, which states its size;
            one Ollama truncates itself is counted from its truncation warning.
            """
            request = translate.chat_request(self.body(), ollama.model, ollama.allocation)
            request["options"]["num_predict"] = 1
            with ollama.turn:
                ollama.truncated = None
                try:
                    with self.native(request) as response:
                        final = [json.loads(line) for line in response if line.strip()][-1]
                except urllib.error.HTTPError as error:
                    text = error.read().decode(errors="replace")
                    tokens = translate.overflow_prompt_tokens(error.code, text)
                    if tokens is None:
                        return self.reply(error.code, {"error": text})
                    return self.reply(200, {"prompt_tokens": tokens, "basis": "overflow-error"})
                truncated = ollama.truncated
            if truncated is not None:
                return self.reply(200, {"prompt_tokens": truncated, "basis": "truncation-warning"})
            self.reply(
                200, {"prompt_tokens": final["prompt_eval_count"], "basis": "prompt-evaluation"}
            )

        def chat(self) -> None:
            body = self.body()
            request = translate.chat_request(body, ollama.model, ollama.allocation)
            expected = ollama.expected.get(translate.prompt_identity(body))
            if args.speculation == "off" and ollama.runner == "mlx":
                # The MLX runner parks its drafter for a request that asks for logprobs.
                request["logprobs"] = True
            identity = "chatcmpl-" + uuid.uuid4().hex
            with ollama.turn:
                try:
                    response = self.native(request)
                except urllib.error.HTTPError as error:
                    return self.reply(error.code, {"error": error.read().decode(errors="replace")})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def event(value: Any) -> None:
                    text = value if isinstance(value, str) else json.dumps(value)
                    self.wfile.write(f"data: {text}\n\n".encode())
                    self.wfile.flush()

                def choice(delta: dict, finish: str | None = None) -> dict:
                    return {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "model": args.served_model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }

                tool_calls = 0
                final = None
                ollama.drafted.clear()
                with response:
                    for line in response:
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            return event({"error": chunk["error"]})
                        delta = translate.delta(chunk, tool_calls)
                        tool_calls += len(delta.get("tool_calls", ()))
                        if delta:
                            event(choice(delta))
                        if chunk.get("done"):
                            final = chunk
                if final is None:
                    return event({"error": "Ollama stream ended without a final chunk"})
                # Ollama's MLX runner evaluates a prompt larger than the context in full and its
                # llama.cpp runner can shift context, so the evaluated prompt is checked itself.
                if expected is not None and final.get("prompt_eval_count") != expected:
                    return event(
                        {
                            "error": "Ollama evaluated "
                            f"{final.get('prompt_eval_count')} prompt tokens, expected {expected}"
                        }
                    )
                drafted = None
                if ollama.runner == "mlx" or ollama.draft_loaded:
                    # The runner logs its speculation statistics as the request ends. A draft
                    # head inside the target (MTP) is not announced at load on the MLX runner,
                    # so every MLX request is checked; the llama.cpp runner is checked when
                    # Ollama launched it with speculation flags.
                    deadline = time.monotonic() + (2 if args.speculation == "default" else 0.3)
                    while not ollama.drafted and time.monotonic() < deadline:
                        time.sleep(0.02)
                    drafted = ollama.drafted[-1] if ollama.drafted else None
                if args.speculation == "off" and drafted is not None:
                    return event({"error": f"Ollama drafted {drafted[0]} tokens in a plain run"})
                try:
                    evidence = translate.terminal(final, ollama.context, drafted)
                    finish = translate.finish_reason(final, tool_calls > 0)
                except ValueError as error:
                    return event({"error": str(error)})
                event(choice({}, finish))
                event(
                    {
                        "id": identity,
                        "object": "chat.completion.chunk",
                        "model": args.served_model,
                        "choices": [],
                        **evidence,
                    }
                )
                # Empty the prompt cache before the response is complete.
                reload_ms = ollama.reload()
                print(f"adapter: reloaded model in {reload_ms:.0f} ms", flush=True)
                event("[DONE]")

    return Handler


def main() -> None:
    args = parser().parse_args()
    args.base_path = args.base_path.resolve()
    args.base_path.mkdir(parents=True, exist_ok=True)
    ollama = Ollama(args)
    server = ThreadingHTTPServer((args.host, args.port), handler(ollama))

    def shutdown(*_: Any) -> None:
        ollama.stop()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ollama.start()
    except BaseException:
        ollama.stop()
        raise
    code = ollama.process.wait() if ollama.process else 1
    sys.exit(code or 1)


if __name__ == "__main__":
    main()
