from __future__ import annotations

import copy
import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.server import (
    GenerationContext,
    Response,
    SequenceStateMachine,
    ToolCallFormatter,
    _process_control_tokens,
    process_message_content,
)

from .protocol import ChatRequest


@dataclass(frozen=True)
class NativeEvidence:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    prompt_ms: float
    generation_ms: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class GeneratedEvent:
    kind: str
    value: Any


@dataclass(frozen=True)
class _GenerationJob:
    request: ChatRequest
    cancelled: threading.Event
    emit: Callable[[GeneratedEvent | Exception | None], None]


class MlxGenerator:
    """Single-model, single-generation MLX-LM adapter with native prompt-cache reuse."""

    def __init__(
        self, model_path: str, *, prefill_step_size: int, prompt_cache_entries: int
    ):
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        if not self.tokenizer.has_chat_template:
            raise RuntimeError("model tokenizer has no chat template")
        if not self.tokenizer.has_tool_calling:
            raise RuntimeError("model tokenizer has no MLX-LM tool-call parser")
        self.prefill_step_size = prefill_step_size
        self.prompt_cache = LRUPromptCache(max_size=prompt_cache_entries)
        self.model_key = (model_path, None, None)
        self._admission = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0

    @contextmanager
    def _generation_turn(self, cancelled: threading.Event):
        with self._admission:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving_ticket:
                self._admission.wait(timeout=0.1)
        try:
            yield not cancelled.is_set()
        finally:
            with self._admission:
                self._serving_ticket += 1
                self._admission.notify_all()

    def _state_machine(self) -> tuple[SequenceStateMachine, dict[tuple[int, ...], str]]:
        transitions: dict[str, list[tuple[tuple[int, ...], str | None]]] = {
            "normal": []
        }
        sequences: dict[tuple[int, ...], str] = {}
        stops: list[tuple[tuple[int, ...], None]] = []
        for token in self.tokenizer.eos_token_ids:
            sequence = (token,)
            stops.append((sequence, None))
            sequences[sequence] = self.tokenizer.convert_ids_to_tokens(token)
        transitions["normal"].extend(stops)
        tool_start = self.tokenizer.tool_call_start_tokens
        tool_end = self.tokenizer.tool_call_end_tokens
        transitions["normal"].append((tool_start, "tool"))
        transitions["tool"] = ([(tool_end, "normal")] if tool_end else []) + stops
        sequences[tool_start] = self.tokenizer.tool_call_start
        if tool_end:
            sequences[tool_end] = self.tokenizer.tool_call_end
        return SequenceStateMachine(transitions, initial="normal"), sequences

    def generate(
        self, request: ChatRequest, cancelled: threading.Event
    ) -> Iterator[GeneratedEvent]:
        with self._generation_turn(cancelled) as admitted:
            if not admitted:
                return
            messages = copy.deepcopy(request.messages)
            tools = copy.deepcopy(request.tools)
            process_message_content(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools or None,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=request.enable_thinking,
            )
            full_prompt = list(prompt)
            cache, suffix = self.prompt_cache.fetch_nearest_cache(
                self.model_key, full_prompt
            )
            cached_tokens = len(full_prompt) - len(suffix)
            if cache is None:
                cache = make_prompt_cache(self.model)
            mx.random.seed(request.seed)
            sampler = make_sampler(temp=request.temperature, top_p=request.top_p)
            state_machine, sequences = self._state_machine()
            state = state_machine.make_state()
            context = GenerationContext(
                has_tool_calling=True,
                has_thinking=False,
                tool_parser=self.tokenizer.tool_parser,
                sequences=sequences,
                prompt=full_prompt,
                prompt_cache_count=cached_tokens,
            )
            generated_tokens: list[int] = []
            final_native: dict[str, float | int] = {}

            def native_stream() -> Iterator[Response]:
                nonlocal state
                for generated in stream_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=suffix,
                    max_tokens=request.max_tokens,
                    sampler=sampler,
                    prompt_cache=cache,
                    prefill_step_size=self.prefill_step_size,
                ):
                    generated_tokens.append(generated.token)
                    final_native.update(
                        prompt_tps=generated.prompt_tps,
                        generation_tps=generated.generation_tps,
                        peak_memory=generated.peak_memory,
                        completion_tokens=generated.generation_tokens,
                    )
                    state, matched, current = state_machine.match(
                        state, generated.token
                    )
                    yield Response(
                        text=generated.text,
                        token=generated.token,
                        state=current,
                        match=matched,
                        logprob=0.0,
                        finish_reason=generated.finish_reason,
                        top_tokens=(),
                    )
                    if cancelled.is_set() or generated.finish_reason is not None:
                        break

            tool_text = ""
            previous_state: str | None = None
            made_tool_call = False
            finish_reason = "stop"
            formatter = ToolCallFormatter(
                self.tokenizer.tool_parser, tools, streaming=True
            )
            for generated in _process_control_tokens(context, native_stream()):
                if generated.state == "tool":
                    tool_text += generated.text
                elif generated.state == "normal":
                    if previous_state == "tool" and tool_text:
                        calls = formatter([tool_text])
                        tool_text = ""
                        made_tool_call = bool(calls)
                        if calls:
                            yield GeneratedEvent("tool_calls", calls)
                    if generated.text:
                        yield GeneratedEvent("content", generated.text)
                if generated.finish_reason is not None:
                    finish_reason = generated.finish_reason
                previous_state = generated.state

            if cancelled.is_set():
                return
            if previous_state == "tool" and tool_text:
                calls = formatter([tool_text])
                made_tool_call = bool(calls)
                if calls:
                    yield GeneratedEvent("tool_calls", calls)
            if made_tool_call and finish_reason == "stop":
                finish_reason = "tool_calls"

            cache_key = full_prompt + generated_tokens
            self.prompt_cache.insert_cache(self.model_key, cache_key, cache)
            completion_tokens = int(
                final_native.get("completion_tokens", len(generated_tokens))
            )
            evaluated_tokens = len(suffix)
            prompt_tps = float(final_native.get("prompt_tps", 0.0))
            generation_tps = float(final_native.get("generation_tps", 0.0))
            evidence = NativeEvidence(
                prompt_tokens=len(full_prompt),
                cached_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                prompt_ms=(1000 * evaluated_tokens / prompt_tps)
                if evaluated_tokens and prompt_tps
                else 0.0,
                generation_ms=(1000 * completion_tokens / generation_tps)
                if completion_tokens and generation_tps
                else 0.0,
                peak_memory_bytes=round(
                    float(final_native.get("peak_memory", 0.0)) * 1_000_000_000
                ),
            )
            yield GeneratedEvent("finish", finish_reason)
            yield GeneratedEvent("evidence", evidence)


class MlxGenerationWorker:
    """Owns model loading and every MLX operation on one persistent thread."""

    def __init__(
        self, model_path: str, *, prefill_step_size: int, prompt_cache_entries: int
    ):
        self._model_path = model_path
        self._prefill_step_size = prefill_step_size
        self._prompt_cache_entries = prompt_cache_entries
        self._jobs: queue.Queue[_GenerationJob | None] = queue.Queue()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name="mlx-generation-worker", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise self._startup_error

    def _run(self) -> None:
        try:
            # MLX streams are thread-local. The model and all later evaluation must
            # be created and used by this same long-lived worker.
            mx.default_stream(mx.default_device())
            generator = MlxGenerator(
                self._model_path,
                prefill_step_size=self._prefill_step_size,
                prompt_cache_entries=self._prompt_cache_entries,
            )
        except Exception as error:  # noqa: BLE001 - startup crosses a thread boundary
            self._startup_error = error
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                for event in generator.generate(job.request, job.cancelled):
                    job.emit(event)
            except Exception as error:  # noqa: BLE001 - generation crosses a thread boundary
                job.emit(error)
            finally:
                job.emit(None)

    def submit(
        self,
        request: ChatRequest,
        cancelled: threading.Event,
        emit: Callable[[GeneratedEvent | Exception | None], None],
    ) -> None:
        self._jobs.put(_GenerationJob(request, cancelled, emit))

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join()
