from __future__ import annotations

import contextvars
import json
import math
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

_request_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "magnitude_omlx_request", default=None
)
_lock = threading.Lock()
_request_ids: dict[str, str] = {}
_metrics: dict[str, RequestMetrics] = {}


@dataclass
class RequestMetrics:
    prompt_ms: float | None = None
    generation_ms: float | None = None
    first_batch_response_seen: bool = False


def _metric(key: str) -> RequestMetrics:
    with _lock:
        return _metrics.setdefault(key, RequestMetrics())


def _key_for_request(request_id: str) -> str | None:
    with _lock:
        return _request_ids.get(request_id)


def _elapsed_ms(started: float, finished: float) -> float:
    """Convert the native monotonic timer's seconds to milliseconds."""
    return max(0.0, finished - started) * 1000.0


def _add_prompt_time(metric: RequestMetrics, elapsed_ms: float) -> None:
    metric.prompt_ms = (metric.prompt_ms or 0.0) + elapsed_ms


def _record_batched_service_time(
    metric: RequestMetrics,
    elapsed_ms: float,
    *,
    has_uncached_prompt_token: bool,
) -> None:
    if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
        raise RuntimeError("oMLX returned an invalid native decode-step duration")
    if not metric.first_batch_response_seen:
        # oMLX externally pre-fills tokens[0:N-1]. Its first BatchGenerator
        # step evaluates the last uncached prompt token and produces the first
        # output token, which is the same prompt/generation boundary used by
        # oMLX's native TTFT and generation-duration reporting.
        if has_uncached_prompt_token:
            _add_prompt_time(metric, elapsed_ms)
        metric.first_batch_response_seen = True
        metric.generation_ms = 0.0
    else:
        metric.generation_ms = (metric.generation_ms or 0.0) + elapsed_ms


def _bind_request_stream(original: Any) -> Any:
    async def stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        # The HTTP keepalive wrapper can resume the generator in another task.
        # Carry identity explicitly to the engine entry, including VLM text calls.
        explicit_key = kwargs.pop("_magnitude_request_key", _request_key.get())
        _request_key.set(explicit_key)
        try:
            async for output in original(self, *args, **kwargs):
                yield output
        finally:
            _request_key.set(None)

    return stream


def _patch_batched_instrumentation() -> None:
    from omlx.engine.batched import BatchedEngine
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.engine_core import EngineCore
    from omlx.scheduler import Scheduler

    BatchedEngine.stream_generate = _bind_request_stream(BatchedEngine.stream_generate)
    VLMBatchedEngine.stream_generate = _bind_request_stream(VLMBatchedEngine.stream_generate)

    original_add = EngineCore.add_request

    async def add_request(
        self: Any,
        prompt: Any,
        sampling_params: Any = None,
        request_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        request_id = request_id or str(uuid.uuid4())
        key = _request_key.get()
        if key is not None:
            with _lock:
                _request_ids[request_id] = key
                _metrics.setdefault(key, RequestMetrics())
        try:
            # Register before admission can wake the scheduler on its other thread.
            return await original_add(self, prompt, sampling_params, request_id, *args, **kwargs)
        except BaseException:
            with _lock:
                _request_ids.pop(request_id, None)
            raise

    EngineCore.add_request = add_request

    original_prefill = Scheduler._do_external_prefill

    def external_prefill(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_prefill(self, request, *args, **kwargs)
        finally:
            key = _key_for_request(request.request_id)
            if key is not None:
                metric = _metric(key)
                _add_prompt_time(metric, _elapsed_ms(started, time.perf_counter()))

    Scheduler._do_external_prefill = external_prefill

    original_chunk = Scheduler._step_prefill_chunk

    def prefill_chunk(self: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_chunk(self, state, *args, **kwargs)
        finally:
            request = state.request
            key = _key_for_request(request.request_id)
            if key is not None:
                metric = _metric(key)
                _add_prompt_time(metric, _elapsed_ms(started, time.perf_counter()))

    Scheduler._step_prefill_chunk = prefill_chunk

    original_decode_debt = Scheduler._repay_decode_debt

    def decode_debt(self: Any, decode_seconds: float) -> Any:
        # Pinned Scheduler.step supplies its native next_generated()/VLM pass
        # duration here, after external prefills and before response dispatch.
        self._magnitude_decode_ms = decode_seconds * 1000.0
        return original_decode_debt(self, decode_seconds)

    Scheduler._repay_decode_debt = decode_debt

    original_responses = Scheduler._process_batch_responses

    def process_responses(self: Any, responses: list[Any]) -> Any:
        elapsed_ms = getattr(self, "_magnitude_decode_ms", None)
        if elapsed_ms is None:
            raise RuntimeError("oMLX response has no native decode-step duration")
        del self._magnitude_decode_ms
        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue
            key = _key_for_request(request_id)
            if key is None:
                continue
            metric = _metric(key)
            request = self.running.get(request_id)
            if request is None:
                raise RuntimeError("oMLX response has no running request for native timing")
            if (
                metric.prompt_ms is None
                and int(request.num_prompt_tokens) == int(request.cached_tokens or 0)
            ):
                # A fully cached prompt performs no prompt evaluation.
                metric.prompt_ms = 0.0
            _record_batched_service_time(
                metric,
                elapsed_ms,
                has_uncached_prompt_token=(
                    int(request.num_prompt_tokens) > int(request.cached_tokens or 0)
                ),
            )
        return original_responses(self, responses)

    Scheduler._process_batch_responses = process_responses


def _terminal_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("oMLX terminal chunk has no usage object")
    details = usage["prompt_tokens_details"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    cached_tokens = details["cached_tokens"]
    if any(type(n) is not int for n in (prompt_tokens, completion_tokens, cached_tokens)):
        raise RuntimeError("oMLX terminal counts are not integers")
    if usage["total_tokens"] != prompt_tokens + completion_tokens:
        raise RuntimeError("oMLX native total disagrees with its prompt and completion counts")
    if prompt_tokens < 0 or completion_tokens < 0 or not 0 <= cached_tokens <= prompt_tokens:
        raise RuntimeError("oMLX terminal chunk has inconsistent token counts")
    metric = _metric(key)
    prompt_ms = metric.prompt_ms
    generation_ms = metric.generation_ms
    if prompt_ms is None:
        raise RuntimeError("oMLX did not expose request-local prompt evaluation duration")
    if generation_ms is None:
        raise RuntimeError("oMLX did not expose request-local generation duration")
    if not all(math.isfinite(value) and value >= 0 for value in (prompt_ms, generation_ms)):
        raise RuntimeError("oMLX returned invalid native durations")

    timings: dict[str, Any] = {
        "cache_n": cached_tokens,
        "prompt_n": prompt_tokens - cached_tokens,
        "prompt_ms": prompt_ms,
        "predicted_n": completion_tokens,
        "predicted_ms": generation_ms,
    }
    return {
        **{key: value for key, value in payload.items() if key not in {"usage", "timings"}},
        "choices": [],
        "usage": usage,
        "timings": timings,
    }


def _patch_streaming_response() -> None:
    import omlx.server as server

    original = server.stream_chat_completion

    async def stream_chat_completion(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        key = uuid.uuid4().hex
        _request_key.set(key)
        kwargs["_magnitude_request_key"] = key
        try:
            async for frame in original(*args, **kwargs):
                if not frame.startswith("data: ") or frame.startswith("data: [DONE]"):
                    yield frame
                    continue
                try:
                    payload = json.loads(frame[6:].strip())
                except json.JSONDecodeError:
                    yield frame
                    continue
                if (
                    isinstance(payload, dict)
                    and payload.get("choices") == []
                    and "usage" in payload
                ):
                    terminal = _terminal_payload(payload, key)
                    yield f"data: {json.dumps(terminal, separators=(',', ':'))}\n\n"
                else:
                    yield frame
        finally:
            # Starlette may resume an async generator in a copied Context. A token
            # may only be reset in the exact Context that created it, while setting
            # the request-local default is safe in either context.
            _request_key.set(None)
            with _lock:
                request_ids = [rid for rid, mapped in _request_ids.items() if mapped == key]
                for request_id in request_ids:
                    _request_ids.pop(request_id, None)
                _metrics.pop(key, None)

    server.stream_chat_completion = stream_chat_completion


def install_instrumentation() -> None:
    _patch_batched_instrumentation()
    _patch_streaming_response()
