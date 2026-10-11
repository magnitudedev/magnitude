from __future__ import annotations

import contextvars
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator


_request_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "magnitude_omlx_request", default=None
)
_lock = threading.Lock()
_request_ids: dict[str, str] = {}
_metrics: dict[str, "RequestMetrics"] = {}
_mtp_by_uid: dict[Any, Any] = {}
_backend = "none"


@dataclass
class RequestMetrics:
    prompt_ms: float | None = None
    decode_started_at: float | None = None
    generation_ms: float | None = None
    draft_tokens: int | None = None
    accepted_draft_tokens: int | None = None
    fallback: bool = False
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


def _dflash_generation_ms(elapsed_us: float, prefill_us: float) -> float:
    """DFlash summary elapsed time spans prefill and decode; remove prefill."""
    return max(0.0, float(elapsed_us) - float(prefill_us)) / 1000.0


def _record_batched_response_time(
    metric: RequestMetrics,
    now: float,
    *,
    has_uncached_prompt_token: bool,
) -> None:
    if metric.decode_started_at is None:
        metric.decode_started_at = now
    if not metric.first_batch_response_seen:
        # oMLX externally pre-fills tokens[0:N-1]. Its first BatchGenerator
        # step evaluates the last uncached prompt token and produces the first
        # output token, which is the same prompt/generation boundary used by
        # oMLX's native TTFT and generation-duration reporting.
        if has_uncached_prompt_token:
            _add_prompt_time(
                metric, _elapsed_ms(metric.decode_started_at, now)
            )
        metric.decode_started_at = now
        metric.first_batch_response_seen = True
    metric.generation_ms = _elapsed_ms(metric.decode_started_at, now)


def _patch_batched_instrumentation() -> None:
    from omlx.engine.batched import BatchedEngine
    from omlx.engine_core import EngineCore
    from omlx.scheduler import Scheduler

    original_stream = BatchedEngine.stream_generate

    async def batched_stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        explicit_key = kwargs.pop("_magnitude_request_key", None)
        _request_key.set(explicit_key)
        try:
            async for output in original_stream(self, *args, **kwargs):
                yield output
        finally:
            _request_key.set(None)

    BatchedEngine.stream_generate = batched_stream

    original_add = EngineCore.add_request

    async def add_request(self: Any, *args: Any, **kwargs: Any) -> str:
        request_id = await original_add(self, *args, **kwargs)
        key = _request_key.get()
        if key is not None:
            with _lock:
                _request_ids[request_id] = key
                _metrics.setdefault(key, RequestMetrics())
        return request_id

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
                metric.decode_started_at = time.perf_counter()

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
                metric.decode_started_at = time.perf_counter()

    Scheduler._step_prefill_chunk = prefill_chunk

    original_responses = Scheduler._process_batch_responses

    def process_responses(self: Any, responses: list[Any]) -> Any:
        now = time.perf_counter()
        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue
            key = _key_for_request(request_id)
            if key is None:
                continue
            metric = _metric(key)
            request = self.running.get(request_id)
            if (
                metric.prompt_ms is None
                and request is not None
                and int(request.num_prompt_tokens) == int(request.cached_tokens or 0)
            ):
                # A fully cached prompt performs no prompt evaluation.
                metric.prompt_ms = 0.0
            if metric.decode_started_at is None:
                metric.decode_started_at = now
            _record_batched_response_time(
                metric,
                now,
                has_uncached_prompt_token=(
                    int(request.num_prompt_tokens)
                    > int(request.cached_tokens or 0)
                ),
            )
            with _lock:
                stats = _mtp_by_uid.pop(response.uid, None)
            if stats is not None:
                drafted = sum(int(value) for value in stats.depth_drafted)
                metric.draft_tokens = drafted
                metric.accepted_draft_tokens = int(stats.accepts)
        return original_responses(self, responses)

    Scheduler._process_batch_responses = process_responses

    from omlx.patches.mlx_lm_mtp import batch_generator as mtp_batch

    original_log = mtp_batch._log_mtp_stats

    def record_mtp(uid: Any, stats: Any, finish_reason: str) -> None:
        with _lock:
            _mtp_by_uid[uid] = stats
        original_log(uid, stats, finish_reason)

    mtp_batch._log_mtp_stats = record_mtp


def _patch_dflash_instrumentation() -> None:
    from omlx.engine.dflash import DFlashEngine

    original_events = DFlashEngine._stream_dflash_events

    def stream_events(self: Any, *args: Any, **kwargs: Any) -> Any:
        event_iter, prefix_flow, stop_ids = original_events(self, *args, **kwargs)
        key = getattr(self, "_magnitude_request_key", None)

        def observed() -> Any:
            for event in event_iter:
                if key is not None:
                    name = type(event).__name__
                    metric = _metric(key)
                    if name == "PrefillCompleteEvent":
                        metric.prompt_ms = float(event.prefill_us) / 1000.0
                    elif name == "CycleCompleteEvent":
                        drafted = int(
                            event.candidate_count
                            if event.candidate_count is not None
                            else event.block_len
                        )
                        metric.draft_tokens = (metric.draft_tokens or 0) + drafted
                    elif name == "SummaryEvent":
                        metric.accepted_draft_tokens = int(event.accepted_from_draft)
                        phase_timings = getattr(event, "phase_timings_us", {})
                        prefill_us = float(
                            phase_timings.get("prefill", (metric.prompt_ms or 0.0) * 1000.0)
                        )
                        metric.generation_ms = _dflash_generation_ms(
                            event.elapsed_us, prefill_us
                        )
                        metric.fallback = bool(event.fallback_ar)
                yield event

        return observed(), prefix_flow, stop_ids

    DFlashEngine._stream_dflash_events = stream_events

    original_stream = DFlashEngine.stream_generate

    async def dflash_stream(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        self._magnitude_request_key = kwargs.pop(
            "_magnitude_request_key", _request_key.get()
        )
        try:
            async for output in original_stream(self, *args, **kwargs):
                yield output
        finally:
            self._magnitude_request_key = None

    DFlashEngine.stream_generate = dflash_stream


def _terminal_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError("oMLX terminal chunk has no usage object")
    details = usage.get("prompt_tokens_details") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    cached_tokens = int(details.get("cached_tokens", 0))
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
    actual_backend = _backend
    if _backend == "dflash" and metric.fallback:
        actual_backend = "none"
    if actual_backend != "none":
        if metric.draft_tokens is None or metric.accepted_draft_tokens is None:
            raise RuntimeError(f"{actual_backend} did not expose request-local draft counters")
        if metric.accepted_draft_tokens > metric.draft_tokens:
            raise RuntimeError(
                f"{actual_backend} accepted more draft tokens than it drafted"
            )
        timings.update(
            {
                "draft_n": metric.draft_tokens,
                "draft_n_accepted": metric.accepted_draft_tokens,
                "speculative_backend": actual_backend,
            }
        )
    return {
        **{key: value for key, value in payload.items() if key not in {"usage", "timings"}},
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
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
                if isinstance(payload, dict) and payload.get("choices") == [] and "usage" in payload:
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


def install_instrumentation(speculative_backend: str) -> None:
    global _backend
    if speculative_backend not in {"none", "mtp", "dflash", "dspark"}:
        raise ValueError(f"unsupported backend: {speculative_backend}")
    _backend = speculative_backend
    _patch_batched_instrumentation()
    _patch_dflash_instrumentation()
    _patch_streaming_response()
