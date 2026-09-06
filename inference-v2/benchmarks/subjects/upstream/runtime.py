"""Load with stock MLX-VLM and measure its text program without engine machinery.

These are neural execution controls, not the upstream server or generate API.
Every decode output consumes exactly one input. The service boundary completes
state before control returns, making bounded and continuous feedback explicit.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.utils import load_model

from benchmarks.contracts import Observation
from magnitude_engine.artifacts.source import LocalArtifact


def tokens(count: int) -> tuple[int, ...]:
    return tuple(1 + index % 16 for index in range(count))


class ModelTrace:
    def __init__(
        self, artifact: LocalArtifact, prefix: tuple[int, ...],
        prefill_tokens: int, cache_limit_bytes: int,
    ):
        if prefill_tokens < 1 or cache_limit_bytes < 0:
            raise ValueError("upstream prefill size must be positive and cache limit nonnegative")
        self.prefix = prefix
        self.prefill_tokens = prefill_tokens
        self.caches: list[Any] = []
        self.model: Any = None
        self.closed = False
        self._previous_cache_limit = mx.set_cache_limit(cache_limit_bytes)
        try:
            # Use the independently maintained upstream loader, including its
            # architecture discovery, sanitization and checkpoint quantization.
            self.model = load_model(Path(artifact.path), lazy=False, strict=True)
            if not hasattr(self.model, "language_model"):
                raise ValueError("upstream control requires a standalone language_model")
        except BaseException:
            self.close()
            raise

    def forward(self, inputs: mx.array) -> mx.array:
        output = self.model.language_model(inputs=inputs, cache=self.caches)
        return output if isinstance(output, mx.array) else output.logits

    def restore_prefix(self) -> None:
        self.complete()
        self.caches = make_prompt_cache(self.model.language_model)
        for start in range(0, len(self.prefix), self.prefill_tokens):
            self.forward(mx.array([self.prefix[start : start + self.prefill_tokens]], mx.int32))
            self.complete()
        mx.reset_peak_memory()

    def complete(self) -> None:
        # This is upstream's public cache state contract, independent of our
        # native-state adapter and transaction/capacity machinery.
        mx.eval([cache.state for cache in self.caches])
        mx.synchronize()

    def close(self) -> None:
        if self.closed:
            return
        self.complete()
        self.caches.clear()
        self.model = None
        mx.set_cache_limit(self._previous_cache_limit)
        self.closed = True


class DecodeTrace(ModelTrace):
    def __init__(
        self, *, artifact: LocalArtifact, prompt_tokens: int, output_tokens: int,
        service_tokens: int, prefill_tokens: int, cache_limit_bytes: int,
    ):
        if min(prompt_tokens, output_tokens, service_tokens) < 1:
            raise ValueError("upstream decode requires positive prompt, output and service sizes")
        prompt = tokens(prompt_tokens)
        self.anchor = prompt[-1]
        self.output_tokens = output_tokens
        self.service_tokens = service_tokens
        self.outputs: list[int] = []
        self.expected: tuple[int, ...] | None = None
        super().__init__(artifact, prompt[:-1], prefill_tokens, cache_limit_bytes)

    def reset(self) -> None:
        self.restore_prefix()
        self.outputs = []

    def invoke(self) -> None:
        anchor = self.anchor

        def predict(token: mx.array) -> mx.array:
            logits = self.forward(token.reshape(1, 1))
            prediction = mx.argmax(logits[0, -1]).astype(mx.int32)
            mx.async_eval(prediction)
            return prediction

        while len(self.outputs) < self.output_tokens:
            count = min(self.service_tokens, self.output_tokens - len(self.outputs))
            current = predict(mx.array(anchor, mx.int32))
            for index in range(count):
                following = predict(current) if index + 1 < count else None
                self.outputs.append(cast(int, current.item()))
                if following is not None:
                    current = following
            self.complete()
            anchor = self.outputs[-1]

    def observe(self) -> Observation:
        actual = tuple(self.outputs)
        if len(actual) != self.output_tokens:
            raise ValueError("upstream decode did not produce its declared output count")
        if self.expected is None:
            self.expected = actual
        if actual != self.expected:
            raise ValueError("upstream decode continuation was not repeatable")
        return Observation(
            hashlib.sha256(json.dumps(self.outputs).encode()).hexdigest(),
            {
                "prompt_tokens": len(self.prefix) + 1,
                "output_tokens": len(actual),
                "evaluated_decode_inputs": len(actual),
                "service_tokens": self.service_tokens,
                "mlx_peak_bytes": mx.get_peak_memory(),
            },
            {"output_tokens": self.outputs},
        )


class PrefillTrace(ModelTrace):
    def __init__(
        self, *, artifact: LocalArtifact, prefix_tokens: int, input_tokens: int,
        prefill_tokens: int, cache_limit_bytes: int,
    ):
        if prefix_tokens < 0 or input_tokens < 1:
            raise ValueError("upstream prefill requires a nonnegative prefix and positive query")
        self.query = tokens(input_tokens)
        self.anchor = 1
        self.expected: mx.array | None = None
        super().__init__(artifact, tokens(prefix_tokens), prefill_tokens, cache_limit_bytes)

    def reset(self) -> None:
        self.restore_prefix()

    def invoke(self) -> None:
        self.forward(mx.array([self.query], mx.int32))

    def observe(self) -> Observation:
        logits = self.forward(mx.array([[self.anchor]], mx.int32))
        if not bool(mx.all(mx.isfinite(logits)).item()):
            raise ValueError("upstream prefill produced nonfinite continuation logits")
        token = cast(int, mx.argmax(logits[0, -1]).item())
        self.complete()
        if self.expected is None:
            self.expected = logits
        if not bool(mx.array_equal(logits, self.expected).item()):
            raise ValueError("upstream prefill continuation was not repeatable")
        return Observation(
            hashlib.sha256(bytes(memoryview(logits.astype(mx.float32)))).hexdigest(),
            {"prefix_tokens": len(self.prefix), "input_tokens": len(self.query)},
            {"continuation_token": token},
        )
